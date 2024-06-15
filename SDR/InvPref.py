import math
import copy
import itertools
from torch import nn
from Utils.Utils import *
from Utils.BaseModel import InvPref
from torch.utils.data import DataLoader
import Argparser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def Evaluate(data_loader, test_model, device="cpu", k=5):
    test_model.eval()
    with torch.no_grad():
        uids, iids, predicts, labels = list(), list(), list(), list()
        for index, (uid, iid, rating, i) in enumerate(data_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            predict = test_model.predict(uid, iid).view(-1)
            uids.extend(uid.cpu())
            iids.extend(iid.cpu())
            predicts.extend(predict.cpu())
            labels.extend(rating.cpu())
        predict_matrix = -np.inf * np.ones((max(uids) + 1, max(iids) + 1))
        predict_matrix[uids, iids] = predicts
        label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids)))).toarray()
        ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=k).mean().item()
        recall = Recall_at_k_batch(predict_matrix, label_matrix, k=k).mean()
        return ndcg, recall

def update_dataloader(original, new_envs):
    dataset = copy.deepcopy(original.dataset)
    dataset.envs = new_envs
    new_random_loader = DataLoader(dataset, original.batch_size, shuffle=True, num_workers=original.num_workers,
                                   pin_memory=original.pin_memory)
    new_sequential_loader = DataLoader(dataset, original.batch_size, shuffle=True, num_workers=original.num_workers,
                                       pin_memory=original.pin_memory)
    return new_random_loader, new_sequential_loader


def train_eval(config):
    n_envs = config["num_envs"]
    inv_coe = config["inv_coe"]
    env_coe = config["env_coe"]
    cluster_interval = config["cluster_interval"]

    random_train_loader, val_loader, test_loader, n_users, n_items = construct_mf_dataloader(config, require_index=True)
    envs = torch.randint(n_envs, size=(len(random_train_loader.dataset),)).to(DEVICE)
    sequential_train_loader = DataLoader(random_train_loader.dataset, random_train_loader.batch_size, shuffle=False,
                                         num_workers=random_train_loader.num_workers,
                                         pin_memory=random_train_loader.pin_memory)
    seed_everything(config["seed"])

    model = InvPref(n_users, n_items, n_envs, config["embedding_dim"]).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_func = nn.MSELoss()
    loss_func_clf = nn.NLLLoss()
    cluster_distance_func = nn.MSELoss(reduction="none")

    best_val = (0, 0)
    patience_count = 0
    patience = config["patience"]
    test_performance = (0, 0)

    batch_num = math.ceil(len(sequential_train_loader.dataset) / config["batch_size"])
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_len = 0
        for batch_index, (uid, iid, y, i) in enumerate(random_train_loader):
            uid, iid, y, i = uid.to(DEVICE), iid.to(DEVICE), y.to(DEVICE), i.to(DEVICE)
            e = envs[i]
            p = float(batch_index + (epoch + 1) * batch_num) / float((epoch + 1) * batch_num)
            alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            inv_score, env_score, env_out = model(uid, iid, e, alpha)

            inv_loss = loss_func(inv_score, y)
            env_loss = loss_func(env_score, y)
            clf_loss = loss_func_clf(env_out, e)

            loss = inv_loss * inv_coe + env_loss * env_coe + clf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_len += len(y)

        train_loss = total_loss / total_len
        model.eval()
        if (epoch + 1) % cluster_interval == 0:
            new_envs = []
            for index, (uid, iid, y, _) in enumerate(sequential_train_loader):
                uid, iid, y, = uid.to(DEVICE), iid.to(DEVICE), y.to(DEVICE)
                all_distances = []
                for env in range(n_envs):
                    env_tensor = torch.full((uid.shape[0],), env).to(DEVICE)
                    _, env_score, _ = model(uid, iid, env_tensor, 0)
                    distances = cluster_distance_func(env_score, y)
                    all_distances.append(distances.view(-1, 1))

                env_distances = torch.cat(all_distances, dim=1)
                new_envs_batch = torch.argmin(env_distances, dim=1)
                new_envs.append(new_envs_batch)
            envs = torch.cat(new_envs, dim=0).to(DEVICE)

        validation_performance = Evaluate(val_loader, model, device=DEVICE)
        test = Evaluate(test_loader, model, device=DEVICE)
        if config['show_log']:
            print(train_loss, validation_performance, test)
        patience_count += 1
        if validation_performance[0] > best_val[0]:
            patience_count = 0
            best_val = validation_performance
            test_performance = test

        if patience_count > patience:
            if config['show_log']:
                print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

    print("best val performance = {0}, test performance is {1}".format(best_val, test_performance))

    return list(best_val), list(test_performance)


if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")# [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        if args.data_params["name"] == "kuai_rand":
            search_grid = {'lr': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6], 'weight_decay': [1e-5, 1e-6],
                           'embedding_dim': [512, 1024], "num_envs": [4, 6, 8], "cluster_interval": [2, 3, 4]}
        else:
            search_grid = {'lr': [5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6],
                'embedding_dim': [256, 512], "num_envs": [4, 6, 8], "cluster_interval": [2, 3, 4]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [0, 0]
        random_seed = random.randint(1000, 2000)
        print("tune seed:", random_seed)
        for index, combination in enumerate(combinations):
            inv_coe = np.random.uniform(0.01, 10)
            env_coe = np.random.uniform(0.01, 10)
            supparams = dict(zip(search_grid.keys(), combination))
            Train_eval_config = {
                "seed": random_seed,
                "epochs": 120,
                "show_log": False,
                "patience": 5,
                "lr": supparams['lr'],
                "weight_decay": supparams['weight_decay'],
                "embedding_dim": supparams['embedding_dim'],
                "batch_size": 512,
                "num_envs": supparams['num_envs'],
                "inv_coe": inv_coe,
                "env_coe": env_coe,
                "cluster_interval": supparams['cluster_interval'],
                "data_params": args.data_params,
            }
            print("index:", index, "index:", supparams, "inv_coe", Train_eval_config["inv_coe"],"env_coe", Train_eval_config["env_coe"])
            result, _ = train_eval(Train_eval_config)
            if result[0] > best_val[0]:
                best_val = result
                best_param = supparams
        print(best_val, best_param)
    elif args.test:
        test_param = None
        if args.data_params["name"] == 'coat':
            test_param = {'lr': 5e-04, 'weight_decay': 1e-06, 'embedding_dim': 512, "num_envs": 4,
                          "inv_coe": 4.964938856429313, "env_coe": 5.585819157533456, "cluster_interval": 2}
        elif args.data_params["name"] == 'yahoo':
            test_param = {'lr': 1e-04, 'weight_decay': 1e-06, 'embedding_dim': 512, "num_envs": 4,
                          "inv_coe": 1.7033849755015773, "env_coe": 9.098100207266121, "cluster_interval": 2}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 5e-05, 'weight_decay': 1e-05, 'embedding_dim': 512, "num_envs": 8,
                          "inv_coe": 9.4056259844272, "env_coe": 9.602621604513832, "cluster_interval": 2}
        elif args.data_params["name"] == 'sim_data':
            test_param = {'lr': 5e-4, 'weight_decay': 1e-05, 'embedding_dim': 256, "num_envs": 6,
                          "inv_coe": 9.21513068099942, "env_coe": 0.6747268785839295, "cluster_interval": 4}
        lisT = []
        for i in range(10):
            random_seed = random.randint(1000, 2000)
            Train_eval_config = {
                "seed": random_seed,
                "epochs": 120,
                "show_log": True,
                "patience": 5,
                "lr": test_param['lr'],
                "weight_decay": test_param['weight_decay'],
                "embedding_dim": test_param['embedding_dim'],
                "batch_size": 512,
                "num_envs": test_param['num_envs'],
                "inv_coe": test_param['inv_coe'],
                "env_coe": test_param['env_coe'],
                "cluster_interval": test_param['cluster_interval'],
                "data_params": args.data_params,
            }
            _, result = train_eval(Train_eval_config)
            lisT.append(result)
        mean = np.mean(lisT, 0)
        std = np.std(lisT, 0)
        print("{} ± {}  {} ± {}".format(mean[0], std[0], mean[1], std[1]))
    else:
        Train_eval_config = {
            "seed": 1258,
            "epochs": 120,
            "show_log": True,
            "patience": 5,
            "lr": 5e-5,
            "weight_decay": 1e-5,
            "embedding_dim": 512,
            "batch_size": 512,
            "num_envs": 8,
            "inv_coe": 9.4056259844272,
            "env_coe": 9.602621604513832,
            "cluster_interval": 2,
            "data_params": args.data_params,
        }
        _, result= train_eval(Train_eval_config)
        print(result)
