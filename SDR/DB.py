import Argparser
import itertools
import torch.nn as nn
from Utils.Utils import *
from Utils.BaseModel import MF
device = "cuda" if torch.cuda.is_available() else "cpu"

def Evaluate(data_loader, test_model, device="cpu", k=5):
    test_model.eval()
    with torch.no_grad():
        uids, iids, predicts, labels = list(), list(), list(), list()
        for index, (uid, iid, rating) in enumerate(data_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            predict = test_model(uid, iid).view(-1)
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

def train_eval(config):
    seed_everything(config['seed'])

    train_loader, val_loader, test_loader, n_users, n_items, y_unique, InvP = construct_ips_dataloader(config, device)

    y_unique = torch.Tensor(y_unique)
    InvP = torch.Tensor(InvP)

    seed_everything(config["seed"])

    model = MF(n_users, n_items, config["embedding_dim"]).to(device)
    imputation_model = MF(n_users, n_items, config["impu_embedding_dim"]).to(device)

    optimizer_base = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optimizer_impu = torch.optim.Adam(params=imputation_model.parameters(), lr=config["impu_lr"], weight_decay=config["impu_weight_decay"])
    none_loss_function = nn.MSELoss(reduction='none')
    mean_loss_function = nn.MSELoss(reduction='mean')

    best_val = (0, 0)
    patience_count = 0
    patience = config["patience"]
    test_performance = (0, 0)

    for epoch in range(config["epochs"]):

        total_loss = 0
        total_len = 0
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)

            weight = torch.ones_like(rating).to(device)
            for i in range(len(y_unique)):
                weight[rating == y_unique[i]] = InvP[iid.type(torch.long)][rating == y_unique[i], i]

            # imputation
            imputation_model.train()
            e_hat = imputation_model(uid, iid)
            e = rating - model(uid, iid)
            cost_e = none_loss_function(e_hat, e)
            loss_imp = torch.sum(weight * cost_e)
            optimizer_impu.zero_grad()
            loss_imp.backward()
            optimizer_impu.step()
            # MF
            model.train()

            all_u, all_i = torch.unique(uid), torch.unique(iid)
            all_pair = torch.cartesian_prod(all_u, all_i).to(device)
            users_all, items_all = all_pair[:, 0], all_pair[:, 1]
            y_hat_all = model(users_all, items_all)
            y_hat_all_detach = torch.detach(y_hat_all)
            g_all = imputation_model(users_all, items_all)
            loss_all = mean_loss_function(y_hat_all, g_all + y_hat_all_detach)

            y_hat_obs = model(uid, iid)
            y_hat_obs_detach = torch.detach(y_hat_obs)
            g_obs = imputation_model(uid, iid)
            e_obs = none_loss_function(y_hat_obs, rating)
            e_hat_obs = none_loss_function(y_hat_obs, g_obs + y_hat_obs_detach)
            cost_obs = e_obs - e_hat_obs
            loss_obs = torch.mean(weight * cost_obs)
            loss_base = loss_all + loss_obs

            optimizer_base.zero_grad()
            loss_base.backward()
            optimizer_base.step()

            total_loss += loss_base.item() * len(rating)
            total_len += len(rating)

        train_loss = total_loss / total_len
        model.eval()
        validation_performance = Evaluate(val_loader, model, device=device)
        test = Evaluate(test_loader, model, device=device)
        if config['show_log']:
            print(train_loss, validation_performance, test)
        patience_count += 1
        if validation_performance[0] > best_val[0]:
            patience_count = 0
            best_val = validation_performance
            test_performance = test
            # torch.save(model.state_dict(), "Best_Shadow_Rec.pt")

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
        print("--tune model--")
        if args.data_params["name"] == "kuai_rand":
            search_grid = {'lr': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6], 'weight_decay': [1e-5, 1e-6],
                           'embedding_dim': [512, 1024], 'impu_lr': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                "impu_weight_decay": [1e-5, 1e-6], "impu_embedding_dim": [512, 1024]}
        else:
            search_grid = {'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6],
                'embedding_dim': [256, 512], 'impu_lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                "impu_weight_decay": [1e-5, 1e-6], "impu_embedding_dim": [256, 512]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [0, 0]
        random_seed = random.randint(1000, 2000)
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            Train_eval_config = {
                "seed": random_seed,
                "epochs": 120,
                "show_log": False,
                "patience": 5,
                "lr": supparams['lr'],
                "weight_decay": supparams['weight_decay'],
                "embedding_dim": supparams['embedding_dim'],
                "impu_lr": supparams['impu_lr'],
                "impu_weight_decay": supparams['impu_weight_decay'],
                "impu_embedding_dim": supparams['impu_embedding_dim'],
                "batch_size": 512,
                'all_batch_size': 256,
                "data_params": args.data_params,
            }
            print("index:", index, "index:", supparams)
            result, _ = train_eval(Train_eval_config)
            if result[0] > best_val[0]:
                best_val = result
                best_param = supparams
        print(best_val, best_param)
    elif args.test:
        test_param = None
        if args.data_params["name"] == 'coat':
            test_param = {'lr': 5e-4, 'weight_decay': 1e-05, 'embedding_dim': 256, "impu_lr": 0.0005,
                          "impu_weight_decay": 5e-5,"impu_embedding_dim": 256}
        elif args.data_params["name"] == 'yahoo':
            test_param = {'lr': 5e-4, 'weight_decay': 1e-05, 'embedding_dim': 256, "impu_lr": 0.0005,
                          "impu_weight_decay": 5e-5,"impu_embedding_dim": 256}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 5e-4, 'weight_decay': 1e-05, 'embedding_dim': 256, "impu_lr": 0.0005,
                          "impu_weight_decay": 5e-5,"impu_embedding_dim": 256}
        elif args.data_params["name"] == 'sim_data':
            test_param = {'lr': 0.0005, 'weight_decay': 1e-05, 'embedding_dim': 256, "impu_lr": 5e-05,
                          "impu_weight_decay": 1e-06,"impu_embedding_dim": 512}
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
                "impu_lr": test_param['impu_lr'],
                "impu_weight_decay": test_param['impu_weight_decay'],
                "impu_embedding_dim": test_param['impu_embedding_dim'],
                "batch_size": 512,
                'all_batch_size': 4096,
                "data_params": args.data_params,
            }
            _, result = train_eval(Train_eval_config)
            lisT.append(result)
        mean = np.mean(lisT, 0)
        std = np.std(lisT, 0)
        print("{} ± {}  {} ± {}".format(mean[0], std[0], mean[1], std[1]))
    else:
        Train_eval_config = {
            "seed": 1208,
            "epochs": 120,
            "show_log": True,
            "patience": 5,
            "lr": 0.0001,
            "weight_decay": 1e-5,
            "embedding_dim": 256,
            "impu_lr": 0.0005,
            "impu_weight_decay": 1e-5,
            "impu_embedding_dim": 256,
            "batch_size": 512,
            'all_batch_size': 4096,
            "data_params": args.data_params,
        }
        _, result= train_eval(Train_eval_config)
        print(result)