import Argparser
import itertools
import torch.nn as nn
from Utils.Utils import *
from Utils.BaseModel import RD_IPS_MF

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
    data_params = config["data_params"]

    train_loader, val_loader, test_loader, n_users, n_items, y_unique, InvP = construct_ips_dataloader(config, device)

    y_unique = torch.Tensor(y_unique)
    InvP = torch.Tensor(InvP)
    lower_bound = torch.ones_like(InvP) + (InvP - torch.ones_like(InvP)) / (torch.ones_like(InvP) * config["Gama"])
    upper_bound = torch.ones_like(InvP) + (InvP - torch.ones_like(InvP)) * (torch.ones_like(InvP) * config["Gama"])

    seed_everything(config["seed"])

    model = RD_IPS_MF(n_users, n_items, config["embedding_dim"], corY=y_unique,upBound=upper_bound, lowBound=lower_bound, InverP=InvP, device=device).to(device)

    Ips_parameters, Base_parameters = [], []
    for param_name, p in model.named_parameters():
        if (param_name in ['invP.weight']):
            Ips_parameters += [p]
        else:
            Base_parameters += [p]
    optimizer_base = torch.optim.Adam(params=Base_parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    optimizer_ips = torch.optim.Adam(params=Ips_parameters, lr=config["ips_lr"], weight_decay=0)

    mf_loss_function = nn.MSELoss(reduction='none')

    best_val = (0, 0)
    patience_count = 0
    patience = config["patience"]
    test_performance = (0, 0)

    for epoch in range(config["epochs"]):
        model.train()
        if config["ips_freq"] != 0 and (epoch + 1) % config["ips_freq"] == 0:
            for index, (uid, iid, rating) in enumerate(train_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
                ips_loss = model.ips_loss(uid, iid, rating, mf_loss_function)
                optimizer_ips.zero_grad()
                ips_loss.backward()
                optimizer_ips.step()
                model.update_ips()
        total_loss = 0
        total_len = 0
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            loss = model.model_ips_loss(uid, iid, rating, mf_loss_function)
            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()
            total_loss += loss.item() * len(rating)
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
        print("--tune model--")#
        if args.data_params["name"] == "kuai_rand":
            search_grid = {'lr': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6], 'weight_decay': [1e-5, 1e-6],
                           'embedding_dim': [512, 1024], "ips_lr": [0.001, 0.01],
                           "ips_freq": [1, 2, 3], "Gama": [2, 4, 6, 8]}
        else:
            search_grid = {'lr': [5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6],
                           'embedding_dim': [256, 512], "ips_lr": [0.001, 0.01],
                           "ips_freq": [1, 2, 3], "Gama": [2, 4]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [0, 0]
        random_seed = random.randint(1000, 2000)
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            Train_eval_config = {
                "seed": random_seed,
                "epochs": 120,
                "show_log": False,
                "patience": 2,
                "Gama": supparams['Gama'],
                "ips_lr": supparams['ips_lr'],
                "lr": supparams['lr'],
                "ips_freq": supparams['ips_freq'],
                "weight_decay": supparams['weight_decay'],
                "embedding_dim": supparams['embedding_dim'],
                "batch_size": 512,
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
            test_param = {'lr': 5e-04, 'weight_decay': 1e-05, 'embedding_dim': 512,
                          'ips_lr': 0.01, 'ips_freq': 3, 'Gama': 2}
        elif args.data_params["name"] == 'yahoo':
            test_param = {'lr': 1e-04, 'weight_decay': 1e-05, 'embedding_dim': 512,
                          'ips_lr': 0.01, 'ips_freq': 3, 'Gama': 2}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 1e-05, 'weight_decay': 1e-06, 'embedding_dim': 512,
                          'ips_lr': 0.01, 'ips_freq': 3, 'Gama': 6}
        elif args.data_params["name"] == 'sim_data':
            test_param = {'lr': 1e-04, 'weight_decay': 1e-06, 'embedding_dim': 256,
                          'ips_lr': 0.01, 'ips_freq': 2, 'Gama': 4}
        lisT = []
        for i in range(10):
            random_seed = random.randint(1000, 2000)
            Train_eval_config = {
                "seed": random_seed,
                "epochs": 120,
                "show_log": True,
                "patience": 5,
                "Gama": test_param['Gama'],
                "ips_lr": test_param['ips_lr'],
                "lr": test_param['lr'],
                "ips_freq": test_param['ips_freq'],
                "weight_decay": test_param['weight_decay'],
                "embedding_dim": test_param['embedding_dim'],
                "batch_size": 512,
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
            "Gama": 2,
            "ips_lr": 0.001,
            "lr": 0.001,
            "ips_freq": 3,
            "weight_decay": 1e-5,
            "embedding_dim": 512,
            "batch_size": 512,
            "data_params": args.data_params,
        }
        _, result= train_eval(Train_eval_config)
        print(result)