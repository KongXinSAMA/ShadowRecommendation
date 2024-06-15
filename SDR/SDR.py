import Argparser
import itertools
import torch.nn as nn
from Utils.Utils import *
from Utils.BaseModel import SBR, AffineTransform, RealNVP
device = "cuda" if torch.cuda.is_available() else "cpu"

def Evaluate(data_loader, test_model, user_ivae_mean, real_nvp_s0, real_nvp_s1, P_Oeq1, P_Oeq0, OR_tilde, device="cpu", k=5, val=False):
    test_model.eval()
    with torch.no_grad():
        uids, iids, predicts, labels = list(), list(), list(), list()
        for index, (uid, iid, rating) in enumerate(data_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            User_shadow = user_ivae_mean[uid.type(torch.long)]
            predict = test_model(uid, iid, User_shadow).view(-1)
            if val == True:
                Ratio = torch.exp(real_nvp_s1.log_prob(User_shadow)) / torch.exp(real_nvp_s0.log_prob(User_shadow))
                predict = predict / (P_Oeq1 + OR_tilde[1] * Ratio * P_Oeq0)
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
    train_loader, val_loader, test_loader, n_users, n_items, P_Oeq1, P_Oeq0, y_unique, OR_tilde = construct_or_dataloader(config, device)

    P_Oeq1, P_Oeq0, y_unique, OR_tilde = torch.Tensor(P_Oeq1).to(device), torch.Tensor(P_Oeq0).to(device), torch.Tensor(y_unique).to(device), torch.Tensor(OR_tilde).to(device)

    user_ivae_mean = torch.load(data_params["Shadow_path"] + "user_mean.pt", map_location='cpu').to(device)

    inp_size = user_ivae_mean.shape[1]
    layer_dim = data_params['dense_layer_dim']
    layer_num = data_params['dense_layer_num']
    real_nvp_s1 = RealNVP([AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)]).to(device)

    real_nvp_s0 = RealNVP([AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)]).to(device)

    real_nvp_s1.load_state_dict(torch.load(data_params["Shadow_path"] + "Best_S1_UDensety_Estimator.pt"))
    real_nvp_s0.load_state_dict(torch.load(data_params["Shadow_path"] + "Best_S0_UDensety_Estimator.pt"))

    model = SBR(n_users, n_items, config["embedding_dim"], U_Shadow_size=data_params["u_shadow_dim"], init=data_params["init"], device=device).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    mf_loss_function = nn.MSELoss()

    patience = config["patience"]
    best_val = (0, 0)
    test_performance = (0, 0)
    patience_count = 0
    real_nvp_s1.eval()
    real_nvp_s0.eval()
    for epoch in range(config["epochs"]):
        total_loss = 0
        total_len = 0
        model.train()
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            User_shadow = user_ivae_mean[uid.type(torch.long)]
            preds = model(uid, iid, User_shadow)
            Ratio = torch.exp(real_nvp_s1.log_prob(User_shadow) - real_nvp_s0.log_prob(User_shadow))
            Adjustment = (P_Oeq1 + OR_tilde[1] * Ratio * P_Oeq0)
            unbias_preds = preds / Adjustment
            loss = mf_loss_function(rating, unbias_preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)
            total_len += len(rating)

        train_loss = total_loss / total_len

        model.eval()
        validation_performance = Evaluate(val_loader, model, user_ivae_mean, real_nvp_s0, real_nvp_s1, P_Oeq1, P_Oeq0, OR_tilde, val=True, device=device)
        test = Evaluate(test_loader, model, user_ivae_mean, real_nvp_s0, P_Oeq1, P_Oeq0, OR_tilde, real_nvp_s1, device=device)
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

# lisT = []
if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")
        if args.data_params["name"] == "kuai_rand":
            search_grid = {'lr': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6], 'weight_decay': [1e-5, 1e-6],
                           'embedding_dim': [512, 1024]}
        else:
            search_grid = {'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6],
                'embedding_dim': [256, 512]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [0, 0]
        random_seed = random.randint(1000, 2000)
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            Train_eval_config = {
                "tune": True,
                "seed": random_seed,
                "epochs": 120,
                "show_log": False,
                "patience": 5,
                "lr": supparams['lr'],
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
            test_param = {'lr': 1e-03, 'weight_decay': 1e-05, 'embedding_dim': 512}
        elif args.data_params["name"] == 'yahoo':
            test_param = {'lr': 5e-05, 'weight_decay': 1e-05, 'embedding_dim': 256}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 5e-06, 'weight_decay': 1e-06, 'embedding_dim': 1024}
        elif args.data_params["name"] == 'sim_data':
            test_param = {'lr': 0.00005, 'weight_decay': 1e-05, 'embedding_dim': 512}
        lisT = []
        for i in range(10):
            random_seed = random.randint(1000, 2000)
            Train_eval_config = {
                "tune": False,
                "seed": random_seed,
                "epochs": 120,
                "show_log": True,
                "patience": 5,
                "lr": test_param['lr'],
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
            "tune": False,
            "seed": 1208,
            "epochs": 120,
            "show_log": True,
            "patience": 5,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "embedding_dim": 512,
            "batch_size": 512,
            "data_params": args.data_params,
        }
        _, result= train_eval(Train_eval_config)
        print(result)
