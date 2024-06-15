import Argparser
import itertools
from torch.optim import Adam
from torch import nn
from Utils.Utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from Utils.BaseModel import ShadowConstructor
from Utils.Dataloader import Shadow_TrainingDataset
device = "cuda" if torch.cuda.is_available() else "cpu"

def iVae_loss_function(x, x_hat, mean, log_var, prior_mean, log_prior_var, anneal=1.):
    reproduction_loss = torch.mean(
        torch.sum(nn.functional.binary_cross_entropy(x_hat, x, reduction="none"), dim=1))

    kld_loss = -0.5 * torch.mean(
        torch.sum(
            1 + log_var - log_prior_var - ((mean - prior_mean).pow(2) + log_var.exp()) / log_prior_var.exp(),
            dim=1)
    )

    return reproduction_loss + kld_loss * anneal

def Q_Test_loss_function(Q, O):
    D = Q * O
    loss = torch.mean(torch.sum(nn.functional.binary_cross_entropy(D, O.float(), reduction="none"), dim=1))
    return loss

def user_shadow_train_eval(config):
    seed_everything(config["seed"])
    # Read Data
    data_params = config["data_params"]
    fullmatrix, train_matrix, val_matrix, train_user_index, test_user_index = construct_user_dataset(
                                                                                data_params["train_path"],
                                                                                train_ratio=data_params["train_ratio"])

    user_feat = pd.read_csv(data_params["user_feature_label"]).to_numpy()
    user_feat = torch.tensor(user_feat, dtype=torch.float).to(device)
    train_Y_data = torch.tensor(train_matrix >= data_params['threshold'] + 1, dtype=torch.float).to(device)
    train_data = torch.tensor(train_matrix > 0, dtype=torch.float).to(device)
    val_Y_data = torch.tensor(val_matrix >= data_params['threshold'] + 1, dtype=torch.float).to(device)
    val_data = torch.tensor(val_matrix > 0, dtype=torch.float).to(device)


    train_dataset = Shadow_TrainingDataset(Y_data=train_Y_data, O_data=train_data,
                                             Feature=user_feat[train_user_index])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Creat model
    layer_dim = config["u_layer_dim"]
    shadow_dim = config["u_shadow_dim"]
    user_feat_dim = user_feat.shape[1]
    input_size = train_matrix.shape[1]
    prior_mean = True if data_params['name'] == 'sim_data' else False
    model = ShadowConstructor(input_dim=input_size,
                 feature_dim=user_feat_dim,
                 shadow_dim=shadow_dim,
                 layer_dim=layer_dim,
                 n_layers=config["n_layers"],
                 device = device,
                 prior_mean=prior_mean)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.to(device)

    # Train Preparation
    best_val_loss = [np.inf, np.inf, np.inf]

    Q_loss_list = []
    Mse_loss_list = []
    val_loss_list = []
    training_loss_list = []

    epochs = config["epochs"]
    patience_count = 0
    patience = config["patience"]
    anneal_count = 0
    use_anneal = config["anneal"]
    anneal_max = config["beta_max"]
    total_batches = int(epochs * train_data.shape[0] / config["batch_size"])
    anneal_max_count = int(0.2 * total_batches / anneal_max)

    for epoch in range(epochs):
        model.train()
        total_len = 0
        total_loss = 0
        total_Q_loss = 0
        total_Mse_loss = 0
        for y, o, x in train_dataloader:

            o_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, Q = model(o, x)

            if use_anneal:
                anneal = min(anneal_max, 1. * anneal_count / anneal_max_count)
            else:
                anneal = anneal_max
            l2_reg = torch.tensor([0]).to(device).float()
            for p in model.parameters():
                l2_reg = l2_reg + torch.norm(p)
            Vae_loss, Q_loss, Y_z_loss = iVae_loss_function(o, o_hat, mean, log_var, prior_mean, prior_log_var, anneal)\
                                        , Q_Test_loss_function(Q, o.bool()), torch.mean(torch.sum(nn.functional.binary_cross_entropy(Y_hat, y, reduction="none"), dim=1))

            loss = Vae_loss + Q_loss + Y_z_loss + l2_reg * config["l2_penalty"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            anneal_count += 1
            total_len += len(x)
            total_loss += loss.item() * len(x)
            total_Q_loss += Q_loss.item() * len(x)
            total_Mse_loss += Y_z_loss.item() * len(x)

        model.eval()
        test_o_hat, mean_val, log_var_val, prior_mean_val, prior_log_var_val, Y_hat, Q = model(val_data,user_feat[test_user_index])
        val_Q_test_loss = Q_Test_loss_function(Q, val_data.bool()).detach().item()
        val_Y_loss = torch.mean(torch.sum(nn.functional.binary_cross_entropy(Y_hat, val_Y_data, reduction="none"), dim=1)).detach().item()
        val_ivae_loss = iVae_loss_function(val_data, test_o_hat, mean_val, log_var_val, prior_mean_val, prior_log_var_val, anneal=anneal_max).detach().item()
        val_loss = val_Y_loss

        patience_count += 1

        if val_Y_loss < best_val_loss[1]:
            patience_count = 0
            best_val_loss = [val_Q_test_loss, val_Y_loss, val_ivae_loss]
            # print(best_val_loss)
            torch.save(model.state_dict(), data_params["Shadow_path"] + "{}_{}_Best_U_Shadow_Model.pt".format(data_params["name"], "Val"))

        if patience_count > patience:
            print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

        if config["show_log"]:
            training_loss = total_loss / total_len
            val_loss_list.append(val_loss)
            training_loss_list.append(training_loss)
            Q_loss_list.append(total_Q_loss / total_len)
            Mse_loss_list.append(total_Mse_loss / total_len)
            print("Epoch {}, Training Loss = {}, Val Loss = {} ".format(epoch, training_loss, val_loss))

    if config["show_log"]:
        plt.plot(val_loss_list, label="Val Loss")
        plt.plot(training_loss_list, label="Training Loss")
        plt.title("iVAE")
        plt.legend()
        plt.show()
        plt.plot(Mse_loss_list, label="MSE loss")
        plt.title("MSE loss")
        plt.legend()
        plt.show()
        plt.plot(Q_loss_list, label="Q-Tester Loss")
        plt.title("Q-Tester")
        plt.legend()
        plt.show()

    print("Best Val loss = {}".format(best_val_loss))
    if config['save_mean']:
        # save to local
        model.load_state_dict(torch.load(data_params["Shadow_path"] + "{}_{}_Best_U_Shadow_Model.pt".format(data_params["name"], "Val")))
        fullmatrix = torch.tensor(fullmatrix > 0, dtype=torch.float).to(device)
        O_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, Q = model(fullmatrix, user_feat)
        torch.save(mean.detach(), data_params["Shadow_path"] + "user_mean.pt")
        torch.save(log_var.detach().exp().sqrt(), data_params["Shadow_path"] + "user_std.pt")
        print('Already Save Prama')
    return best_val_loss

if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")
        u_layer_dim_search = []
        u_shadow_dim_search = []
        if args.data_params["name"] == 'coat':
            u_layer_dim_search = [512, 1024]
            u_shadow_dim_search = [8, 16]
        elif args.data_params["name"] == 'yahoo':
            u_layer_dim_search = [512, 1024]
            u_shadow_dim_search = [8, 16]
        elif args.data_params["name"] == 'kuai_rand':
            u_layer_dim_search = [512, 1024]
            u_shadow_dim_search = [8, 16]
        elif args.data_params["name"] == 'sim_data':
            u_layer_dim_search = [16]
            u_shadow_dim_search = [2]
        search_grid = {'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6], 'l2_penalty': [1e-3, 1e-4, 1e-5],
                'u_layer_dim': u_layer_dim_search, 'u_shadow_dim': u_shadow_dim_search}

        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [np.inf, np.inf, np.inf]
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            grid_config = {
                "seed": 1208,
                "save_mean": False,
                "show_log": False,
                "anneal": True,
                "beta_max": 1.0,
                "patience": 10,
                "epochs": 1000,
                "lr": supparams['lr'],
                "l2_penalty": supparams['l2_penalty'],
                "n_layers": 3,
                "u_layer_dim": supparams['u_layer_dim'],
                "u_shadow_dim": supparams['u_shadow_dim'],
                "batch_size": 512,
                "weight_decay": supparams['weight_decay'],
                "data_params": args.data_params,
            }
            result = user_shadow_train_eval(grid_config)
            print("index:", index, "params:", supparams)
            if result[1] < best_val[1]:
                best_val = result
                best_param = supparams
        print("best val is:", best_val, "params is:", best_param)
    else:
        Train_eval_config = {
            "seed": 1208,
            "save_mean": True,
            "show_log": False,
            "anneal": True,
            "beta_max": 1.0,
            "patience": 10,
            "epochs": 1000,
            "lr": 0.001,
            "l2_penalty": 1e-03,
            "n_layers": 3,
            "u_layer_dim": 64,
            "u_shadow_dim": 2,
            "batch_size": 512,
            "weight_decay": 1e-03,
            "data_params": args.data_params,
        }
        user_shadow_train_eval(Train_eval_config)
