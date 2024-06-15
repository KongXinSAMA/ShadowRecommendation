import Argparser
import itertools
from torch.optim import Adam
from torch import nn
from Utils.Utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from Utils.BaseModel import Ivae
from Utils.Dataloader import iVae_TrainingDataset
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

def user_shadow_train_eval(config):
    seed_everything(config["seed"])
    # Read Data
    data_params = config["data_params"]
    fullmatrix, train_matrix, val_matrix, train_user_index, test_user_index = construct_user_dataset(
                                                                                data_params["train_path"],
                                                                                train_ratio=data_params["train_ratio"])

    user_feat = pd.read_csv(data_params["user_feature_label"]).to_numpy()
    user_feat = torch.tensor(user_feat, dtype=torch.float).to(device)
    train_data = torch.tensor(train_matrix > 0, dtype=torch.float).to(device)
    val_data = torch.tensor(val_matrix > 0, dtype=torch.float).to(device)


    train_dataset = iVae_TrainingDataset(train_data, user_feat[train_user_index])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Creat model
    layer_dim = config["u_layer_dim"]
    shadow_dim = config["u_shadow_dim"]
    user_feat_dim = user_feat.shape[1]
    input_size = train_matrix.shape[1]
    prior_mean = True if data_params['name'] == 'sim_data' else False
    model = Ivae(input_dim=input_size,
                 feature_dim=user_feat_dim,
                 shadow_dim=shadow_dim,
                 layer_dim=layer_dim,
                 n_layers=config["n_layers"],
                 device = device,
                 prior_mean=prior_mean)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.to(device)

    # Train Preparation
    best_val_loss = np.inf

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
        for o, x in train_dataloader:

            o_hat, mean, log_var, prior_mean, prior_log_var = model(o, x)

            if use_anneal:
                anneal = min(anneal_max, 1. * anneal_count / anneal_max_count)
            else:
                anneal = anneal_max
            l2_reg = torch.tensor([0]).to(device).float()
            for p in model.parameters():
                l2_reg = l2_reg + torch.norm(p)
            Vae_loss = iVae_loss_function(o, o_hat, mean, log_var, prior_mean, prior_log_var, anneal)

            loss = Vae_loss + l2_reg * config["l2_penalty"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            anneal_count += 1
            total_len += len(x)
            total_loss += loss.item() * len(x)

        model.eval()
        test_o_hat, mean_val, log_var_val, prior_mean_val, prior_log_var_val = model(val_data,user_feat[test_user_index])
        val_loss = iVae_loss_function(val_data, test_o_hat, mean_val, log_var_val, prior_mean_val, prior_log_var_val, anneal=anneal_max)

        patience_count += 1
        if val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), data_params["ivae_path"] + "{}_{}_Best_ivae.pt".format(data_params["name"], "Val"))

        if patience_count > patience:
            print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

        if config["show_log"]:
            training_loss = total_loss / total_len
            val_loss_list.append(val_loss.item())
            training_loss_list.append(training_loss)
            print("Epoch {}, Training Loss = {}, Val Loss = {} ".format(epoch, training_loss, val_loss))

    if config["show_log"]:
        plt.plot(val_loss_list, label="Val Loss")
        plt.plot(training_loss_list, label="Training Loss")
        plt.title("iVAE")
        plt.legend()
        plt.show()


    print("Best Val loss = {}".format(best_val_loss))
    # save to local
    model.load_state_dict(torch.load(data_params["ivae_path"] + "{}_{}_Best_ivae.pt".format(data_params["name"], "Val")))
    fullmatrix = torch.tensor(fullmatrix > 0, dtype=torch.float).to(device)
    O_hat, mean, log_var, prior_mean, prior_log_var = model(fullmatrix, user_feat)
    torch.save(mean.detach(), data_params["ivae_path"] + "{}_user_mean.pt".format(data_params["name"]))
    torch.save(log_var.detach().exp().sqrt(), data_params["ivae_path"] + "{}_user_std.pt".format(data_params["name"]))
    return float(best_val_loss)

if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")
        u_layer_dim_search = []
        u_shadow_dim_search = []
        if args.data_params["name"] == 'coat':
            u_layer_dim_search = [512, 1024]
            u_shadow_dim_search = [16, 32]
        elif args.data_params["name"] == 'yahoo':
            u_layer_dim_search = [512, 1024]
            u_shadow_dim_search = [16, 32]
        elif args.data_params["name"] == 'kuai_rand':
            u_layer_dim_search = [512, 1024]
            u_shadow_dim_search = [16, 32]
        elif args.data_params["name"] == 'sim_data':
            u_layer_dim_search = [64, 128]
            u_shadow_dim_search = [2]
        search_grid = {'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6], 'l2_penalty': [1e-3, 1e-4, 1e-5],
                'u_layer_dim': u_layer_dim_search, 'u_shadow_dim': u_shadow_dim_search}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, np.inf
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
            if result < best_val:
                best_val = result
                best_param = supparams
        print("best val is:", best_val, "params is:", best_param)
    else:
        Train_eval_config = {
            "tune": False,
            "seed": 1208,
            "show_log": True,
            "anneal": True,
            "beta_max": 1.0,
            "patience": 10,
            "epochs": 1000,
            "lr": 0.0005,
            "l2_penalty": 0.001,
            "n_layers": 3,
            "u_layer_dim": 128,
            "u_shadow_dim": 2,
            "batch_size": 512,
            "weight_decay": 1e-06,
            "data_params": args.data_params,
        }
        user_shadow_train_eval(Train_eval_config)