import Argparser
import itertools
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from Utils.Utils import *
from Utils.BaseModel import AffineTransform, RealNVP
from Utils.Dataloader import Base_TrainingDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def s0_density_train_eval(config):
    seed_everything(config["seed"])
    data_params = config["data_params"]

    # pca_module = PCA(n_components=32)
    ShadowData = torch.load(data_params["Shadow_path"] + "user_mean.pt", map_location='cpu')
    ShadowStd = torch.load(data_params["Shadow_path"] + "user_std.pt", map_location='cpu').to(device)
    user_svc_mean = ShadowData.to(device)
    if config['showfigandlog']:
        print("Size", user_svc_mean.shape)

    _, _, _, _, matrix, sample_num = construct_dense_data(config)

    random_user = np.array(random.choices(range(0, matrix.shape[0] - 1), k=sample_num))
    random_item = np.array(random.choices(range(0, matrix.shape[1] - 1), k=sample_num))
    ranset_rating = matrix[random_user, random_item]
    ranset_rating = ranset_rating.reshape(-1, 1)
    mask = (ranset_rating == 0)
    mask.resize(mask.shape[0])
    Shadow_s0 = user_svc_mean[torch.Tensor(random_user[mask]).type(torch.long)]
    prob = torch.rand_like(Shadow_s0)
    Shadow_s0 += prob * ShadowStd[torch.Tensor(random_user[mask]).type(torch.long)]
    split_num = int(len(Shadow_s0) * 0.8)
    Train_Shadow_s0, Val_Shadow_s0 = Shadow_s0[0:split_num, :], Shadow_s0[split_num:, :]
    S0_train_dataset = Base_TrainingDataset(data=Train_Shadow_s0)
    S0_train_dataloader = DataLoader(S0_train_dataset, batch_size=config["batch_size"], shuffle=True)
    S0_val_dataset = Base_TrainingDataset(data=Val_Shadow_s0)
    S0_val_dataloader = DataLoader(S0_val_dataset, batch_size=config["batch_size"], shuffle=True)
    if config['showfigandlog']:
        print("train size:", split_num, "val size:", Val_Shadow_s0.shape[0])

    inp_size = user_svc_mean.shape[1]
    layer_dim = config['layer_dim']
    layer_num = config['layer_num']
    real_nvp_s0 = RealNVP(
        [AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
         AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
         AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
         AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
         AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)])
    real_nvp_s0.to(device)
    optimizer = torch.optim.Adam(params=real_nvp_s0.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])

    best_val_loss = np.inf
    patience = config["patience"]
    patience_count = 0
    S0_Density_List = []

    for epoch in range(config["epochs"]):
        real_nvp_s0.train()
        total_loss, total_len = 0, 0
        for index, Shadow_s0 in enumerate(S0_train_dataloader):
            optimizer.zero_grad()
            l2_reg = torch.tensor([0]).to(device)
            for param in real_nvp_s0.parameters():
                l2_reg = l2_reg + torch.norm(param)
            loss = real_nvp_s0.nll(Shadow_s0) + l2_reg * config["l2_penalty"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(Shadow_s0)
            total_len += len(Shadow_s0)
        epoch_loss = total_loss / total_len
        S0_Density_List.append(epoch_loss)
        # val
        real_nvp_s0.eval()
        val_total_loss, val_total_len = 0, 0
        for index, Shadow_s0 in enumerate(S0_val_dataloader):
            val_loss = real_nvp_s0.nll(Shadow_s0)
            val_total_loss += val_loss.item() * len(Shadow_s0)
            val_total_len += len(Shadow_s0)
        epoch_val_loss = val_total_loss / val_total_len

        if config['showfigandlog']:
            print("Train loss:", epoch_loss, ", Val loss:", epoch_val_loss)

        patience_count += 1
        if epoch_val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = epoch_val_loss
            torch.save(real_nvp_s0.state_dict(), data_params["Shadow_path"] + "Best_S0_UDensety_Estimator.pt")

        if patience_count > patience:
            print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

    if config['showfigandlog']:
        plt.plot(S0_Density_List, label="S0_Density_Loss")
        plt.title("S0_Density_Loss")
        plt.legend()
        plt.show()

    print("S0_best_val_loss:", best_val_loss)
    return best_val_loss


def s1_density_train_eval(config):

    seed_everything(config["seed"])
    data_params = config["data_params"]
    ShadowData = torch.load(data_params["Shadow_path"] + "user_mean.pt", map_location='cpu').to(device)
    ShadowStd = torch.load(data_params["Shadow_path"] + "user_std.pt", map_location='cpu').to(device)
    user_svc_mean = ShadowData.to(device)
    if config['showfigandlog']:
        print("Size", user_svc_mean.shape)

    train_users_index, train_items_index, val_users_index, val_items_index, matrix, sample_num = construct_dense_data(
        config)
    print("train size:", train_users_index.shape[0], "val size:", val_users_index.shape[0])
    Train_Shadow_s1 = user_svc_mean[torch.Tensor(train_users_index).type(torch.long)]
    prob = torch.rand_like(Train_Shadow_s1)
    Train_Shadow_s1 += prob * ShadowStd[torch.Tensor(train_users_index).type(torch.long)]
    Val_Shadow_s1 = user_svc_mean[torch.Tensor(val_users_index).type(torch.long)]
    prob = torch.rand_like(Val_Shadow_s1)
    Val_Shadow_s1 += prob * ShadowStd[torch.Tensor(val_users_index).type(torch.long)]

    S1_train_dataset = Base_TrainingDataset(data=Train_Shadow_s1)
    S1_train_dataloader = DataLoader(S1_train_dataset, batch_size=config["batch_size"], shuffle=True)
    S1_val_dataset = Base_TrainingDataset(data=Val_Shadow_s1)
    S1_val_dataloader = DataLoader(S1_val_dataset, batch_size=config["batch_size"], shuffle=True)

    inp_size = user_svc_mean.shape[1]
    layer_dim = config['layer_dim']
    layer_num = config['layer_num']
    real_nvp_s1 = RealNVP(
        [AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
         AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
         AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
         AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
         AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)])
    real_nvp_s1.to(device)
    optimizer = torch.optim.Adam(params=real_nvp_s1.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])

    best_val_loss = np.inf
    patience_count = 0
    patience = config['patience']
    S1_Density_List = []
    epochs = config["epochs"]

    for epoch in range(epochs):
        real_nvp_s1.train()
        total_loss, total_len = 0, 0
        # Test
        for index, Shadow_s1 in enumerate(S1_train_dataloader):
            optimizer.zero_grad()
            l2_reg = torch.tensor([0]).to(device)
            for param in real_nvp_s1.parameters():
                l2_reg = l2_reg + torch.norm(param)
            loss = real_nvp_s1.nll(Shadow_s1) + l2_reg * config["l2_penalty"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(Shadow_s1)
            total_len += len(Shadow_s1)
        epoch_loss = total_loss / total_len
        S1_Density_List.append(epoch_loss)
        # Val
        real_nvp_s1.eval()
        val_total_loss, val_total_len = 0, 0
        for index, Shadow_s1 in enumerate(S1_val_dataloader):
            val_loss = real_nvp_s1.nll(Shadow_s1)
            val_total_loss += val_loss.item() * len(Shadow_s1)
            val_total_len += len(Shadow_s1)
        epoch_val_loss = val_total_loss / val_total_len
        if config['showfigandlog']:
            print("Train loss:", epoch_loss, ", Val loss:", epoch_val_loss)
        # Other
        patience_count += 1
        if epoch_val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = epoch_val_loss
            torch.save(real_nvp_s1.state_dict(), data_params["Shadow_path"] + "Best_S1_UDensety_Estimator.pt")

        if patience_count > patience:
            if config['showfigandlog']:
                print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

    if config['showfigandlog']:
        plt.plot(S1_Density_List, label="S1_Density_Loss")
        plt.title("S1_Density_Loss")
        plt.legend()
        plt.show()

        real_nvp_s0 = RealNVP(
        [AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
         AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
         AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
         AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
         AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)])
        real_nvp_s1.load_state_dict(torch.load(data_params["Shadow_path"] + "Best_S1_UDensety_Estimator.pt"))
        real_nvp_s0.load_state_dict(torch.load(data_params["Shadow_path"] + "Best_S0_UDensety_Estimator.pt"))

        Ratio = torch.exp(real_nvp_s0.log_prob(Shadow_s1)) / torch.exp(real_nvp_s1.log_prob(Shadow_s1))
        plt.plot(Ratio.detach().cpu())
        plt.title("Ratio")
        plt.legend()
        plt.show()

    print("S1_best_val_loss:", best_val_loss)
    return best_val_loss


if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")
        search_grid = {'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6],
                       'l2_penalty': [1e-4, 1e-5], 'layer_dim': [512, 1024]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, np.inf
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            grid_config = {
                "seed": 1208,
                "epochs": 1000,
                "patience": 10,
                "lr": supparams['lr'],
                "batch_size": 512,
                "layer_dim": supparams['layer_dim'],
                "layer_num": 3,
                "l2_penalty": supparams['l2_penalty'],
                "weight_decay": supparams['weight_decay'],
                "showfigandlog": False,
                "data_params": args.data_params,
            }
            result = s1_density_train_eval(grid_config)
            print("index:", index, "index:", supparams)
            if result < best_val:
                best_val = result
                best_param = supparams
        print(best_val, best_param)
    else:
        Train_eval_config = {
            "seed": 1208,
            "epochs": 1000,
            "patience": 5,
            "lr": 5e-04,
            "batch_size": 1024,
            "layer_dim": 512,
            "layer_num": 3,
            "l2_penalty": 1e-04,
            "weight_decay": 1e-06,
            "showfigandlog": True,
            "data_params": args.data_params,
        }
        s0_density_train_eval(Train_eval_config)
        s1_density_train_eval(Train_eval_config)

