import argparse
import os

File = os.getcwd()

coat_params = {
    "name": "coat",
    "threshold": 4.0,
    "init": 0.01,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "train_path": File + "/DataSet/coat/train.csv",
    "random_path": File + "/DataSet/coat/random.csv",
    "u_shadow_dim": 16,
    "i_shadow_dim": 32,
    'dense_layer_dim': 512,
    'dense_layer_num': 3,
    "user_feature_label": File + "/DataSet/coat/user_feat_onehot.csv",
    "user_feature_label_no": File + "/DataSet/coat/user_feat_label.csv",
    "user_feature_dim": [2, 6, 3, 3,],
    "ivae_dim": 32,
    "ivae_path": File + "/Weight/ivae/",
    "vae_dim": 16,
    "vae_path": File + "/Weight/vae/",
    "Shadow_path": File + "/Weight/coat_shadow/",
}

yahoo_params = {
    "name": "yahoo",
    "threshold": 4.0,
    "init": 0.01,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "train_path": File + "/DataSet/yahoo_R3/train.csv",
    "random_path": File + "/DataSet/yahoo_R3/random.csv",
    "user_feature_label": File + "/DataSet/yahoo_R3/user_feat_onehot.csv",
    "user_feature_label_no": File + "/DataSet/yahoo_R3/user_feat_label.csv",
    "user_feature_dim": [5, 5, 5, 5, 5, 5, 5],
    "u_shadow_dim": 16,
    'dense_layer_dim': 1024,
    'dense_layer_num': 3,
    "ivae_dim": 16,
    "ivae_path": File + "/Weight/ivae/",
    "vae_dim": 16,
    "vae_path": File + "/Weight/vae/",
    "Shadow_path": File + "/Weight/yahoo_shadow/",
}

kuai_rand_params = {
    "name": "kuai_rand",
    "threshold": 0.9,
    "init": 0.001,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "train_path": File + "/DataSet/kuai_rand/train.csv",
    "random_path": File + "/DataSet/kuai_rand/random.csv",
    "u_shadow_dim": 16,
    "i_shadow_dim": 16,
    'dense_layer_dim': 512,
    'dense_layer_num': 3,
    "user_feature_label": File + "/DataSet/kuai_rand/user_feat_onehot.csv",
    "user_feature_label_no": File + "/DataSet/kuai_rand/user_feat_label.csv",
    "user_feature_dim": [9, 2, 2, 8, 9, 7, 8, 2, 7, 50, 3, 7, 5, 4],
    "ivae_dim": 32,
    "ivae_path": File + "/Weight/ivae/",
    "vae_dim": 32,
    "vae_path": File + "/Weight/vae/",
    "Shadow_path": File + "/Weight/kuai_rand_shadow/",
}

sim_data = {
    "name": "sim_data",
    "threshold": 4,
    "init": 0.01,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "train_path": File + "/DataSet/sim_data/train_sp_0.1_cl_0.8_ns_10.csv",
    "random_path": File + "/DataSet/sim_data/random_sp_0.1_cl_0.8_ns_10.csv",
    "u_shadow_dim": 2,
    "i_shadow_dim": 16,
    'dense_layer_dim': 512,
    'dense_layer_num': 3,
    "user_feature_label": File + "/DataSet/sim_data/user_feat_onehot.csv",
    "user_feature_label_no": File + "/DataSet/sim_data/user_feat_label.csv",
    "item_feature_label": File + "/DataSet/sim_data/item_feat_onehot_base_stic.csv",
    "user_feature_dim": [5],
    "ivae_dim": 2,
    "ivae_path": File + "/Weight/ivae/",
    "vae_dim": 2,
    "vae_path": File + "/Weight/vae/",
    "Shadow_path": File + "/Weight/sim_shadow/",
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", type=str, default="coat")

    args = parser.parse_args()
    data_params = None
    if args.dataset == "yahoo":
        data_params = yahoo_params
    elif args.dataset == "coat":
        data_params = coat_params
    elif args.dataset == "kuai_rand":
        data_params = kuai_rand_params
    elif args.dataset == "sim_data":
        data_params = sim_data
    else:
        raise Exception("invalid dataset")

    setattr(args, "data_params", data_params)
    return args
