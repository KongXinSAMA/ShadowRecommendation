import bottleneck as bn
import numpy as np
import random
import torch
import os
import pandas as pd
from scipy.sparse import csr_matrix

from torch.utils.data import Dataset, DataLoader
from Utils.Dataloader import MFRatingDataset

def seed_everything(seed=1208):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_by_user(df, ratio, seed=1208):
    np.random.seed(seed)
    unique_uids = df["user_id"].unique()
    test_users = np.random.choice(unique_uids, size=int(unique_uids.size * ratio), replace=False)
    val_users = np.setdiff1d(unique_uids, test_users)
    df_val = df.loc[df["user_id"].isin(val_users)]
    df_test = df.loc[df["user_id"].isin(test_users)]
    return df_val, df_test, val_users, test_users

def df_to_csr(df):
    rows = df["user_id"]
    cols = df["item_id"]
    values = df["rating"]
    mat = csr_matrix((values, (rows, cols)))
    return mat

def np_to_csr(array):
    rows = array[:, 0].astype(int)
    cols = array[:, 1].astype(int)
    values = array[:, 2]
    mat = csr_matrix((values, (rows, cols)))
    return mat

def construct_rating_dataset(train_df_path, random_df_path, test_ratio):
    train_df = pd.read_csv(train_df_path)
    random_df = pd.read_csv(random_df_path)

    val_df, test_df, val_users, test_users = split_by_user(random_df, test_ratio)

    return train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()

def construct_user_dataset(df_path, train_ratio, seed=1208):
    df = pd.read_csv(df_path)
    unique_users = df["user_id"].unique()

    n_users = unique_users.shape[0]
    n_items = df["item_id"].max() + 1
    n_train_users = int(train_ratio * n_users)
    df['rating'] += 1

    np.random.seed(seed)
    train_user_index = np.random.choice(unique_users, size=n_train_users, replace=False)
    train_user_index = np.sort(train_user_index)
    test_user_index = np.setdiff1d(unique_users, train_user_index)

    matrix = df_to_csr(df)
    train_matrix = matrix[train_user_index]
    test_matrix = matrix[test_user_index]
    return matrix.toarray(), train_matrix.toarray(), test_matrix.toarray(), train_user_index, test_user_index

def construct_item_dataset(df_path, train_ratio, seed=1208):
    df = pd.read_csv(df_path)
    unique_items = df["item_id"].unique()
    n_users = df["user_id"].max() + 1
    n_items = unique_items.shape[0]
    df['rating'] += 1

    n_train_items = int(train_ratio * n_items)
    np.random.seed(seed)
    train_item_index = np.random.choice(unique_items, size=n_train_items, replace=False)
    train_item_index = np.sort(train_item_index)
    test_item_index = np.setdiff1d(unique_items, train_item_index)

    matrix = df_to_csr(df).transpose()
    train_matrix = matrix[train_item_index]
    test_matrix = matrix[test_item_index]
    return matrix.toarray(), train_matrix.toarray(), test_matrix.toarray(), train_item_index, test_item_index

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=5):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)])
    valid_index = np.nonzero(IDCG)
    return DCG[valid_index] / IDCG[valid_index]

def Recall_at_k_batch(X_pred, heldout_batch, k=5):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = heldout_batch > 0
    hit = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    total_size = X_true_binary.sum(axis=1)
    valid_index = np.nonzero(total_size)
    recall = hit[valid_index] / total_size[valid_index]
    return recall

def get_dataloader(train_mat, train_ratings, val_mat, test_mat, batch_size, require_index=False, num_workers=4, pin_memory=True):

    train_dataset = MFRatingDataset(train_mat[:, 0].astype(int),
                                    train_mat[:, 1].astype(int),
                                    train_ratings,
                                    require_index=require_index)
    val_dataset = MFRatingDataset(val_mat[:, 0].astype(int),
                                  val_mat[:, 1].astype(int),
                                  val_mat[:, 2],
                                  require_index=require_index)
    test_dataset = MFRatingDataset(test_mat[:, 0].astype(int),
                                   test_mat[:, 1].astype(int),
                                   test_mat[:, 2],
                                   require_index=require_index)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

def construct_mf_dataloader(config, require_index=False):
    data_params = config["data_params"]

    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    threshold = data_params["threshold"]

    train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
    val_mat[:, 2] = val_mat[:, 2] >= threshold
    test_mat[:, 2] = test_mat[:, 2] >= threshold

    print("train size:", train_mat.shape[0], "val size:", val_mat.shape[0], "test size:", test_mat.shape[0])
    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"],
                                                           require_index=require_index)

    return train_loader, val_loader, test_loader, n_users, n_items

def load_uniform_data_from_np(ratio, array, shape):
    size = int(ratio * array.shape[0])
    index = np.random.permutation(np.arange(array.shape[0])[:size])
    rows, cols, rating = array[index, 0], array[index, 1], array[index, 2]
    return csr_matrix(
        (rating, (rows, cols)), shape=shape
    ), index

def construct_ips_dataloader(config, device):
    data_params = config["data_params"]
    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    threshold = data_params["threshold"]

    train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
    val_mat[:, 2] = val_mat[:, 2] >= threshold
    test_mat[:, 2] = test_mat[:, 2] >= threshold

    uniform_data, index = load_uniform_data_from_np(0.167, val_mat, shape=(n_users, n_items))
    val_mat = np.delete(val_mat, index, axis=0)

    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"])

    def Naive_Bayes_Propensity(train, unif):
        # follow [1] Jiawei Chen et, al, AutoDebias: Learning to Debias for Recommendation 2021SIGIR and
        # [2] Tobias Schnabel, et, al, Recommendations as Treatments: Debiasing Learning and Evaluation
        P_Oeq1 = train.getnnz() / (train.shape[0] * train.shape[1])
        train.data[train.data < threshold] = 0
        train.data[train.data >= threshold] = 1

        y_unique = np.unique(train.data)
        P_y_givenO = np.zeros(y_unique.shape)
        P_y = np.zeros(y_unique.shape)

        for i in range(len(y_unique)):
            P_y_givenO[i] = np.sum(train.data == y_unique[i]) / np.sum(
                np.ones(train.data.shape))
            P_y[i] = np.sum(unif.data == y_unique[i]) / np.sum(np.ones(unif.data.shape))
        Propensity = P_y_givenO * P_Oeq1 / P_y
        Propensity = Propensity * (np.ones((n_items, 2)))

        return y_unique, Propensity

    y_unique, Propensity = Naive_Bayes_Propensity(np_to_csr(train_mat), uniform_data)
    InvP = torch.reciprocal(torch.tensor(Propensity, dtype=torch.float)).to(device)
    return train_loader, val_loader, test_loader, n_users, n_items, y_unique, InvP

def construct_or_dataloader(config, device):
    data_params = config["data_params"]
    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    threshold = data_params["threshold"]

    train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
    val_mat[:, 2] = val_mat[:, 2] >= threshold
    test_mat[:, 2] = test_mat[:, 2] >= threshold

    uniform_data, index = load_uniform_data_from_np(0.166, val_mat, shape=(n_users, n_items))
    val_mat = np.delete(val_mat, index, axis=0)

    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"])

    def Naive_Bayes_Propensity(train, unif):
        # follow [1] Jiawei Chen et, al, AutoDebias: Learning to Debias for Recommendation 2021SIGIR and
        # [2] Tobias Schnabel, et, al, Recommendations as Treatments: Debiasing Learning and Evaluation
        P_Oeq1 = train.getnnz() / (train.shape[0] * train.shape[1])
        P_Oeq0 = 1 - P_Oeq1
        train.data[train.data < threshold] = 0
        train.data[train.data >= threshold] = 1

        y_unique = np.unique(train.data)
        P_y_givenO = np.zeros(y_unique.shape)
        P_y = np.zeros(y_unique.shape)

        for i in range(len(y_unique)):
            P_y_givenO[i] = np.sum(train.data == y_unique[i]) / np.sum(
                np.ones(train.data.shape))
            P_y[i] = np.sum(unif.data == y_unique[i]) / np.sum(np.ones(unif.data.shape))
        Propensity = P_y_givenO * P_Oeq1 / P_y
        OR = np.array([1, ((1 - Propensity[1]) * Propensity[0]) / ((1 - Propensity[0]) * Propensity[1])])
        OR_tilde = np.array([OR[0] / (OR[1] * P_y_givenO[0] + OR[0] * P_y_givenO[1]), OR[1] / (OR[1] * P_y_givenO[0] + OR[0] * P_y_givenO[1])])
        # OR_tilde = OR_tilde * (np.ones((n_items, 2)))


        return [P_Oeq1], [P_Oeq0], y_unique, OR_tilde

    P_Oeq1, P_Oeq0, y_unique, OR_tilde = Naive_Bayes_Propensity(np_to_csr(train_mat), uniform_data)

    return train_loader, val_loader, test_loader, n_users, n_items, P_Oeq1, P_Oeq0, y_unique, OR_tilde

def construct_dense_data(config):
    data_params = config["data_params"]
    train_df = pd.read_csv(data_params['train_path'])
    n_users = train_df["user_id"].max() + 1
    n_items = train_df["item_id"].max() + 1
    sample_num = int(len(train_df) * 1.2)
    # 要的是o的数据，kuairand里有0, 所以加1
    train_df["rating"] += 1
    matrix = df_to_csr(train_df)
    random_df = pd.read_csv(data_params['random_path'])
    val_df, _, _, _ = split_by_user(random_df, data_params["test_ratio"])

    return train_df['user_id'].to_numpy(), train_df['item_id'].to_numpy(), val_df['user_id'].to_numpy(), val_df['item_id'].to_numpy(), matrix, sample_num