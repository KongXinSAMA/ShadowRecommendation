import numpy as np
import pandas as pd
import torch.random
import random
from matplotlib import pyplot as plt
import sys
import torch.nn as nn
from Utils.Utils import seed_everything

def gen_gaussian_embedding(size, embedding_dim, add_bias=False):
    emb = torch.randn((size, embedding_dim)) * 5
    if add_bias:
        bias = torch.randint(4, (size, 1))
        emb += bias
    return emb


def gen_uniform_embedding(size, embedding_dim):
    emb = torch.rand((size, embedding_dim))
    return emb


def drop_probability(rating, a):
    return 0.5 * (1 - a) + a * (5 - rating) / 4.0

def save_csv(uids, iids, ratings, name):
    df = pd.DataFrame(
        data={"user_id": uids, "item_id": iids, "rating": ratings}
    )
    df.to_csv(name, sep=",", index=None)
    return df

if __name__ == '__main__':
    user_num = 1200
    item_num = 400
    sparse_ratio = 0.1
    embedding_dim = 32
    shadow_dim = 2
    a = 0.8
    print("a:",a)
    rating_noise_ratio = 0
    name = "_sp_{}_cl_{}_ns_{}".format(sparse_ratio, a, rating_noise_ratio)
    # X
    x_size = 5
    x_true = np.array(np.tile(np.arange(x_size), int(user_num / x_size)), dtype="int")
    x_mu = np.random.uniform(-10, 10, [5])
    x_var = np.random.uniform(1, 8, [5])
    x_effect_true = np.random.uniform(x_mu, np.sqrt(x_var), (5))
    x_effect = np.array(np.tile(x_effect_true, int(user_num / x_size)),
                        dtype="float").reshape(-1, 1)
    user_feat_onehot = pd.get_dummies(x_true)
    pd.Series(x_true).to_csv("user_feat_label.csv", index=None)
    user_feat_onehot.to_csv("user_feat_onehot.csv", index=None)
    # 生成阴影变量Z
    mu_true = np.random.uniform(-16, 16, [shadow_dim, x_size])
    var_true = np.random.uniform(1, 6, [shadow_dim, x_size])
    z_true = np.vstack([
        np.random.normal(mu_true[i][x_true], np.sqrt(var_true[i][x_true])) for i in range(shadow_dim)]).T
    # print(x_true.shape, z_true.shape)

    # S
    s_mu = np.random.uniform(-5, 5)
    s_var = np.random.uniform(1, 3)
    s_effect = np.array(np.random.uniform(s_mu, np.sqrt(s_var), (1200)), dtype="float").reshape(-1, 1) + 0.2 * x_effect
    s_rate = torch.sigmoid(torch.tensor(s_effect))

    # Treament
    treatment_noise_ratio = rating_noise_ratio
    z_true = torch.tensor(z_true, dtype=torch.float)
    emb_z = gen_uniform_embedding(item_num, shadow_dim)
    x_effect = torch.tensor(x_effect, dtype=torch.float)
    emb_x = 0.2 * gen_uniform_embedding(item_num, 1)

    M2 = 0.2 * torch.rand((z_true.shape[1], z_true.shape[1]))
    M1 = 0.4 * torch.rand((x_effect.shape[1], x_effect.shape[1]))

    expousure_prob = (z_true @ M2 @ emb_z.T + x_effect @ M1 @ emb_x.T)
    noise = (torch.randn_like(expousure_prob)) * treatment_noise_ratio
    expousure_prob += noise
    # print(expousure_prob.abs().mean(), noise.abs().mean())
    expousure_prob = torch.sigmoid(expousure_prob) * sparse_ratio
    expousure = torch.bernoulli(expousure_prob)
    # print(torch.sum(expousure), expousure.sum()/600/1200)
    # print(expousure.sum(0).min(), expousure.sum(0).max())

    # Rating
    emb_u = gen_uniform_embedding(user_num, embedding_dim)
    emb_i = gen_gaussian_embedding(item_num, embedding_dim, add_bias=True)
    exp_effect = emb_u @ emb_i.T  # 1200 * 400
    x_z_effect = z_true @ emb_z.T + x_effect @ emb_x.T
    noise = torch.randn((user_num, item_num)) * rating_noise_ratio

    mf_res = exp_effect + x_z_effect + noise

    soft_mf_res = torch.pow(
        (mf_res - torch.quantile(mf_res, 0.05)) / (torch.quantile(mf_res, 0.95) - torch.quantile(mf_res, 0.05)), 1)

    rating_matrix = torch.ceil(soft_mf_res * 5)
    rating_matrix[rating_matrix > 5] = 5
    rating_matrix[rating_matrix < 1] = 1
    # plt.hist(expousure_prob.mean(1))
    # plt.hist(expousure.sum(1))
    # plt.show()

    # OS
    uids, iids = expousure.nonzero(as_tuple=True)
    ratings = rating_matrix[uids, iids]
    drop_rate = 0.8
    keep_rate = torch.ones_like(ratings) * (1 - drop_rate)
    s_keep_rate = torch.tensor(s_rate[uids]).view(-1)
    keep = (torch.bernoulli(keep_rate).bool() | torch.bernoulli(s_keep_rate).bool()).bool()
    ratings_keep = ratings[keep]
    uids_keep = uids[keep]
    iids_keep = iids[keep]
    # print(ratings.shape, ratings_keep)

    keep_counts = torch.bincount(ratings_keep.int())[1:]

    counts = torch.bincount(ratings.int())[1:]
    print("raw:",counts)
    print("train", keep_counts)

    # x = torch.arange(1, 6)
    # plt.bar(x.numpy(), counts.numpy())
    # plt.xlabel('Rating')
    # plt.ylabel('Count')
    # plt.title('Rating Distribution')
    # plt.show()
    # plt.bar(x.numpy(), keep_counts.numpy())
    # plt.xlabel('Rating')
    # plt.ylabel('Count')
    # plt.title('Rating Distribution')
    # plt.show()

    # Random test
    random_iids_list = list()
    random_item_size = 20
    for i in range(user_num):
        random_iids_list.append(torch.randperm(item_num)[:random_item_size])
    random_iids = torch.cat(random_iids_list)
    random_uids = torch.arange(0, user_num).view(-1, 1).repeat(1, random_item_size).view(-1)
    random_ratings = rating_matrix[random_uids, random_iids]

    random_prob = drop_probability(random_ratings, a)
    random_keep = torch.bernoulli(random_prob).bool()
    random_ratings_keep = random_ratings[random_keep]
    random_uids_keep = random_uids[random_keep]
    random_iids_keep = random_iids[random_keep]

    random_keep_counts = torch.bincount(random_ratings_keep.int())[1:]
    t_counts = torch.bincount(random_ratings.int())[1:]
    print("raw test",t_counts)
    print("test", random_keep_counts)

    print("Train Size:", ratings_keep.shape,"Dense:", ratings_keep.shape[0]/400/1200)
    print("Test Size:", random_ratings.shape)
    print("raw Nav:Pos:", counts[:3].sum(), counts[3:].sum())
    print("train Nav:Pos", keep_counts[:3].sum(), keep_counts[3:].sum())
    print("random Nav:Pos", random_keep_counts[:3].sum(), random_keep_counts[3:].sum())


    # x = torch.arange(1, 6)
    # plt.bar(x.numpy(), counts.numpy())
    # plt.xlabel('Rating')
    # plt.ylabel('Count')
    # plt.title('Rating Distribution')
    # plt.show()

    df_train = save_csv(uids_keep, iids_keep, ratings_keep, "train{}.csv".format(name))
    df_random = save_csv(random_uids_keep, random_iids_keep, random_ratings_keep, "random{}.csv".format(name))

    cmap = plt.get_cmap('brg', 6)
    c = cmap(x_true)

    shadow_z_mean = torch.load("../../Weight/sim_shadow/user_mean.pt", map_location='cpu')
    vae_z_mean = torch.load("../../Weight/vae/sim_data_user_mean.pt", map_location='cpu')
    ivae_z_mean = torch.load("../../Weight/ivae/sim_data_user_mean.pt", map_location='cpu')
    plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("True 2-dim latent")
    plt.scatter(z_true.T[0], z_true.T[1], c=c, s=1)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Shadow Construct")
    plt.scatter(shadow_z_mean.T[0].detach().numpy(), shadow_z_mean.T[1].detach().numpy(), c=c, s=1)
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("VAE")
    plt.scatter(vae_z_mean.T[0].detach().numpy(), vae_z_mean.T[1].detach().numpy(), c=c, s=1)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("iVAE")
    plt.scatter(ivae_z_mean.T[0].detach().numpy(), ivae_z_mean.T[1].detach().numpy(), c=c, s=1)
    plt.show()