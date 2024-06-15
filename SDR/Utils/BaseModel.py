import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activations = activations

        self.linear_nets = nn.Sequential()
        prev_dim = input_dim
        for i, (hidden_dim, activation) in enumerate(zip(hidden_dims, activations)):
            self.linear_nets.add_module("fc_{}".format(i), nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            if activation == "relu":
                self.linear_nets.add_module("act_{}".format(i), nn.ReLU())
            elif activation == "lrelu":
                self.linear_nets.add_module("act_{}".format(i), nn.LeakyReLU(0.2))
            elif activation == "sigmoid":
                self.linear_nets.add_module("act_{}".format(i), nn.Sigmoid())
            elif activation == "softmax":
                self.linear_nets.add_module("act_{}".format(i), nn.Softmax(dim=1))
            elif activation == "tanh":
                self.linear_nets.add_module("act_{}".format(i), nn.Tanh())

    def forward(self, x):
        return self.linear_nets(x)

class ShadowConstructor(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(ShadowConstructor, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

        # Q_Test
        self.Y_By_Z = MLP(input_dim=shadow_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                          activations=[activation] * (n_layers - 1) + ['sigmoid'])

        self.P_Y = MLP(input_dim=input_dim,
                       hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                       activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t, x):
        # tx = torch.cat((t, x), 1)
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        Y_hat = self.Y_By_Z(z)
        Q = self.P_Y(Y_hat)

        O_hat = self.decode(z, x)
        return O_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, Q

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z, x)
        return x_hat

class Shadow_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, I_Shadow_size, U_Shadow_size, device="cpu"):
        super(Shadow_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 给对撞建模
        self.shadow_u_emb = nn.Embedding(num_users, I_Shadow_size)
        self.shadow_u_emb.weight.data.uniform_(-0.01, 0.01)
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow, ItemShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        UShadows = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        IShadows = (ItemShadow * self.shadow_u_emb(u_id)).sum(1)
        Shadows_Feature = (UShadows + IShadows)
        # Shadows_Feature = UShadows
        # Shadows_Feature = IShadows
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

    def model_debias_loss(self, u_id, i_id, UserShadow, ItemShadow, PCAShadow, real_nvp_s0, real_nvp_s1, y_train, loss_f):
        preds = self.forward(u_id, i_id, UserShadow, ItemShadow).view(-1)

        Ratio = torch.exp(real_nvp_s1.log_prob(PCAShadow)) / torch.exp(real_nvp_s0.log_prob(PCAShadow))
        preds = preds * Ratio
        loss = loss_f(preds, y_train)
        return loss

class U_Shadow_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, U_Shadow_size, device="cpu"):
        super(U_Shadow_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 给对撞建模
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        # self.shadow_i_emb = nn.Embedding(num_users, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        Shadows_Feature = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

class SBR(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, U_Shadow_size, init, device="cpu"):
        super(SBR, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-init, init)
        self.user_bias.weight.data.uniform_(-init, init)
        self.item_emb.weight.data.uniform_(-init, init)
        self.item_bias.weight.data.uniform_(-init, init)
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-init*0.1, init*0.1)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        Shadows_Feature = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return (U * I).sum(1) + b_u + b_i + self.mean

class IPS_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, corY, InverP, device="cpu"):
        super(IPS_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.corY = corY
        self.invP = nn.Embedding(num_items, 2)
        self.invP.weight = torch.nn.Parameter(InverP)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return (U * I).sum(1) + b_u + b_i + self.mean

    def model_ips_loss(self, u_id, i_id, y_train, loss_f):
        preds = self.forward(u_id, i_id).view(-1)

        weight = torch.ones(y_train.shape).to(self.device)
        weight[y_train == self.corY[0]] = self.invP(i_id)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(i_id)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = torch.mean(weight * cost)
        return loss

class RD_IPS_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, corY, upBound, lowBound, InverP, device="cpu"):
        super(RD_IPS_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.corY = corY
        self.upBound = upBound
        self.lowBound = lowBound
        self.invP = nn.Embedding(num_items, 2)
        self.invP.weight = torch.nn.Parameter(InverP)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return (U * I).sum(1) + b_u + b_i + self.mean

    def model_ips_loss(self, u_id, i_id, y_train, loss_f):
        preds = self.forward(u_id, i_id).view(-1)

        weight = torch.ones(y_train.shape).to(self.device)
        weight[y_train == self.corY[0]] = self.invP(i_id)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(i_id)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = torch.mean(weight * cost)
        return loss

    def ips_loss(self, u_id, i_id, y_train, loss_f):
        preds = self.forward(u_id, i_id).view(-1)

        weight = torch.ones(y_train.shape).to(self.device)
        weight[y_train == self.corY[0]] = self.invP(i_id)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(i_id)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = - torch.mean(weight * cost)
        return loss

    def update_ips(self):
        with torch.no_grad():
            self.invP.weight.data[self.invP.weight.data > self.upBound] = self.upBound[
                self.invP.weight.data > self.upBound]
            self.invP.weight.data[self.invP.weight.data < self.lowBound] = self.lowBound[
                self.invP.weight.data < self.lowBound]

class MFwithFeature(nn.Module):
    def __init__(self, num_users, num_items, feature_size, embedding_size, device="cpu"):
        super(MFwithFeature, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.feature_embs_u = []
        self.feature_embs_i = []
        for feature_dim in feature_size:
            emb_u = nn.Embedding(feature_dim, 32)
            emb_u.weight.data.uniform_(-0.01, 0.01)
            emb_i = nn.Embedding(num_items, 32)
            emb_i.weight.data.uniform_(-0.01, 0.01)
            self.feature_embs_i.append(emb_i.to(device))
            self.feature_embs_u.append(emb_u.to(device))

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, features):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        Y = torch.zeros_like(u_id).to(self.device)
        for i in range(features.shape[1]):
            Y = Y + (self.feature_embs_u[i](features[:, i]) * self.feature_embs_i[i](i_id)).sum(1)
        return (U * I).sum(1) + b_u + b_i + self.mean + Y

class RealNVP_MLP(nn.Module):
    def __init__(self, input_size, layer_num, layer_dim, output_size):
        super().__init__()
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(input_size, layer_dim))
            layers.append(nn.GELU())
            input_size = layer_dim
        layers.append(nn.Linear(layer_dim, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AffineTransform(nn.Module):
    def __init__(self, type, input_size=2, layer_num=2, layer_dim=64, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.mask = self.build_mask(type=type, input_size=input_size).to(device)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)

        self.mlp = RealNVP_MLP(input_size=input_size, layer_num=layer_num, layer_dim=layer_dim, output_size=2).to(device)

    def build_mask(self, type, input_size):
        if input_size // 2 == 0:
            size = input_size // 2
        else :
            size = input_size // 2 + 1
        assert type in {"left", "right"}
        if type == "left":
            mask = torch.cat(
                (torch.FloatTensor([1]), torch.FloatTensor([0]))).repeat(size)
        elif type == "right":
            mask = torch.cat(
                (torch.FloatTensor([0]), torch.FloatTensor([1]))).repeat(size)
        else:
            raise NotImplementedError
        return mask[:input_size]

    def forward(self, x):

        batch_size = x.shape[0]
        mask = self.mask.repeat(batch_size, 1)

        x_ = x * mask
        log_s, t = self.mlp(x_).split(1, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)
        x = x * torch.exp(log_s) + t
        return x, log_s

class RealNVP(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))

    def flow(self, x):
        z, log_det = x, torch.zeros_like(x)
        for op in self.transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return torch.sum(log_det, dim=1) + torch.sum(self.prior.log_prob(z), dim=1)

    def nll(self, x):
        return - self.log_prob(x).mean()

class DeepCausal(nn.Module):
    def __init__(self,num_users,num_items,feature_dim,embedding_size,vae_mean, vae_std,device="cpu"):
        super(DeepCausal, self).__init__()
        self.mf_layer = MFwithFeature(num_users, num_items, feature_dim, embedding_size, device=device)
        self.vae_mean = vae_mean
        self.vae_std = vae_std

        self.item_emb = nn.Embedding(num_items, self.vae_mean.shape[1])
        self.item_emb.weight.data.uniform_(-0.05, 0.05)

        self.device = device
        self.to(device)

    def forward(self, uid, iid, u_feat, sample=False):
        mf_output = self.mf_layer(uid, iid, u_feat)
        mean = self.vae_mean[uid.type(torch.long)]

        if sample:
            std = self.vae_std[uid]
            eps = torch.randn_like(std).to(self.device)
            z = mean + std * eps
        else:
            z = mean

        i_emb = self.item_emb(iid)

        latent_regression = (i_emb * z).sum(1)
        return mf_output + latent_regression

    def predict(self, uid, iid, u_feat):
        return self.forward(uid, iid, u_feat, sample=False)

class IDCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, ivae_dim, device="cpu"):
        super(IDCF, self).__init__()

        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.shadow_i_emb = nn.Embedding(num_items, ivae_dim)
        self.shadow_i_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        UShadows = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        Shadows_Feature = UShadows
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

class Ivae(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(Ivae, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t):
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        O_hat = self.decode(z)
        return O_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat

class CIvae(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(CIvae, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim + feature_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t, x):
        tx = torch.cat((t, x), 1)
        return self.decoder(tx)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        O_hat = self.decode(z, x)
        return O_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z, x)
        return x_hat

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers=3, activation="lrelu", out_activation=None,
                 device="cpu"):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        # encoder
        self.mean_z = MLP(input_dim=input_dim,
                          hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation],)
        self.log_var_z = MLP(input_dim=input_dim,
                             hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation],)

        # decoder
        self.decoder = MLP(input_dim=latent_dim,
                           hidden_dims=[hidden_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'],)


    def encode(self, x):
        mean = self.mean_z(x)
        log_var = self.log_var_z(x)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def reconstruct(self, x, sample=False):
        mean, log_var = self.encode(x)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat

class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class InvPref(nn.Module):
    def __init__(self, num_users, num_items, num_envs, embedding_size):
        super(InvPref, self).__init__()
        self.user_emb_inv = nn.Embedding(num_users, embedding_size)
        self.user_bias_inv = nn.Embedding(num_users, 1)
        self.item_emb_inv = nn.Embedding(num_items, embedding_size)
        self.item_bias_inv = nn.Embedding(num_items, 1)

        self.user_emb_env = nn.Embedding(num_users, embedding_size)
        self.user_bias_env = nn.Embedding(num_users, 1)
        self.item_emb_env = nn.Embedding(num_items, embedding_size)
        self.item_bias_env = nn.Embedding(num_items, 1)

        self.env_emb = nn.Embedding(num_envs, embedding_size)
        self.env_bias = nn.Embedding(num_envs, 1)

        self.user_emb_inv.weight.data.uniform_(-0.01, 0.01)
        self.user_bias_inv.weight.data.uniform_(-0.01, 0.01)
        self.item_emb_inv.weight.data.uniform_(-0.01, 0.01)
        self.item_bias_inv.weight.data.uniform_(-0.01, 0.01)
        self.user_emb_env.weight.data.uniform_(-0.01, 0.01)
        self.user_bias_env.weight.data.uniform_(-0.01, 0.01)
        self.item_emb_env.weight.data.uniform_(-0.01, 0.01)
        self.item_bias_env.weight.data.uniform_(-0.01, 0.01)
        self.env_emb.weight.data.uniform_(-0.01, 0.01)
        self.env_bias.weight.data.uniform_(-0.01, 0.01)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(embedding_size, num_envs)

    def forward(self, users_id, items_id, envs_id, alpha=0):
        users_embed_invariant = self.user_emb_inv(users_id)
        items_embed_invariant = self.item_emb_inv(items_id)

        users_embed_env_aware = self.user_emb_env(users_id)
        items_embed_env_aware = self.item_emb_env(items_id)

        envs_embed = self.env_emb(envs_id)

        invariant_preferences = users_embed_invariant * items_embed_invariant
        env_aware_preferences = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score = torch.sum(invariant_preferences, dim=1) \
                          + self.user_bias_inv(users_id).view(-1) \
                          + self.item_bias_inv(items_id).view(-1)

        env_aware_mid_score = torch.sum(env_aware_preferences, dim=1) \
                              + self.user_bias_env(users_id).view(-1) \
                              + self.item_bias_env(items_id).view(-1) \
                              + self.env_bias(envs_id).view(-1)

        env_aware_score = invariant_score + env_aware_mid_score

        reverse_invariant_preferences = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.view(-1), env_aware_score.view(-1), env_outputs.view(-1, self.env_emb.num_embeddings)

    def predict(self, users_id, items_id):
        users_embed_invariant = self.user_emb_inv(users_id)
        items_embed_invariant = self.item_emb_inv(items_id)
        invariant_preferences = users_embed_invariant * items_embed_invariant

        invariant_score = torch.sum(invariant_preferences, dim=1) \
                          + self.user_bias_inv(users_id).view(-1) \
                          + self.item_bias_inv(items_id).view(-1)

        return invariant_score
