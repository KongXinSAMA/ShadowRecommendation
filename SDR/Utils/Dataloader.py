from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class MFRatingDataset(Dataset):
    def __init__(self, uid, iid, rating, require_index=False):
        self.uid = uid
        self.iid = iid
        self.rating = rating
        self.index = None
        if require_index:
            self.index = torch.Tensor(np.arange(0, self.uid.shape[0])).type(torch.long)

    def __getitem__(self, index):
        if self.index is None:
            return self.uid[index], self.iid[index], self.rating[index]
        else:
            return self.uid[index], self.iid[index], self.rating[index], self.index[index]

    def __len__(self):
        return len(self.rating)

class MFDataset(Dataset):
    def __init__(self, uid, iid):
        self.uid = uid
        self.iid = iid

    def __getitem__(self, index):
        return self.uid[index], self.iid[index]

    def __len__(self):
        return len(self.uid)

class iVae_TrainingDataset(Dataset):
    def __init__(self, data, feature):
        self.data = data
        self.feature = feature

    def __getitem__(self, index):
        return self.data[index], self.feature[index]

    def __len__(self):
        return len(self.data)

class Shadow_TrainingDataset(Dataset):
    def __init__(self, Y_data, O_data, Feature):
        self.Y_data = Y_data
        self.O_data = O_data
        self.Feature = Feature

    def __getitem__(self, index):
        return self.Y_data[index], self.O_data[index], self.Feature[index]

    def __len__(self):
        return len(self.Y_data)


class InvPref_TrainingDataset(Dataset):
    def __init__(self, feature, rating, envs):
        self.feature = feature
        self.rating = rating
        self.envs = envs

    def __getitem__(self, index):
        if self.envs is None:
            return self.feature[index], self.rating[index]
        return self.feature[index], self.rating[index], self.envs[index]

    def __len__(self):
        return len(self.rating)

class Base_TrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)