# -*- coding: utf-8 -*-
"""
@author

██╗     ███╗   ███╗ ██████╗      ███████╗ ██████╗
██║     ████╗ ████║██╔════╝      ╚══███╔╝██╔════╝
██║     ██╔████╔██║██║     █████╗  ███╔╝ ██║
██║     ██║╚██╔╝██║██║     ╚════╝ ███╔╝  ██║
███████╗██║ ╚═╝ ██║╚██████╗      ███████╗╚██████╗
╚══════╝╚═╝     ╚═╝ ╚═════╝      ╚══════╝ ╚═════╝
"""


import torch
import numpy as np


class Movielens(object):

    def __init__(self, data_path, threshold):
        data = np.load(data_path + 'movielens-1m/processed/movielens-1m.npz')

        self.train_set = self.rating_threshold(data['train_set'], threshold)
        self.test_set = self.rating_threshold(data['test_set'], threshold)
        self.sparse_n_features = data['sparse_n_features']
        self.dense_n_features = data['dense_n_features']

        self.max_user_id = int(max(np.max(self.train_set[:, 0]), np.max(self.test_set[:, 0])) + 1)
        self.max_item_id = int(max(np.max(self.train_set[:, 1]), np.max(self.test_set[:, 1])) + 1)

    def rating_threshold(self, dataset, threshold):
        print(threshold)
        dataset[:, -1] = np.float32((dataset[:, -1] >= threshold))
        return dataset


class CustomerSet(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        X = self.dataset[index][:-1]
        Y = self.dataset[index][-1]
        return X, Y

    def __len__(self):
        return self.dataset.shape[0]


