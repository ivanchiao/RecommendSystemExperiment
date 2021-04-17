# -*- coding: utf-8 -*-
"""
@author: LMC_ZC
"""

import pickle
import random
import torch
import numpy as np


# movielens-1m /lastfm360k
class FeaturesData(object):

    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name

    def load(self):
        d = np.load(self.data_path + '/' + self.dataset_name + '/processed/preprocess.npz', allow_pickle=True)
        train_set, test_set = d['train_set'], d['test_set']

        user_feat = d['user_feat']

        with open(self.data_path + '/' + self.dataset_name + '/processed/preprocess.pkl', 'rb') as f:
            train_U2I = pickle.load(f)
            test_U2I = pickle.load(f)
            n_users, n_items = pickle.load(f)

        return train_set, train_U2I, test_U2I, n_users, n_items, user_feat

# gowalla / amazon-book
class UserItemData(object):

    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name

    def load(self):
        with open(self.data_path + '/' + self.dataset_name + '/processed/preprocess.pkl', 'rb') as f:
            train_set = pickle.load(f)
            train_U2I = pickle.load(f)
            test_U2I = pickle.load(f)
            n_users, n_items = pickle.load(f)

            return train_set, train_U2I, test_U2I, n_users, n_items


class BPRTrainLoader(torch.utils.data.Dataset):

    def __init__(self, train_set, train_U2I, n_items):
        self.train_set = train_set
        self.train_U2I = train_U2I
        self.all_items = list(range(0, n_items))

    def __getitem__(self, index):
        user = self.train_set[index][0]
        pos = self.train_set[index][1]
        neg = random.choice(self.all_items)

        while neg in self.train_U2I[user]:
            neg = random.choice(self.all_items)

        return [user, pos, neg]

    def __len__(self):
        return len(self.train_set)