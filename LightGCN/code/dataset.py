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
import pickle
import random


class DATA(object):

    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name

    def load(self):
        with open(self.data_path + '/processed/pre_' + self.dataset_name + '.pkl', 'rb') as f:
            train_set = pickle.load(f)
            train_U2I = pickle.load(f)
            test_U2I = pickle.load(f)
            edge_indices = pickle.load(f)
            edge_weight = pickle.load(f)
            n_users, n_items = pickle.load(f)

            return train_set, train_U2I, test_U2I, edge_indices, edge_weight, n_users, n_items


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, train_U2I, n_items):
        self.train_set = train_set
        self.train_U2I = train_U2I
        self.all_items = list(range(0, n_items))
        print('funck')

    def __getitem__(self, index):

        user = self.train_set[index][0]
        pos = self.train_set[index][1]
        neg = random.choice(self.all_items)

        while neg in self.train_U2I[user]:
            neg = random.choice(self.all_items)

        return [user, pos, neg]

    def __len__(self):
        return len(self.train_set)


"""
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_user_item, num_neg_items=1):

        self.dataset = []
        for u, items in train_user_item.items():
            temp_u = np.array([u] * len(items) * num_neg_items, dtype=np.long)
            temp_i = np.array(list(items) * num_neg_items, dtype=np.long)
            temp = np.vstack((temp_u, temp_i))
            self.dataset.extend(temp.T)

        self.train_user_item = train_user_item
        self.all_items = list(range(len(train_user_item)))
        # self.dataset = [ [u].extend(items)  for u, items in train_user_item.items()]

    def __getitem__(self, index):
        user = self.dataset[index][0]
        pos_item = self.dataset[index][1]
        neg_item = random.choice(self.all_items)
        while neg_item in self.train_user_item[user]:
            neg_item = random.choice(self.all_items)

        # neg_pool = set(self.all_items) - pos
        # np.random.shuffle(neg_pool)
        # neg = neg_pool[0]

        return [user, pos_item, neg_item]

    def __len__(self):
        return len(self.dataset)
"""

