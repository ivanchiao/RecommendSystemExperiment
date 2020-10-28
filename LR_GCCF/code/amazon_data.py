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
import random
import numpy as np
from collections import defaultdict
import scipy.sparse as sp


class Amazon_Data(object):

    def __init__(self, data_path):

        train_path = data_path + '/train.txt'
        test_path = data_path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.train_user_item = self.load_dataset(train_path)
        self.test_user_item = self.load_dataset(test_path)
        self.n_users += 1
        self.n_items += 1

        self.d_users, self.d_items = self.get_D(self.train_user_item)

        self.user_item_sparse, self.item_user_sparse = self.get_sparse_user_item(self.train_user_item)

        self.train_uij = []
        for u, items in self.train_user_item.items():
            for i in items:
                self.train_uij.append([u, i])

    def load_dataset(self, root_path):
        """
        load .txt data
        """

        user_item = defaultdict(set)
        with open(root_path, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break

                line = line.strip('\n').strip(' ')
                text = line.split(" ")
                text = list(map(lambda x: int(x), text))

                self.n_users = max(text[0], self.n_users)
                if text[1:]:
                    self.n_items = max(max(text[1:]), self.n_items)
                    user_item[text[0]] = set(text[1:])

        return user_item

    def get_sparse_user_item(self, user_item):
        """
        get user_item matrix and item_user matrix
        """

        matrix_x, matrix_y = [], []
        matrix_value = []

        for u, items in user_item.items():
            for i in items:
                matrix_x.append(u)  # save the indices
                matrix_y.append(i)
                dij = np.sqrt(self.d_users[u] * self.d_items[i])
                matrix_value.append(dij)

        user_item_sparse = sp.coo_matrix((matrix_value, (matrix_x, matrix_y)),
                                         shape=(self.n_users, self.n_items),
                                         dtype=np.float32)

        item_user_sparse = (user_item_sparse.copy()).T

        return user_item_sparse, item_user_sparse

    def get_D(self, user_item):

        d_user = np.zeros(self.n_users, dtype=np.float32)
        d_item = np.zeros(self.n_items, dtype=np.float32)

        item_user = defaultdict(set)

        for u in user_item.keys():
            len_set = 1.0 / (len(user_item[u]) + 1)

            for i in user_item[u]:
                item_user[i].add(u)
            d_user[u] = len_set

        for u in item_user.keys():
            len_set = 1.0 / (len(item_user[u]) + 1)
            d_item[u] = len_set

        return d_user, d_item


class TrainDataset(torch.utils.data.Dataset):

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