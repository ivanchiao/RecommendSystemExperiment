# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""


import torch
import numpy as np


class Graph(object):

    def __init__(self, n_users, n_items, train_U2I):

        self.n_users = n_users
        self.n_items = n_items
        self.train_U2I = train_U2I

    def to_edge(self):

        train_U, train_I = [], []

        for u, items in self.train_U2I.items():
            train_U.extend([u] * len(items))
            train_I.extend(items)

        train_U = np.array(train_U)
        train_I = np.array(train_I)

        row = np.concatenate([train_U, train_I + self.n_users])
        col = np.concatenate([train_I + self.n_users, train_U])

        edge_weight = np.ones_like(row).tolist()
        edge_index = np.stack([row, col]).tolist()

        return edge_index, edge_weight


class LaplaceGraph(Graph):

    def __init__(self, n_users, n_items, train_U2I):
        Graph.__init__(self, n_users, n_items, train_U2I)

    def generate(self):
        edge_index, edge_weight = self.to_edge()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)
        edge_index, edge_weight = self.norm(edge_index, edge_weight)

        return self.mat(edge_index, edge_weight)

    def add_self_loop(self, edge_index, edge_weight):
        """ add self-loop """

        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)

        return edge_index, edge_weight

    def norm(self, edge_index, edge_weight):
        """ D^{-1/2} * A * D^{-1/2}"""

        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    @property
    def num_nodes(self):
        return self.n_users + self.n_items

    def mat(self, edge_index, edge_weight):
        return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([self.num_nodes, self.num_nodes]))