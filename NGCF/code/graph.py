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


class Graph(object):

    def __init__(self, edge_indexs, edge_weight):

        self.weights = torch.tensor(edge_weight, dtype=torch.float32)
        self.indices = torch.tensor(edge_indexs, dtype=torch.long)

    @property
    def num_nodes(self):
        nodes = torch.unique(self.indices[0])
        return nodes.shape[0]

    @property
    def num_edges(self):
        return len(self.weights)

    @property
    def mat(self):
        return torch.sparse.FloatTensor(self.indices, self.weights, torch.Size([self.num_nodes, self.num_nodes]))

    def add_self_loop(self):
        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        self.indices = torch.cat([self.indices, loop_index], dim=-1)
        self.weights = torch.cat([self.weights, loop_weight], dim=-1)

    def norm(self):
        row, col = self.indices[0], self.indices[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, self.weights)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        self.weights = deg_inv_sqrt[row] * self.weights * deg_inv_sqrt[col]