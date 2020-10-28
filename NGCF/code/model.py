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
import torch.nn as nn
import torch.nn.functional as F


class FeatTransform(nn.Module):

    def __init__(self, in_features, out_features, mess_dropout):
        super(FeatTransform, self).__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(mess_dropout)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, mat, node_feat):
        agg_feat = torch.sparse.mm(mat, node_feat)  # L * E

        part_1 = self.act(self.linear_1(agg_feat))
        part_2 = self.act(self.linear_2(agg_feat * node_feat))
        out = part_1 + part_2
        out = self.dropout(out)
        return out


class NGCF(nn.Module):

    def __init__(self, n_users, n_items, adj, args):
        super(NGCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.adj = adj  # norm-Laplace matrix

        self.decay = args.decay
        self.batch_size = args.batch_size
        self.decay = args.decay  # l2-norm coefficient
        self.layers = args.layers
        self.emb_size = args.emb_size
        self.node_dropout = args.node_dropout
        self.mess_dropout = args.mess_dropout

        self.dims = [self.emb_size] + self.layers   # transformation layers and embedding layers

        # embedding 层
        self.embeddings = nn.Embedding(self.n_users + self.n_items, self.emb_size)

        # node dropout layer
        self.dropout = nn.Dropout(args.node_dropout)

        # feature transform layer contains the message dropout
        self.layer_stack = nn.ModuleList([
            FeatTransform(self.dims[i], self.dims[i + 1], self.mess_dropout[i])
            for i in range(len(self.layers))
        ])

        # initial
        nn.init.xavier_uniform_(self.embeddings.weight)

    def dropout_sparse(self):
        i = self.adj._indices()
        v = self.adj._values()
        v = self.dropout(v)
        adj = torch.sparse_coo_tensor(i, v, self.adj.shape).to(self.adj.device)
        return adj

    def forward(self, user, pos_item, neg_item):
        node_feat = self.embeddings.weight  # [user, item] embedding矩阵
        mat = self.dropout_sparse()  # node dropout 后的拉普拉斯矩阵
        all_embeddings = [node_feat]

        for feat_trans in self.layer_stack:
            node_feat = feat_trans(mat, node_feat)
            all_embeddings += [F.normalize(node_feat)]

        embedding_weight = torch.cat(all_embeddings, -1)
        gcn_user_embeddings, gcn_item_embeddings = torch.split(embedding_weight, [self.n_users, self.n_items], dim=0)

        user_emb = gcn_user_embeddings[user, :]
        pos_item_emb = gcn_item_embeddings[pos_item, :]
        neg_item_emb = gcn_item_embeddings[neg_item, :]

        return user_emb, pos_item_emb, neg_item_emb

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        """
        bpr loss function
        """
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        criterion = nn.LogSigmoid()
        mf_loss = criterion(pos_scores - neg_scores)
        mf_loss = -1.0 * torch.mean(mf_loss)

        regularizer = 0.5 * (torch.norm(user_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2)
        emb_loss = self.decay * regularizer / self.batch_size

        loss = mf_loss + emb_loss

        return loss, mf_loss, emb_loss
