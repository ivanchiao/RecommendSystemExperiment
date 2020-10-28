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


class LR_GCCF(nn.Module):

    def __init__(self, n_users, n_items, batch_size, n_factors, user_item_matrix,
                 item_user_matrix, d_users, d_items, layers, decay):
        super(LR_GCCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.n_factors = n_factors

        self.user_item_matrix = user_item_matrix.cuda()
        self.item_user_matrix = item_user_matrix.cuda()
        self.d_users = d_users.cuda()
        self.d_items = d_items.cuda()

        self.layers = layers
        self.decay = decay
        self.emb_user = nn.Embedding(self.n_users, self.n_factors)
        self.emb_item = nn.Embedding(self.n_items, self.n_factors)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.emb_user.weight, std=0.01)
        nn.init.normal_(self.emb_item.weight, std=0.01)

    def forward(self, users, pos_item, neg_item):

        gcn_user_embedding = [self.emb_user.weight]
        gcn_item_embedding = [self.emb_item.weight]

        for k in range(self.layers):
            gcn_user_embedding.append(torch.sparse.mm(self.user_item_matrix, gcn_item_embedding[-1]) + \
                                      gcn_user_embedding[-1].mul(self.d_users))
            gcn_item_embedding.append(torch.sparse.mm(self.item_user_matrix, gcn_user_embedding[-2]) + \
                                      gcn_item_embedding[-1].mul(self.d_items))

        gcn_user_embedding = torch.cat(gcn_user_embedding, dim=1)
        gcn_item_embedding = torch.cat(gcn_item_embedding, dim=1)

        u_embeddings = gcn_user_embedding[users,:]
        pos_i_embeddings = gcn_item_embedding[pos_item, :]
        neg_i_embeddings = gcn_item_embedding[neg_item, :]

        return u_embeddings, pos_i_embeddings, neg_i_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        regularizer = torch.mean(torch.sum(users ** 2 + pos_items ** 2 + neg_items ** 2, dim=1))

        emb_loss = self.decay * regularizer
        loss = mf_loss + emb_loss

        return loss, mf_loss, emb_loss