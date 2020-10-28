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


class BPRMF(nn.Module):

    def __init__(self, n_users, n_items, args):
        super(BPRMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = args.n_factors
        self.decay = args.decay
        self.batch_size = args.batch_size

        self.user_embeddings = nn.Embedding(self.n_users, self.n_factors)
        self.item_embeddings = nn.Embedding(self.n_items, self.n_factors)

    def _init(self):
        # initial
        #nn.init.xavier_uniform_(self.user_embeddings.weight)
        #nn.init.xavier_uniform_(self.item_embeddings.weight)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)


    def forward(self, user, pos_item, neg_item):

        return self.user_embeddings.weight[user, :], self.item_embeddings.weight[pos_item, :], self.item_embeddings.weight[neg_item, :]

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
