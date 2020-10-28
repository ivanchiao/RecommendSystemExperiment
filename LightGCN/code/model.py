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

class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, adj, args):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj = adj
        self.decay = args.decay
        self.batch_size = args.batch_size
        self.layers = args.layers
        self.emb_size = args.emb_size

        # embedding层
        self.user_embeddings = nn.Embedding(self.n_users, self.emb_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_size)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, user, pos_item, neg_item):

        users_emb = self.user_embeddings.weight
        items_emb = self.item_embeddings.weight
        node_feat = torch.cat([users_emb, items_emb], dim=0)

        all_emb = [node_feat]
        mat = self.adj

        for layer in range(self.layers):
            node_feat = torch.sparse.mm(mat, node_feat)
            all_emb += [node_feat]

        all_emb = torch.stack(all_emb, dim=1)
        light_out = torch.mean(all_emb, dim=1)

        gcn_user_embeddings, gcn_item_embeddings = torch.split(light_out, [self.n_users, self.n_items])
        user_emb = gcn_user_embeddings[user]
        pos_item_emb = gcn_item_embeddings[pos_item]
        neg_item_emb = gcn_item_embeddings[neg_item]

        return user_emb, pos_item_emb, neg_item_emb

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        pos_score = torch.sum(user_emb * pos_emb, dim=1)
        neg_score = torch.sum(user_emb * neg_emb, dim=1)
        mf_loss = torch.mean(F.softplus(neg_score - pos_score))

        reg_loss = 0.5 * (torch.norm(user_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2)
        reg_loss = self.decay * reg_loss / self.batch_size

        loss = mf_loss + reg_loss
        return loss, mf_loss, reg_loss