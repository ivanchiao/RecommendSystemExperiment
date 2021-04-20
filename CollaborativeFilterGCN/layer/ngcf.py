# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

SIGIR-2019
Neural Graph Collaborative Filtering
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
        self.layers = eval(args.layers)
        self.emb_size = args.emb_size
        self.node_dropout = args.node_dropout
        self.mess_dropout = eval(args.mess_dropout)

        self.dims = [self.emb_size] + self.layers   # transformation layers and embedding layers

        # embedding
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

    def dropout_sparse(self, adj):
        i = adj._indices()
        v = adj._values()
        v = self.dropout(v)
        drop_adj = torch.sparse_coo_tensor(i, v, adj.shape).to(adj.device)
        return drop_adj
    
    def forward(self, adj):
        node_feat = self.embeddings.weight
        mat = self.dropout_sparse(adj)
        all_emb = [node_feat]
        
        for feat_trans in self.layer_stack:
            node_feat = feat_trans(mat, node_feat)
            all_emb += [F.normalize(node_feat)]
        return all_emb
    
    def fusion(self, embeddings):

        embeddings = torch.cat(embeddings, dim=1)
        return embeddings
    
    def split_emb(self, embeddings):
        
        user_emb, item_emb = torch.split(embeddings, [self.n_users, self.n_items])
        return user_emb, item_emb
    
    def get_embedding(self, user_emb, item_emb, users, pos_items, neg_items):
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        return u_emb, pos_emb, neg_emb
    
    def propagate(self):
        all_emb = self.forward(self.adj)
        f_emb = self.fusion(all_emb)
        user_emb, item_emb = self.split_emb(f_emb)
        
        return user_emb, item_emb
        
    def bpr_loss(self, user_emb, pos_emb, neg_emb):

        # bpr loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        criterion = nn.LogSigmoid()

        mf_loss = criterion(pos_scores - neg_scores)
        mf_loss = -1.0 * torch.mean(mf_loss)

        # reg loss
        regularizer = 0.5 * (torch.norm(user_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2)
        emb_loss = self.decay * regularizer / self.batch_size

        loss = mf_loss + emb_loss

        return loss, mf_loss, emb_loss
