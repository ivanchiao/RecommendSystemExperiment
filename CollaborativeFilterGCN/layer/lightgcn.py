# -*- coding: utf-8 -*-
"""

@author: LMC_ZC

SIGIR-2020
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, adj, args, user_feat=None, item_feat=None):
        
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.decay = args.decay
        self.layers = args.layers
        self.adj = adj

        user_emb_weight = self._weight_init(user_feat, n_users, args.emb_size)
        item_emb_weight = self._weight_init(item_feat, n_items, args.emb_size)

        self.user_embeddings = nn.Embedding(self.n_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_size, _weight=item_emb_weight)

    def _weight_init(self, feat, rows, cols):

        if feat is None:
            free_emb = nn.init.normal_(torch.empty(rows, cols), std=0.01)
            return free_emb
        else:
            free_emb = nn.init.normal_(torch.empty(rows, cols - feat.shape[-1]), std=0.01)
            feat_emb = torch.tensor(feat) * 0.01
            return torch.cat([free_emb, feat_emb], dim=1)

    def forward(self, adj):

        x = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

        all_emb = [x]

        for _ in range(self.layers):
            x = torch.sparse.mm(adj, x)
            all_emb += [x]

        return all_emb

    def fusion(self, embeddings):
        
        # differ from lr-gccf
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = torch.mean(embeddings, dim=1)
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
        
        pos_score = torch.sum(user_emb * pos_emb, dim=1)
        neg_score = torch.sum(user_emb * neg_emb, dim=1)
        mf_loss = torch.mean(F.softplus(neg_score - pos_score))
        
        reg_loss = (1/2) * (user_emb.norm(2).pow(2) + 
                            pos_emb.norm(2).pow(2) + 
                            neg_emb.norm(2).pow(2)) / user_emb.shape[0] * self.decay

        loss = mf_loss + reg_loss
        return loss, mf_loss, reg_loss
