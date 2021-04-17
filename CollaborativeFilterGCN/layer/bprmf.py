# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""

import torch
import torch.nn as nn


class BPRMF(nn.Module):

    def __init__(self, n_users, n_items, args, user_feat=None, item_feat=None):
        
        super(BPRMF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.decay = args.decay

        user_emb_weight = self._weight_init(user_feat, n_users, args.emb_size)
        item_emb_weight = self._weight_init(item_feat, n_items, args.emb_size)

        self.user_embeddings = nn.Embedding(self.n_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_size, _weight=item_emb_weight)

    def _weight_init(self, feat, rows, cols):

        if feat is None:
            return nn.init.normal_(torch.empty(rows, cols), std=0.01)
        else:
            free_emb = nn.init.normal_(torch.empty(rows, cols - feat.shape[-1]), std=0.01)
            feat_emb = torch.tensor(feat) * 0.01
            return torch.cat([free_emb, feat_emb], dim=1)
        
        
    def forward(self):
        
        all_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        
        return all_emb
    
    def split_emb(self, embeddings):
        
        user_emb, item_emb = torch.split(embeddings, [self.n_users, self.n_items])
        return user_emb, item_emb

    def get_embedding(self, user_emb, item_emb, users, pos_items, neg_items):
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        return u_emb, pos_emb, neg_emb
    
    def propagate(self):
        all_emb = self.forward()
        user_emb, item_emb = self.split_emb(all_emb)
        
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
