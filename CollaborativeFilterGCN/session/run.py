# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""


import torch
from utility.decorate import logger


class Session(object):

    def __init__(self, model):

        self.model = model
            
    @logger(begin_message=None, end_message=None)
    def train(self, loader, optimizer, args):
        
        self.model.train()
        all_loss, all_mf_loss, all_reg_loss = 0.0, 0.0, 0.0

        for uij in loader:

            u = uij[0].type(torch.long).cuda()
            i = uij[1].type(torch.long).cuda()
            j = uij[2].type(torch.long).cuda()
            
            user_emb, item_emb = self.model.propagate()
            user_emb, pos_emb, neg_emb = self.model.get_embedding(user_emb, item_emb, u, i, j)
            bpr_loss, mf_loss, reg_loss = self.model.bpr_loss(user_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            bpr_loss.backward()
            optimizer.step()

            all_loss += bpr_loss.item()
            all_mf_loss += mf_loss.item()
            all_reg_loss += reg_loss.item()
            
        mean_loss = all_loss / len(loader)
        mean_mf_loss = all_mf_loss / len(loader)
        mean_reg_loss = all_reg_loss / len(loader)

        return mean_loss, mean_mf_loss, mean_reg_loss