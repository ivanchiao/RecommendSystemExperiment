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
from utility.decorate import logger


class Session(object):

    def __init__(self, model):
        self.model = model

    def auc(self, user_emb, pos_emb, neg_emb):
        x = user_emb * (pos_emb - neg_emb)
        x = torch.sum(x, dim=1)
        x = (x > 0).float()
        return torch.mean(x)

    @logger(begin_message=None, end_message=None)
    def train(self, train_loader, optimizer):

        self.model.train()
        all_loss, all_mf_loss, all_emb_loss = .0, .0, .0
        for uij in train_loader:
            optimizer.zero_grad()

            u = uij[0].type(torch.long).cuda()
            i = uij[1].type(torch.long).cuda()
            j = uij[2].type(torch.long).cuda()

            user_emb, pos_emb, neg_emb = self.model(u, i, j)
            loss, mf_loss, emb_loss = self.model.bpr_loss(user_emb, pos_emb, neg_emb)

            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            all_mf_loss += mf_loss.item()
            all_emb_loss += emb_loss.item()

        mean_loss = all_loss / len(train_loader)
        mean_mf_loss = all_mf_loss / len(train_loader)
        mean_emb_loss = all_emb_loss / len(train_loader)

        return mean_loss, mean_mf_loss, mean_emb_loss

    def test(self, test_loader):
        self.model.eval()
        all_auc, all_loss, all_mf_loss, all_emb_loss = .0, .0, .0, .0

        with torch.no_grad():
            for uij in test_loader:
                u = uij[0][:, 0].cuda()
                i = uij[0][:, 1].cuda()
                j = uij[0][:, 2].cuda()

                user_emb, pos_emb, neg_emb = self.model(u, i, j)
                loss, mf_loss, emb_loss = self.model.bpr_loss(user_emb, pos_emb, neg_emb)

                all_loss += loss.item()
                all_mf_loss += mf_loss.item()
                all_emb_loss += emb_loss.item()
                all_auc += self.auc(user_emb, pos_emb, neg_emb)

            mean_auc = all_auc / len(test_loader)
            mean_loss = all_loss / len(test_loader)
            mean_mf_loss = all_mf_loss / len(test_loader)
            mean_emb_loss = all_emb_loss / len(test_loader)

            return mean_auc, mean_loss, mean_mf_loss, mean_emb_loss
