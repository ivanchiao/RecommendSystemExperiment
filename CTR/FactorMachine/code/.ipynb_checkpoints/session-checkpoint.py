# -*- coding: utf-8 -*-
"""
@author: LMC_ZC
"""

import torch
import numpy as np
from utility.decorate import logger
from sklearn.metrics import roc_auc_score


class Session(object):
    def __init__(self, model, max_user_id, max_item_id):
        self.model = model
        self.max_user_id = max_user_id
        self.max_item_id = max_item_id

    def to_onehot(self, X):
        """
        In this function, we use the sparsity-form to storage the one-hot vector.
        """
        user_id = X[:, 0]
        item_id = X[:, 1]

        batch_size = X.shape[0]
        item_id = item_id + self.max_user_id # 列方向的偏移

        rows = np.hstack((np.arange(batch_size), np.arange(batch_size)))
        cols = np.hstack((user_id, item_id))

        i = torch.LongTensor([rows, cols])
        v = torch.ones(batch_size * 2, )

        ui_sparse = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, int(self.max_user_id + self.max_item_id)]))

        return ui_sparse, X[:, 2:]

    @logger(begin_message=None, end_message=None)
    def train(self, train_loader, optimizer):

        self.model.train()
        loss, cross_loss, l2_loss = 0.0, 0.0, 0.0

        for d in train_loader:
            
            X = d[0]
            Y = d[1]

            ui_sparse, other_features = self.to_onehot(X)

            ui_sparse, other_features, y_truth = ui_sparse.cuda(), other_features.cuda(), Y.cuda()
            y_pred = self.model(ui_sparse, other_features)
            batch_loss, batch_cross_loss, batch_l2_loss = self.model.loss_func(y_truth, y_pred)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            cross_loss += batch_cross_loss.item()
            l2_loss += batch_l2_loss.item()

        loss = loss / len(train_loader)
        cross_loss = cross_loss / len(train_loader)
        l2_loss = l2_loss / len(train_loader)

        return loss, cross_loss, l2_loss

    @logger(begin_message=None, end_message=None)
    def test(self, test_loader):
        self.model.eval()

        loss, cross_loss, l2_loss = 0.0, 0.0, 0.0
        auc = 0.0

        with torch.no_grad():
            for d in test_loader:
                X = d[0]
                Y = d[1]

                ui_sparse, other_features = self.to_onehot(X)
                ui_sparse, other_features, y_truth = ui_sparse.cuda(), other_features.cuda(), Y.cuda()

                y_pred = self.model(ui_sparse, other_features)
                batch_loss, batch_cross_loss, batch_l2_loss = self.model.loss_func(y_truth, y_pred)

                y_pred = y_pred.cpu().numpy()
                Y = Y.cpu().numpy()

                auc += roc_auc_score(Y, y_pred)
                loss += batch_loss.item()
                cross_loss += batch_cross_loss.item()
                l2_loss += batch_l2_loss.item()

            loss = loss / len(test_loader)
            cross_loss = cross_loss / len(test_loader)
            l2_loss = l2_loss / len(test_loader)
            auc = auc / len(test_loader)

        return loss, cross_loss, l2_loss, auc