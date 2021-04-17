# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""


root_path = '/zhaochen/RS_FIRST_EXPERIMENT/CTR/FactorMachine/'

import os
import sys
sys.path.append(root_path)

from model import FactorMachine
from utility.parser import parse_args
from dataset import Movielens, CustomerSet
from torch.utils.data import DataLoader
import torch.optim as optim
from session import Session

def get_Dataloader(train_set, test_set, batch_size, cores):

    train_ds, test_ds = CustomerSet(train_set), CustomerSet(test_set)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cores)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=cores)

    return train_dl, test_dl


def run():
    parser = parse_args()
    d = Movielens(parser.data_path, parser.threshold)
    train_dl, test_dl = get_Dataloader(d.train_set, d.test_set, parser.batch_size, parser.cores)

    model = FactorMachine(d.sparse_n_features, d.dense_n_features, parser.n_classes,
                          parser.n_factors, parser.batch_size, parser.decay)
    
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=parser.lr)

    sess = Session(model, d.max_user_id, d.max_item_id)

    for epoch in range(parser.num_epoch):
        loss, cross_loss, l2_loss = sess.train(train_dl, optimizer)
        print("epoch: {:d}, loss = [{:.6f} == {:.6f} + {:.6f}]".format(epoch, loss, cross_loss, l2_loss))
        loss, cross_loss, l2_loss, auc = sess.test(test_dl)
        print("loss = [{:.6f} == {:.6f} + {:.6f}], auc = [{:.6f}]".format(loss, cross_loss, l2_loss, auc))
        print('\n')


if __name__ == '__main__':
    run()