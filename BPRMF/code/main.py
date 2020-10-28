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

import sys
sys.path.append("/caimiaomiao/zc_experiments/NGCF/")


import torch.optim as optim
from torch.utils.data import DataLoader

from session import Session
from dataset import DATA, TrainDataset
from model import BPRMF
from utility.parser import parse_args
from performance import evaluate


def get_loader(train_set, train_U2I, n_items, batch_size, cores=6):
    train_ds = TrainDataset(train_set, train_U2I, n_items)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cores)

    return train_dl


class BPRMF_Args(object):

    def __init__(self, parser):
        self.dataset_name = parser.dataset_name
        self.data_path = parser.data_path
        self.batch_size = parser.batch_size
        self.num_epochs = parser.num_epochs
        self.n_factors = parser.n_factors
        self.decay = parser.decay
        self.lr = parser.lr
        self.topk = parser.topk
        self.cores = parser.cores


def run():

    args = BPRMF_Args(parse_args())

    # 获取训练集的dataloader形式
    data = DATA(args.data_path, args.dataset_name)
    train_set, train_U2I, test_U2I, n_users, n_items = data.load()
    train_dl = get_loader(train_set, train_U2I, n_items, args.batch_size, args.cores)

    # 定义网络
    model = BPRMF(n_users, n_items, args)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 定义会话
    sess = Session(model)

    for epoch in range(args.num_epochs):
        loss, mf_loss, emb_loss = sess.train(train_dl, optimizer)
        print("epoch: {:d}, loss = [{:.6f} == {:.6f} + {:.6f}]".format(epoch, loss, mf_loss, emb_loss))
        perf_info = evaluate(model, n_users, n_items, train_U2I, test_U2I, args)
        print("precision: [{:.6f}] recall: [{:.6f}] ndcg: [{:.6f}]".format(perf_info[0], perf_info[1], perf_info[2]))


if __name__ == '__main__':
    run()
