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
from graph import Graph
from model import LightGCN
from utility.parser import parse_args
from performance import evaluate


def get_loader(train_set, train_U2I, n_items, batch_size, cores=6):
    train_ds = TrainDataset(train_set, train_U2I, n_items)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cores)

    return train_dl


class LightGCN_Args(object):

    def __init__(self, parser):
        self.dataset_name = parser.dataset_name
        self.data_path = parser.data_path
        self.batch_size = parser.batch_size
        self.num_epochs = parser.num_epochs
        self.emb_size = parser.emb_size
        self.layers = parser.layers
        self.decay = parser.decay
        self.lr = parser.lr
        self.topk = parser.topk
        self.node_dropout = parser.node_dropout
        self.mess_dropout = eval(parser.mess_dropout)
        self.node_dropout_flag = eval(parser.node_dropout_flag)
        self.cores = parser.cores


def run():

    args = LightGCN_Args(parse_args())

    # 获取训练集的dataloader形式
    data = DATA(args.data_path, args.dataset_name)
    train_set, train_U2I, test_U2I, edge_indices, edge_weight, n_users, n_items = data.load()
    train_dl = get_loader(train_set, train_U2I, n_items, args.batch_size, args.cores)

    # 获取归一化的拉普拉斯矩阵
    laplace_graph = Graph(edge_indices, edge_weight)
    laplace_graph.add_self_loop()
    laplace_graph.norm()
    norm_adj = laplace_graph.mat.cuda()

    # 定义网络
    model = LightGCN(n_users, n_items, norm_adj, args)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 定义会话
    sess = Session(model)

    for epoch in range(args.num_epochs):
        loss, mf_loss, emb_loss = sess.train(train_dl, optimizer)
        print("epoch: {:d}, loss = [{:.6f} == {:.6f} + {:.6f}]".format(epoch, loss, mf_loss, emb_loss))
        #perf_info = evaluate(model, n_users, n_items, train_U2I, test_U2I, args)
        #print("precision: [{:.6f}] recall: [{:.6f}] ndcg: [{:.6f}]".format(perf_info[0], perf_info[1], perf_info[2]))


if __name__ == '__main__':
    run()
