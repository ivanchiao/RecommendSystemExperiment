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
import numpy as np
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import DataLoader

from init import *
from FM.utility.decorate import logger
import metrics

from model import LR_GCCF
from amazon_data import Amazon_Data, TrainDataset


class LR_GCCF_Args(object):

    def __init__(self, epoch=50, layers=3, n_factors=64, decay=0.01,
                 lr=0.001, batch_size=2048, topk=20):
        self.epoch = epoch
        self.layers = layers
        self.n_factors = n_factors
        self.decay = decay
        self.lr = lr
        self.batch_size = batch_size
        self.topk = topk


def sparse_to_tensor(sparse_matrix):
    """
    sparse_matrix : coo_matrix
    """

    i = torch.LongTensor([sparse_matrix.row, sparse_matrix.col])
    v = torch.from_numpy(sparse_matrix.data)

    return torch.sparse.FloatTensor(i, v, sparse_matrix.shape)


def data_process(amazon_book):
    user_item_matrix = amazon_book.user_item_sparse
    item_user_matrix = amazon_book.item_user_sparse
    d_users = amazon_book.d_users
    d_items = amazon_book.d_items

    user_item_matrix = sparse_to_tensor(user_item_matrix)
    item_user_matrix = sparse_to_tensor(item_user_matrix)

    d_users = np.expand_dims(np.array(d_users, dtype=np.float32), axis=1)
    d_items = np.expand_dims(np.array(d_items, dtype=np.float32), axis=1)
    d_users, d_items = torch.tensor(d_users), torch.tensor(d_items)

    """
    # d_user, d_items 转换成对角稀疏阵
    d_users = np.expand_dims(np.array(d_users), axis = 0)
    d_items = np.expand_dims(np.array(d_items), axis = 0)
    d_user_sparse_matrix = sp.dia_matrix( (d_users, np.array([0])), dtype = np.float32,
            shape = (amazon_book.n_users, amazon_book.n_users))
    d_item_sparse_matrix = sp.dia_matrix( (d_items, np.array([0])), dtype = np.float32,
            shape = (amazon_book.n_items, amazon_book.n_items))
    d_users = sparse_to_tensor(d_user_sparse_matrix.tocoo())
    d_items = sparse_to_tensor(d_item_sparse_matrix.tocoo())
    """

    return user_item_matrix, item_user_matrix, d_users, d_items


def get_Dataset(amazon_book, args):
    train_set = amazon_book.train_uij
    train_U2I = amazon_book.train_user_item
    n_items = amazon_book.n_items

    train_ds = TrainDataset(train_set, train_U2I, n_items)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4)
    return train_dl


@logger(begin_message = None, end_message = None)
def train(model, train_dl, optimizer):
    model.train()
    loss, mf_loss, emb_loss = 0.0, 0.0, 0.0

    for uij in train_dl:
        optimizer.zero_grad()
        u = uij[0].type(torch.long).cuda()
        i = uij[1].type(torch.long).cuda()
        j = uij[2].type(torch.long).cuda()

        u_emb, i_emb, j_emb = model(u, i, j)
        batch_loss, batch_mf_loss, batch_emb_loss = model.bpr_loss(u_emb, i_emb, j_emb)
        batch_loss = batch_loss.cuda()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss
        mf_loss += batch_mf_loss
        emb_loss += batch_emb_loss

    print("loss = [{:.6f} == {:.6f} + {:.6f}]".format(loss / len(train_dl), mf_loss/ len(train_dl), emb_loss/ len(train_dl)))


def test(model, test_dl):
    model.eval()

    loss = 0
    # auc = 0

    with torch.no_grad():
        for uij in test_dl:
            u = uij[0][:, 0].cuda()
            i = uij[0][:, 1].cuda()
            j = uij[0][:, 2].cuda()

            u_emb, i_emb, j_emb = model(u, i, j)
            batch_loss = model.bpr_loss(u_emb, i_emb, j_emb)
            loss += batch_loss

            """
            x = u_emb * (i_emb - j_emb)
            x = torch.sum(x, dim=1)
            x = (x > 0).float()
            auc += torch.mean(x)
            """

        print("test_loss:{:.4f}".format(loss / len(test_dl)))


@logger(begin_message = None, end_message=None)
def evaluate(model, amazon_book, args):
    with torch.no_grad():
        all_test_users = torch.arange(0, amazon_book.n_users).cuda()
        all_test_items = torch.arange(0, amazon_book.n_items).cuda()

        gcn_user_emb, gcn_item_emb, _ = model(all_test_users, all_test_items, torch.tensor([0]).cuda())
        scores = np.matmul(gcn_user_emb.cpu().numpy(), gcn_item_emb.cpu().numpy().T)

        perf_info = performance_speed(scores, amazon_book, args)
        perf_info = np.mean(perf_info, axis=0)
        print("precision: {:.6f}  recall: {:.6f}  ndcg: {:.6f}\n".format(
                perf_info[0], perf_info[1], perf_info[2]))

        return perf_info[0], perf_info[1], perf_info[2]


def _init(_scores, _all_items, _topk, _amazon_book):
    global scores, all_items, topk, amazon_book
    scores = _scores
    all_items = _all_items
    topk = _topk
    amazon_book = _amazon_book


def performance_speed(_scores, _amazon_book, _args):
    test_user_set = list(_amazon_book.test_user_item.keys())
    _all_items = set(range(_amazon_book.n_items))

    # 测试四个指标
    perf_info = np.zeros((len(test_user_set), 4), dtype=np.float32)
    _topk = _args.topk

    test_parameters = zip(test_user_set, )
    with mp.Pool(processes=6, initializer=_init, initargs=(_scores, _all_items, _topk, _amazon_book, )) as pool:
        all_perf = pool.map(test_one_perf, test_parameters)

    for i, one_perf in enumerate(all_perf):
        perf_info[i] = one_perf
    return perf_info


def test_one_perf(X):
    u_id = X[0]
    score = scores[u_id]
    uid_train_pos_items = list(amazon_book.train_user_item[u_id])
    uid_test_pos_items = list(amazon_book.test_user_item[u_id])

    # 不考虑出现在训练集的items
    score[uid_train_pos_items] = -np.inf

    score_indices = largest_indices(score, topk)
    rank = rank_list(score_indices, np.array(uid_test_pos_items, dtype=np.float32))
    result = get_perf(rank, uid_test_pos_items, topk)

    return result


def largest_indices(score, topk):
    indices = np.argpartition(score, -topk)[-topk:]
    indices = indices[np.argsort(-score[indices])]
    return indices


def rank_list(score_indices, uid_test_pos_items):
    rank = np.zeros((score_indices.shape[-1], ), dtype=np.int32)
    for i in range(score_indices.shape[-1]):
        if np.sum(score_indices[i] == uid_test_pos_items):
            rank[i] = 1

    return rank


def get_perf(rank, uid_test_pos_items, topk):
    topk_eval = np.zeros((4,), dtype=np.float32)
    topk_eval[0] = metrics.precision_k(rank, topk)
    topk_eval[1] = metrics.recall_k(rank, topk, len(uid_test_pos_items))
    topk_eval[2] = metrics.ndcg_k(rank, topk, len(uid_test_pos_items))
    return topk_eval


def main():
    lr_gccf_args = LR_GCCF_Args()
    amazon_book = Amazon_Data(data_path)
    user_item_matrix, item_user_matrix, d_users, d_items = data_process(amazon_book)
    train_dl = get_Dataset(amazon_book, lr_gccf_args)

    model = LR_GCCF(amazon_book.n_users, amazon_book.n_items, lr_gccf_args.batch_size, lr_gccf_args.n_factors,
                    user_item_matrix, item_user_matrix, d_users, d_items, lr_gccf_args.layers, lr_gccf_args.decay)
    model = model.cuda()
    optimizer = optim.Adam(params=model.parameters(), lr=lr_gccf_args.lr)

    print("----------training----------")
    precision, recall, ndcg = [], [], []
    for k in range(lr_gccf_args.epoch):
        print("epoch: " + str(k))
        train(model, train_dl, optimizer)
        #test(model, test_dl)
        d1, d2, d3 = evaluate(model, amazon_book, lr_gccf_args)
        precision.append(d1)
        recall.append(d2)
        ndcg.append(d3)

    return precision, recall, ndcg


if __name__ == '__main__':
    precision, recall, ndcg = main()