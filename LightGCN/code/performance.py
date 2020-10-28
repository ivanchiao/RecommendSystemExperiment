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
import multiprocessing as mp
from utility.decorate import logger
from utility.metrics import precision_k, recall_k, ndcg_k

@logger(begin_message=None, end_message=None)
def evaluate(model, n_users, n_items, train_U2I, test_U2I, args):
    with torch.no_grad():
        all_test_users = torch.arange(0, n_users).cuda()
        all_test_items = torch.arange(0, n_items).cuda()

        # 获得评分矩阵
        gcn_user_emb, gcn_item_emb, _, = model(all_test_users, all_test_items, torch.tensor([0]).cuda())
        scores = np.matmul(gcn_user_emb.cpu().numpy(), gcn_item_emb.cpu().numpy().T)

        perf_info = performance_speed(scores, n_users, n_items, train_U2I, test_U2I, args)
        perf_info = np.mean(perf_info, axis=0)
        
        return perf_info


def _init(_scores, _train_U2I, _test_U2I, _topk):
    global scores, train_user_item, test_user_item, topk
    scores = _scores
    train_user_item = _train_U2I
    test_user_item = _test_U2I
    topk = _topk


def performance_speed(scores, n_users, n_items, train_U2I, test_U2I, args):
    test_user_set = list(test_U2I.keys())

    # 测试3个指标
    perf_info = np.zeros((len(test_user_set), 3), dtype=np.float32)
    topk = args.topk

    # 多进程加速
    test_parameters = zip(test_user_set, )
    with mp.Pool(processes=4, initializer=_init,
                 initargs=(scores, train_U2I, test_U2I, topk,)) as pool:
        all_perf = pool.map(test_one_perf, test_parameters)

    for i, one_perf in enumerate(all_perf):
        perf_info[i] = one_perf
    return perf_info


def test_one_perf(X):
    u_id = X[0]
    score = scores[u_id]
    uid_train_pos_items = list(train_user_item[u_id])
    uid_test_pos_items = list(test_user_item[u_id])

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
    rank = np.zeros((score_indices.shape[-1],), dtype=np.int32)
    for i in range(score_indices.shape[-1]):
        if np.sum(score_indices[i] == uid_test_pos_items):
            rank[i] = 1

    return rank


def get_perf(rank, uid_test_pos_items, topk):
    topk_eval = np.zeros((3,), dtype=np.float32)
    topk_eval[0] = precision_k(rank, topk)
    topk_eval[1] = recall_k(rank, topk, len(uid_test_pos_items))
    topk_eval[2] = ndcg_k(rank, topk, len(uid_test_pos_items))

    return topk_eval
