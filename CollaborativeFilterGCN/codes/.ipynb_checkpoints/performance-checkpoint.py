# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""

import math
import torch
import numpy as np
import multiprocessing as mp
from utility.decorate import logger
from utility.metrics import recall_k, ndcg_k


@logger(begin_message=None, end_message=None)
def evaluate(user_emb, item_emb, n_users, n_items, train_U2I, test_U2I, args):

    scores = np.matmul(user_emb, item_emb.T)
    perf_info = performance_speed(scores,  train_U2I, test_U2I, args)
    perf_info = np.mean(perf_info, axis=0)

    return perf_info


def _init(_scores, _train_U2I, _test_U2I, _topks):
    global scores, train_user_item, test_user_item, topks
    scores = _scores
    train_user_item = _train_U2I
    test_user_item = _test_U2I
    topks = _topks


def performance_speed(scores, train_U2I, test_U2I, args):
    
    topks = eval(args.topks)
    
    test_user_set = list(test_U2I.keys())

    perf_info = np.zeros((len(test_user_set), 2 * len(topks)), dtype=np.float32)

    test_parameters = zip(test_user_set, )
    with mp.Pool(processes=args.cores, initializer=_init, initargs=(scores, train_U2I, test_U2I, topks,)) as pool:
        all_perf = pool.map(test_one_perf, test_parameters)

    for i, one_perf in enumerate(all_perf):
        perf_info[i] = one_perf

    return perf_info


def test_one_perf(X):
    u_id = X[0]
    score = scores[u_id]
    uid_train_pos_items = list(train_user_item[u_id])
    uid_test_pos_items = list(test_user_item[u_id])

    score[uid_train_pos_items] = -np.inf
    score_indices = largest_indices(score, topks)
    result = get_perf(score_indices, uid_test_pos_items, topks)

    return result


def largest_indices(score, topks):
    max_topk = max(topks)
    indices = np.argpartition(score, -max_topk)[-max_topk:]
    indices = indices[np.argsort(-score[indices])]
    return indices


def get_perf(rank, uid_test_pos_items, topks):
    topk_eval = np.zeros(2 * len(topks), dtype=np.float32)
    for i, topk in enumerate(topks):
        topk_eval[i * 2 + 0] = recall_k(rank[:topk], uid_test_pos_items)
        topk_eval[i * 2 + 1] = ndcg_k(rank[:topk], uid_test_pos_items)

    return topk_eval