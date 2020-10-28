# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:28:48 2020

@author: LMC_ZC
"""

import numpy as np

def precision_k(rank, topk):
    rank = np.array(rank)[:topk]
    return np.mean(rank)


def recall_k(rank, topk, num_pos_item):
    if num_pos_item == 0:
        return 0

    rank = np.array(rank)[:topk]
    return np.sum(rank) / num_pos_item


def F1(pred, rec):
    if pred + rec > 0:
        return (2.0 * pred * rec) / (pred + rec)
    else:
        return 0


def ndcg_k(rank, topk, num_pos_item):
    if num_pos_item == 0:
        return 0

    min_len = min(topk, num_pos_item)
    max_len = max(topk, num_pos_item)

    val = 1.0 / np.log2(np.arange(2, topk + 2))

    idcg = np.sum(val[:min_len])
    dcg = np.sum(rank * val[:topk])
    return dcg / idcg
