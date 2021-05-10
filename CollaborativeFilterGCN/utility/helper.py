# -*- coding: utf-8 -*-
"""

@author : LMC_ZC
"""

import numpy as np

def record_perf(*args, recall=[], ndcg=[], statistial_parity=[], equal_opportunity=[]):
    recall += [args[0]]
    ndcg += [args[1]]
    statistial_parity += [args[2]]
    equal_opportunity += [args[3]]
    return recall, ndcg, statistial_parity, equal_opportunity


def isbestperf(perf):
    if perf[:-1] == [] or perf[-1] > max(perf[:-1]):
        return True
    else:
        return False


def early_stopping(evaluates, inval=10):
    if len(evaluates) <= inval:
        return False
    else:
        for i in range(len(evaluates) - inval - 1, len(evaluates) - 1):
            if evaluates[i] < evaluates[i + 1]:
                return False
        return True
    
def fold_matmul(a, b, n_fold=0):

    # this function is used to calculate the matrix matmul
    # if your memory space is small, directly matmul two large matrix is costly, and may be out of memory.
    # so you can set n_fold to compute block matrix multiplication.

    if n_fold == 0 or n_fold == 1:
        return np.matmul(a, b)

    a_row, a_col = a.shape[0], a.shape[1]
    b_row, b_col = b.shape[0], b.shape[1]

    a_per_row = a_row // n_fold
    b_per_col = b_col // n_fold

    blocks = []
    for i in range(n_fold):
        for j in range(n_fold):
            if i == n_fold - 1 and j == n_fold - 1:
                res = np.matmul(a[i * a_per_row:, :], b[:, j * b_per_col:])
                blocks += [res]
                continue
            if i == n_fold - 1:
                res = np.matmul(a[i * a_per_row:, :], b[:, j * b_per_col:(j + 1) * b_per_col])
                blocks += [res]
                continue
            if j == n_fold - 1:
                res = np.matmul(a[i * a_per_row:(i + 1) * a_per_row, :], b[:, j * b_per_col:])
                blocks += [res]
                continue

            res = np.matmul(a[i * a_per_row:(i + 1) * a_per_row, :], b[:, j * b_per_col:(j + 1) * b_per_col])
            blocks += [res]

    col_concat = []
    for k in range(n_fold):
        col_concat += [np.concatenate((blocks[k * n_fold:(k + 1) * n_fold]), axis=1)]
    total_matrix = np.concatenate(col_concat, axis=0)

    return total_matrix
