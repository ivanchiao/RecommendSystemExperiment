# -*- coding: utf-8 -*-
"""

@author : LMC_ZC
"""

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