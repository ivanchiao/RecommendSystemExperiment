# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:05:46 2020

@author: LMC_ZC
"""


def early_stopping(evaluates, inval=3):
    if len(evaluates) <= inval:
        return False
    else:
        for i in range(len(evaluates) - inval - 1, len(evaluates) - 1):
            if evaluates[i] < evaluates[i + 1]:
                return False
        return True
