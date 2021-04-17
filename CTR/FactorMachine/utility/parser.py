# -*- coding: utf-8 -*-
"""
@author: LMC_ZC
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Factor Machine")
    parser.add_argument('--dataset_name', default='movielens-1m', type=str)
    parser.add_argument('--data_path', default='/zhaochen/RS_FIRST_EXPERIMENT/CTR/FactorMachine/data/', type=str)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--n_factors', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=300, type=int)
    parser.add_argument('--threshold', default=4, type=int)
    parser.add_argument('--cores', default=4, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)

    return parser.parse_args()