# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:05:46 2020

@author: LMC_ZC
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BPRMF")
    parser.add_argument('--dataset_name', default='amazon-book', type=str)
    parser.add_argument('--data_path', default='C:/Users/LMC_ZC/Desktop/RS_Experiment/BPRMF/data', type=str)
    parser.add_argument('--batch_size', default=2046, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--n_factors', default=64, type=int)
    parser.add_argument('--decay', default=0.01, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--cores', default=6, type=int)

    return parser.parse_args()
