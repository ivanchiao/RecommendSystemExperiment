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

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN")
    parser.add_argument('--dataset_name', default='amazon-book', type=str)

    parser.add_argument('--data_path', default='C:/Users/LMC_ZC/Desktop/RS_Experiment/LightGCN/data', type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--emb_size', default=64, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--decay', default='1e-5', type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--node_dropout', default=0.0, type=float)
    parser.add_argument('--mess_dropout', default='[0.0,0.0,0.0]', type=str)
    parser.add_argument('--node_dropout_flag', default='True', type=str)
    parser.add_argument('--cores', default=6, type=int)

    return parser.parse_args()
