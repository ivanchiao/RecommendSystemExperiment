# -*- coding: utf-8 -*-
"""

@author: LMC_ZC
"""

import os
import pickle
import numpy as np


def read_raw(path):
    with open(path) as f:
        User, Item, User2Item = [], [], {}
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                uid = int(l[0])
                try:
                    items = [int(i) for i in l[1:]]
                except BaseException as e:
                    print(uid, e)
                    continue
                else:
                    User2Item[uid] = items
                    User.extend([uid] * len(items))
                    Item.extend(items)
    return np.array(User), np.array(Item), User2Item


if __name__ == '__main__':
    root = 'raw'
    #dataset_name = 'gowalla'
    dataset_name = 'amazon-book'
    
    train_path = os.path.join(dataset_name, root, 'train.txt')
    test_path = os.path.join(dataset_name, root, 'test.txt')
    train_U, train_I, train_U2I = read_raw(train_path)
    test_U, test_I, test_U2I = read_raw(test_path)
    num_user = len(np.unique(np.concatenate([train_U, test_U])))
    num_item = len(np.unique(np.concatenate([train_I, test_I])))

    train_set = np.vstack([train_U, train_I]).T.tolist()

    with open(dataset_name + '/processed/preprocess.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_U2I, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_U2I, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump([num_user, num_item], f, pickle.HIGHEST_PROTOCOL)
