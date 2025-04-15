import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.loadtxt(train_path)
    valid_list = np.loadtxt(valid_path)
    test_list = np.loadtxt(test_path)

    uid_max = 0
    iid_max = 0
    train_dict = {}
    tuid_max = 0
    tiid_max = 0
    test_dict = {}

    for uid, iid,rate in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = int(uid_max + 1)
    n_item = int(iid_max + 1)
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    # for tuid, tiid, trate in test_list:
    #     if tuid not in test_dict:
    #         test_dict[tuid] = []
    #     test_dict[tuid].append(tiid)
    #     if tuid > tuid_max:
    #         tuid_max = tuid
    #     if tiid > tiid_max:
    #         tiid_max = tiid
    #
    # tn_user = tuid_max + 1
    # tn_item = tiid_max + 1
    # print(f'user num: {tn_user}')
    # print(f'item num: {tn_item}')

    train_data = sp.csr_matrix((train_list[:, 2], \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((valid_list[:, 2],
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((test_list[:,2],
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
