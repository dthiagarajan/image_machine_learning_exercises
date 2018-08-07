import os.path

import random
import math
import time
import sys
import string
import shlex, subprocess

import numpy as np

common_concat = lambda x: np.concatenate(x, axis=0)


# get peak data
def obtain_data_from_list(the_data, data_list):
    obtained_data = []
    for i in range(0, len(data_list)):
        obtained_data.append(the_data[data_list[i]])

    return np.array(obtained_data)


def get_xy(x, y, l):
    return obtain_data_from_list(x, l), obtain_data_from_list(y, l)


# get random batch
def get_batch(x_data, y_data, s):
    lowest = 0
    highest = len(x_data)

    rand_list = np.random.randint(low=lowest, high=highest, size=s)

    return x_data[rand_list], y_data[rand_list]


# get balanced batch
def get_balanced_batch(x_p, y_p, x_n, y_n, s, p_rate):
    # calc each batch size
    p_batch_size = int(s * p_rate)
    n_batch_size = int(s - p_batch_size)

    x_batch_p, y_batch_p = get_batch(x_p, y_p, p_batch_size)
    x_batch_n, y_batch_n = get_batch(x_n, y_n, n_batch_size)

    return common_concat([x_batch_p, x_batch_n]), common_concat([y_batch_p, y_batch_n])


# make train and test list
def get_load_list(ll, train_num, total_num):
    print("Whether to use load_list:")
    print(ll)

    if ll:
        train_list = np.loadtxt("train_list", dtype=np.int32)
        test_list = np.loadtxt("test_list", dtype=np.int32)
    else:
        # separate data
        index_list = list(range(0, total_num))
        random.shuffle(index_list)

        train_list = index_list[:train_num]
        test_list = index_list[train_num:]

        np.savetxt("train_list", train_list)
        np.savetxt("test_list", test_list)

    return train_list, test_list


# make train, test and val list
def get_load_list_ttv(ll, train_num, test_num, total_num):
    print("Whether to use load_list:")
    print(ll)

    if ll:
        train_list = np.loadtxt("train_list", dtype=np.int32)
        test_list = np.loadtxt("test_list", dtype=np.int32)
        val_list = np.loadtxt("val_list", dtype=np.int32)
    else:
        # separate data
        index_list = list(range(0, total_num))
        random.shuffle(index_list)

        train_list = index_list[:train_num]
        test_list = index_list[train_num:train_num+test_num]
        val_list = index_list[train_num+test_num:]

        np.savetxt("train_list", train_list)
        np.savetxt("test_list", test_list)
        np.savetxt("val_list", val_list)

    return train_list, test_list, val_list


# return the list sizes for train and total
def list_size_calc(p_data, n_data, train_rate):
    total_p = len(p_data)
    total_n = len(n_data)
    train_p = int(total_p * train_rate)
    train_n = int(total_n * train_rate)

    return total_p, train_p, total_n, train_n


# make train and test list
def get_balanced_load_list(ll, train_num_p, total_num_p, train_num_n, total_num_n):
    print("Whether to use load_list:")
    print(ll)

    if ll:
        train_list_p = np.loadtxt("train_list_p", dtype=np.int32)
        test_list_p = np.loadtxt("test_list_p", dtype=np.int32)
        train_list_n = np.loadtxt("train_list_n", dtype=np.int32)
        test_list_n = np.loadtxt("test_list_n", dtype=np.int32)
    else:
        # separate data
        index_list_p = list(range(0, total_num_p))
        random.shuffle(index_list_p)
        index_list_n = list(range(0, total_num_n))
        random.shuffle(index_list_n)

        train_list_p = index_list_p[:train_num_p]
        test_list_p = index_list_p[train_num_p:]
        train_list_n = index_list_n[:train_num_n]
        test_list_n = index_list_n[train_num_n:]

        np.savetxt("train_list_p", train_list_p)
        np.savetxt("test_list_p", test_list_p)
        np.savetxt("train_list_n", train_list_n)
        np.savetxt("test_list_n", test_list_n)

    return train_list_p, test_list_p, train_list_n, test_list_n
