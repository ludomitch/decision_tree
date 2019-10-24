import pandas as pd
import numpy as np
data = np.loadtxt('noisy_dataset.txt')

import math

def entropy(labels):
    size = labels.size
    res = 0
    for i in np.unique(labels):
        pk = len(labels[labels==i])/size
        res += pk * math.log(pk, 2)
    return -res

def remainder(s_left, s_right):
    norm_const = lambda x,y: x.size/(x.size+y.size)
    res = norm_const(s_left, s_right)*entropy(s_left)+norm_const(s_right, s_left)*entropy(s_right)
    return res

def info_gain(s_all, bound):
    sorted_arr = s_all[s_all[:,0].argsort()]
    s_left = sorted_arr[sorted_arr[:,0]>bound][:,1]
    s_right = sorted_arr[sorted_arr[:,0]<bound][:,1]
    sorted_arr = sorted_arr[:, 1]
    return entropy(sorted_arr) - remainder(s_left, s_right)

def get_boundaries(arr):
    unique = np.unique(arr.copy())
    x = unique[:-1]
    y = unique[1:]
    return (x+y)/2

def find_split(data):
    top_gain = 0
    split = {}
    for col in range(0, data.shape[1]-1): #don't include last col: label col
        for boundary in get_boundaries(data[:, col]):
            temp_gain = info_gain(data[:, [col,-1]], boundary)
            if temp_gain > top_gain:
                top_gain = temp_gain
                split = {'attribute':col, 'value':boundary}
    return split

def split_data(arr, col, bound):
    sorted_arr = arr[arr[:,col].argsort()]
    left = arr[arr[:,col]>bound]
    right = arr[arr[:,col]<bound]
    return left, right

def tree_learn(data, depth, tree):
    max_depth = 4
    if depth==max_depth:
        return tree, depth
    if np.all(data[:, -1]==data[0, -1]): # check if all labels are identical
        return tree, depth
    split = find_split(data)
    split['left'] = {}
    split['right'] = {}
    tree = split
    l_data, r_data = split_data(data, split['attribute'], split['value'])
    l_branch, l_depth = tree_learn(l_data, depth+1, split['left'])
    r_branch, r_depth = tree_learn(r_data, depth+1, split['right'])
    tree['left'] = l_branch
    tree['right'] = r_branch

    return tree, max(l_depth, r_depth)

res = tree_learn(data, 0, {})
