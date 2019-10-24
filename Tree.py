import numpy as np
import math

#####
# method : entropy
# labels : type : what it is
# 
# returns "what is the return of this function ?
##"###
def entropy(labels):
    size = labels.size
    res = 0
    for i in np.unique(labels):
        pk = len(labels[labels==i])/size
        res += pk * math.log(pk, 2)
    return -res

#####
# method : remainder
# s_left : 
# s_right :
# returns res
#####
def remainder(s_left, s_right):
    norm_const = lambda x,y: x.size/(x.size+y.size)
    res = norm_const(s_left, s_right)*entropy(s_left)+norm_const(s_right, s_left)*entropy(s_right)
    return res

#####
# method : info_gain
# s_all : 
# bound :
# returns 
#####
def info_gain(s_all, bound):
    sorted_arr = s_all[s_all[:,0].argsort()]
    s_left = sorted_arr[sorted_arr[:,0]>bound][:,1]
    s_right = sorted_arr[sorted_arr[:,0]<bound][:,1]
    sorted_arr = sorted_arr[:, 1]
    return entropy(sorted_arr) - remainder(s_left, s_right)

#####
# method : get_boundaries
# arr : 
#
# returns 
#####
def get_boundaries(arr):
    unique = np.unique(arr.copy())
    x = unique[:-1]
    y = unique[1:]
    return (x+y)/2

#####
# method : find_split
# data : array : this is the training set ?
#
# returns 
#####
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

#####
# method : split_data
# arr :
# col :
# bound :
# returns 
#####
def split_data(arr, col, bound):
    sorted_arr = arr[arr[:,col].argsort()]
    left = arr[arr[:,col]>bound]
    right = arr[arr[:,col]<bound]
    return left, right

#####
# method : tree_learn
# data : 
# depth :
# tree : dictionnary of dictionnaries : this is the decision tree
# returns the score
#####
def tree_learn(data, depth, tree):
    max_depth = 2
    if depth==max_depth:
        unique, counts = np.unique(data[:,-1], return_counts=True)
        tree = unique[np.argmax(counts)]
        return tree, depth
    if np.all(data[:, -1]==data[0, -1]): # check if all labels are identical
        tree = data[0,-1]
        return tree, depth, 
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

#####
# method : score_tree
# tree : dictionnary of dictionnaries : this is the decision tree
# score : Que esta ?
# returns the score
#####
def score_tree(tree):
    score = 1
    return score

#####
# method : predict
# tree : dictionnary of dictionnaries : this is the decision tree
# data : numpy array of floats : this is the data which is used to predict an outcome
# returns the predicted label
#####
def predict(tree, data):
    if isinstance(tree, float):
        return tree
    if data[tree['attribute']]>tree['value']:
        return predict(tree['left'], data)
    else:
        return predict(tree['right'], data)
    return tree