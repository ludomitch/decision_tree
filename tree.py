import copy

import numpy as np

from evaluation import evaluate


def entropy(labels):
    """method : entropy
    labels : type : what it is

    returns "what is the return of this function ?"""
    size = labels.size
    res = 0
    for i in np.unique(labels):
        pk = len(labels[labels == i]) / size
        res += pk * np.log2(pk)
    return -res


def remainder(s_left, s_right):
    """method : remainder
    s_left :
    s_right :
    returns res"""
    norm_const = lambda x, y: x.size / (x.size + y.size)
    res = norm_const(s_left, s_right) * entropy(s_left) + norm_const(
        s_right, s_left
    ) * entropy(s_right)
    return res


def info_gain(s_all, bound):
    """method : info_gain
    s_all :
    bound :
    returns"""
    sorted_arr = np.sort(s_all, kind="heapsort")
    s_left = sorted_arr[sorted_arr[:, 0] > bound][
        :, 1
    ]  # labels are solely required for entropy calculation... hence only those are passed
    s_right = sorted_arr[sorted_arr[:, 0] < bound][:, 1]
    return entropy(sorted_arr[:, 1]) - remainder(s_left, s_right)


def get_boundaries(arr,reduction):
    """method : get_boundaries
    arr :
    returns"""
    depth = 1

    unique = np.unique(arr.copy())
    x = unique[:-1]
    y = unique[1:]
    boundary = (x + y) / 2

    print('Boundary Cut {}'.format(reduction))

    return boundary[::reduction]


def find_split(data,reduction):
    """method : find_split
    data : array : this is the training set ?

    returns"""
    top_gain = 0
    split = {}
    for col in range(0, data.shape[1] - 1):  # don't include last col: label col
        for boundary in get_boundaries(data[:, col],reduction):
            temp_gain = info_gain(data[:, [col, -1]], boundary)
            if temp_gain > top_gain:
                top_gain = temp_gain
                split = {
                    "wifi_signal": col,
                    "dB": boundary,
                    "info_label": find_leaf_value(data)
                }  # unindent till out of first for loop...

    return split

def find_leaf_value(data:np.array):
    """Given a set of data, find most recurrent label"""
    unique, counts = np.unique(data[:, -1], return_counts=True)
    value = unique[np.argmax(counts)]
    return value

def split_data(arr, col, bound):
    """Sort and split data based on column and boundary."""
    arr = arr[arr[:, col].argsort()]  # sort array
    left = arr[arr[:, col] > bound]
    right = arr[arr[:, col] < bound]
    return left, right


def tree_learn(data, depth, tree, max_depth,reduction):
    """method : tree_learn
    data :
    depth :
    tree : dictionnary of dictionnaries : this is the decision tree
    returns the score"""
    if depth == max_depth:
        unique, counts = np.unique(data[:, -1], return_counts=True)
        tree = unique[np.argmax(counts)]
        return tree, depth
    if np.all(data[:, -1] == data[0, -1]):  # check if all labels are identical
        tree = data[0, -1]
        return tree, depth
    split = find_split(data,reduction)
    split["left"] = {}
    split["right"] = {}
    tree = split
    l_data, r_data = split_data(data, split["wifi_signal"], split["dB"])
    l_branch, l_depth = tree_learn(l_data, depth + 1, split["left"], max_depth, reduction)
    r_branch, r_depth = tree_learn(r_data, depth + 1, split["right"], max_depth, reduction)
    tree["left"] = l_branch
    tree["right"] = r_branch

    return tree, max(l_depth, r_depth)


def evaluate_prune(
    tree: dict, train: np.array, test: np.array, base_score: float, track: list
) -> dict:
    """Prune and evaluate whether we want to keep pruned tree or original tree."""
    original = copy.deepcopy(tree)  # original tree
    leaf_value = get_nested_value(tree, track)['info_label']
    set_nested_value(tree, track, leaf_value)
    # chop off branches and turn into leaf
    ### Prune score needs to be replaced by better error loss function ###
    prune_score = evaluate(tree, test)[2]  # currently just using f1 mean
    if prune_score > base_score:
        return tree  # pruned
    return original


def parse_tree(
    tree: dict,
    branch: dict,
    train: np.array,
    test: np.array,
    base_score: float,
    track: list,
) -> dict:
    """Recursively loop through tree until you get to bottom
    and then prune appropriately once you reach the bottom.

    tree: nested dictionary decision tree
    branch: smaller segments of the tree as we go down the levels
    train: training dataset
    test: test set on which we evaluate. Used test to not confuse with evaluate function
    base_score: the original score of decision tree before pruning
    track: Keeps track of list of keys as we go down tree
    """
    # Initialize branch to be full tree before we recursively enter the branches
    if branch is None:
        branch = tree
    if isinstance(branch, float):
        return tree, prune_count
    if (isinstance(branch["left"], float))|(isinstance(branch['right'], float)):
        # if pruning is worth it, the pruned tree becomes the base tree
        tree = evaluate_prune(tree, train, test, base_score, track)
        return tree
    for i in ["left", "right"]:
        init_track = copy.deepcopy(track)
        track.append(i)
        tree = parse_tree(tree, branch[i], train, test, base_score, track)
        track = copy.deepcopy(
            init_track
        )  # reinitialise track for right after done with left
    return tree


def get_nested_value(nested_dict: dict, key_list: list):
    for k in key_list:
        nested_dict = nested_dict[k]
    return nested_dict


def set_nested_value(nested_dict: dict, key_list: list, value: float):
    for key in key_list[:-1]:
        nested_dict = nested_dict.setdefault(key, {})
    nested_dict[key_list[-1]] = value


#####################
# This is how to run
#####################
# import tree as dt
# import evaluation as ev
# import numpy as np
# import copy
# data = np.loadtxt('noisy_dataset.txt')
# size = data.shape[0]
# train = data[:int(size*0.8), :]
# test = data[int(size*0.8):, :]
# tree, _ = dt.tree_learn(train, 0, {}, 5)
# # dt.predict(tree, data[0, :])
# base_tree = copy.deepcopy(tree)
# base_score = ev.evaluate(base_tree, copy.deepcopy(test))
# pruning_tree = copy.deepcopy(tree)
# new_tree = dt.parse_tree(pruning_tree, None, train, test, base_score, [])
# new_tree == tree
