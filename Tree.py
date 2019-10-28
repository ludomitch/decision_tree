import numpy as np

def entropy(labels):
    """method : entropy
    labels : type : what it is

    returns "what is the return of this function ?"""
    size = labels.size
    res = 0
    for i in np.unique(labels):
        pk = len(labels[labels==i])/size
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
    sorted_arr = np.sort(s_all,kind = 'heapsort')
    s_left = sorted_arr[sorted_arr[:,0]>bound][:,1] # labels are solely required for entropy calculation... hence only those are passed
    s_right = sorted_arr[sorted_arr[:,0]<bound][:,1]
    return entropy(sorted_arr[:, 1]) - remainder(s_left, s_right)

def get_boundaries(arr):
    """method : get_boundaries
    arr :

    returns"""
    unique = np.unique(arr.copy())
    x = unique[:-1]
    y = unique[1:]
    return (x + y) / 2


def find_split(data):
    """method : find_split
    data : array : this is the training set ?

    returns"""
    top_gain = 0
    split = {}
    for col in range(0, data.shape[1] - 1):  # don't include last col: label col
        for boundary in get_boundaries(data[:, col]):
            temp_gain = info_gain(data[:, [col, -1]], boundary)
            if temp_gain > top_gain:
                top_gain = temp_gain
                split = {'attribute':col, 'value':boundary} # unindent till out of first for loop...

    return split


def split_data(arr, col, bound):
    """method : split_data
    arr :
    col :
    bound :
    returns"""
    sorted_arr = arr[arr[:, col].argsort()]
    left = arr[arr[:, col] > bound]
    right = arr[arr[:, col] < bound]
    return left, right


def tree_learn(data, depth, tree):
    """method : tree_learn
    data :
    depth :
    tree : dictionnary of dictionnaries : this is the decision tree
    returns the score"""
    max_depth = 2
    if depth == max_depth:
        unique, counts = np.unique(data[:, -1], return_counts=True)
        tree = unique[np.argmax(counts)]
        return tree, depth
    if np.all(data[:, -1] == data[0, -1]):  # check if all labels are identical
        tree = data[0, -1]
        return tree, depth
    split = find_split(data)
    split["left"] = {}
    split["right"] = {}
    tree = split
    l_data, r_data = split_data(data, split["attribute"], split["value"])
    l_branch, l_depth = tree_learn(l_data, depth + 1, split["left"])
    r_branch, r_depth = tree_learn(r_data, depth + 1, split["right"])
    tree["left"] = l_branch
    tree["right"] = r_branch

    return tree, max(l_depth, r_depth)


def predict(tree, data):
    """method : predict
    tree : dictionnary of dictionnaries : this is the decision tree
    data : numpy array of floats : this is the data which is used to predict an outcome
    returns the predicted label"""
    if isinstance(tree, float):
        return tree
    if data[tree['attribute']]>tree['value']:
        return predict(tree['left'], data)
    else:
        return predict(tree['right'], data)
    return tree

# data = np.loadtxt('noisy_dataset.txt')
# tdict = tree_learn(data,0,0)
# pprint.pprint(tdict[0])
# predict(tdict[0],data[2,:-1]) # ROW, Class
