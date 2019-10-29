import numpy as np
from tree import predict, tree_learn
from evaluation import evaluate
def split(arr, pos, n):
    """method : split : takes an array, splits it in 2 uneven arrays.
    arr : array : The array to be split
    pos : int : the position on which the split occurs
    n : int : the number of rows to take when splitting
    the upper matrix is of size n, the lower one is (size of arr) - N
    returns upper, lower"""

    upper = arr[pos : pos + n]
    lower = np.vstack((arr[:pos], arr[pos + n :]))

    return upper, lower


def cross_validation(data, folds, test_percentage):
    """method : cross_validation : we'll explain once it's finished
        data : array : this is the complete dataset used
        folds : int : how many folds are used during cross validation
        test_percentage : float : the percentage of the dataset to be used for testing
        returns "we'll see"""
    
    # copy data to avoid damaging the dataset
    df = data.copy()
    split_size = int(test_percentage * df.shape[0])
    
    tree_structures = {} # Trees trained on different test sets
    tree_dictionary = {} # Trees trained on the same Validation + Training

    # Keep index before shuffling
    # index = np.arange(df.shape[0])
    # np.hstack((df, index)) # adding an index before shuffling
    np.random.shuffle(df)

    # Splitting Test from Validation and Training
    for i in range(folds):
        print("-------------------- Test/Train separation "+ str(i)+" --------------------")
        test_set, train_and_validate = split(df, i * split_size, split_size)
        
        # Splitting Validation from Training
        variance = np.zeros((4,folds-1))
        for j in range(folds - 1):
            print("------- Train/Validate separation "+ str(j)+" -------")
            # Split Validation from Training
            validate, train = split(train_and_validate, j * split_size, split_size)
            # Train Tree on Training
            print("Training Tree")
            tree, depth = tree_learn(train, depth = 3, tree = {})
            print("Pruning not done yet\n")
            # Pruning
            
            # Evaluate Tree on Validate
            uar, uap, f1, uac = evaluate(tree, validate, 4)  # for later use
            variance[0,j] = uar
            variance[1,j] = uap
            variance[2,j] = f1
            variance[3,j] = uac
            tree_dictionary["tree_" + str(j)] = {"tree" : tree, "UAR" : uar, "UAP" : uap, "F1" : f1, "UAC" : uac}

        print("\n----------- Compute Variances -----------\n")
        # Compute Variance on UAR, UAP, F1 and UAC (in this order)
        print(variance.shape)
        variance = np.var(variance, axis = 1)
        print(variance.shape)
        print("UAR : %.5f   UAP : %.5f   F1 : %.5f   UAC : %.5f"%(variance[0], variance[1], variance[2], variance[3]))
        print("\n\n")
        # Each structure stores n-1 trees and the variances computed
        tree_structures["structure_"+str(i)] = {"tree_struct" : tree_dictionary, "variances" : variance}

    return tree_structures
