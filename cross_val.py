import numpy as np
from tree import tree_learn, parse_tree
from evaluation import evaluate, predict
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

def hyperparamters_list():
    #Define the different potential sets of hyperparameters
    hyparameters = []
    depths = [2, 3]
    boundaries = [1, 2]
    for m in range(len(depths)):
        for n in range(len(boundaries)):
            hyparameters.append({"depth":depths[m], "boundary":boundaries[n]})
    return hyparameters

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
    tree_hyperparameters = {} # Trees trained with different hyperparamters

    #Collecting the list of hyperparameters
    hyperparameters = hyperparamters_list()

    # Keep index before shuffling
    # index = np.arange(df.shape[0])
    # np.hstack((df, index)) # adding an index before shuffling
    np.random.shuffle(df)

    test_scores = []
    best_trees = []
    # Splitting Test from Validation and Training
    for i in range(folds):
        print("-------------------- Test/Train separation "+ str(i)+" --------------------")
        test_set, train_and_validate = split(df, i * split_size, split_size)

        trained_trees = []
        f1_scores = []
        # Splitting Validation from Training
        variance = np.zeros((4,folds-1))
        for j in range(folds - 1):
            print("------- Train/Validate separation "+ str(j)+" -------")
            # Split Validation from Training
            validate, train = split(train_and_validate, j * split_size, split_size)
            # Train Tree on Training
            tree, depth = tree_learn(train, 0, tree = {}, max_depth=10)
            # Pruning
            uar, uap, f1, uac = evaluate(tree, validate, 4)  # for later use
            tree = parse_tree(tree, None, train, validate, f1, [])
            print("Pruning done")
            # Evaluate Tree on Validate
            uar, uap, f1, uac = evaluate(tree, validate, 4)  # for later use
            variance[0,j] = uar
            variance[1,j] = uap
            variance[2,j] = f1
            variance[3,j] = uac
            tree_dictionary["tree_" + str(j)] = {"tree" : tree, "UAR" : uar, "UAP" : uap, "F1" : f1, "UAC" : uac}
            trained_trees.append(tree)
            f1_scores.append(f1)

        # For all the sets of hyperparemeters dictionnary
        for hyp in range(len(hyperparameters)):
            trained_trees = []
            f1_scores = []
            # Splitting Validation from Training
            variance = np.zeros((4,folds-1))
            for j in range(folds - 1):
                print("------- Train/Validate separation "+ str(j)+" -------")
                # Split Validation from Training
                validate, train = split(train_and_validate, j * split_size, split_size)
                # Train Tree on Training
                print("Training Tree")
                tree, depth = tree_learn(
                                    train,
                                    0,
                                    tree = {},
                                    max_depth = hyperparameters[hyp]["depth"],
                                    reduction = hyperparameters[hyp]["boundary"])
                print("Training done")
                # Pruning
                print("Pruning tree")
                uar, uap, f1, uac = evaluate(tree, validate, 4)  # for later use

                tree = parse_tree(tree, None, train, validate, f1, [])
                print("Pruning done")
                # Evaluate Tree on Validate
                uar, uap, f1, uac = evaluate(tree, validate, 4)  # for later use
                variance[0,j] = uar
                variance[1,j] = uap
                variance[2,j] = f1
                variance[3,j] = uac
                tree_dictionary["tree_" + str(j)] = {"tree" : tree, "UAR" : uar, "UAP" : uap, "F1" : f1, "UAC" : uac}
                trained_trees.append(tree)
                f1_scores.append(f1)

            best_tree = trained_trees[np.argmax(f1_scores)]
            best_trees.append(best_tree)
            test_score = evaluate(best_tree, test_set, 4)
            test_scores.append(test_score)
            print("\n----------- Compute Variances -----------\n")
            # Compute Variance on UAR, UAP, F1 and UAC (in this order)
            #print(variance.shape)
            #variance = np.var(variance, axis = 1)
            #print(variance.shape)
            #print("UAR : %.5f   UAP : %.5f   F1 : %.5f   UAC : %.5f"%(variance[0], variance[1], variance[2], variance[3]))
            #print("\n\n")
            # Each structure stores n-1 trees and the variances computed
            tree_hyperparameters["hyperparamters_"+str(hyp)] = {"tree_struct" : tree_dictionary, "variances" : variance}
        tree_structures["fold_"+str(i)] = {"tree_given_fold" : tree_hyperparameters}

    return best_trees, test_scores
