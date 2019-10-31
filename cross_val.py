import numpy as np
from tree import tree_learn, run_pruning
from evaluation import evaluate
import config as cfg


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
    """Define the different potential sets of hyperparameters"""
    hyparameters = []
    depths = [2, 3]
    boundaries = [1, 2]
    for m in range(len(depths)):
        for n in range(len(boundaries)):
            hyparameters.append({"depth": depths[m], "boundary": boundaries[n]})
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

    # Collecting the list of hyperparameters
    hyperparameters = hyperparamters_list()

    # Keep index before shuffling
    # index = np.arange(df.shape[0])
    # np.hstack((df, index)) # adding an index before shuffling
    np.random.shuffle(df)

    test_scores = []
    best_trees = []
    # Splitting Test from Validation and Training
    for i in range(folds):
        print(
            "-------------------- Test/Train separation "
            + str(i)
            + " --------------------"
        )
        test_set, train_and_validate = split(df, i * split_size, split_size)

        # Splitting Validation from Training
        variance = np.zeros((4, folds - 1))
        # For all the sets of hyperparemeters dictionnary
        for hyp in hyperparameters:
            print(f"Running with hyperparameters: {hyp}")
            trained_trees = []
            moi = []  # metric of interest
            # Splitting Validation from Training
            variance = np.zeros((4, folds - 1))
            for j in range(folds - 1):
                print("------- Train/Validate separation " + str(j) + " -------")
                # Split Validation from Training
                validate, train = split(train_and_validate, j * split_size, split_size)
                # Train Tree on Training
                tree, _ = tree_learn(
                    train, 0, tree={}, max_depth=hyp["depth"], reduction=hyp["boundary"]
                )
                # Pruning
                metric_scores = evaluate(tree, validate)  # for later use
                tree = run_pruning(
                    tree, train, validate, metric_scores[cfg.METRIC_CHOICE]
                )

                # variance[0, j] = uar
                # variance[1, j] = uap
                # variance[2, j] = f1
                # variance[3, j] = uac
                # variance = np.var(variance, axis = 1)

                trained_trees.append(tree)
                moi.append(metric_scores[cfg.METRIC_CHOICE])

            best_tree = trained_trees[np.argmax(moi)]
            best_trees.append(best_tree)
            test_score = evaluate(best_tree, test_set)
            test_scores.append(test_score)

    return best_trees, test_scores
