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
    depths = cfg.DEPTH
    boundaries = cfg.BOUNDARIES
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
    global_best_hyperparams = []
    global_F1 = []
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
        err_all_hyperparams = []
        # For all the sets of hyperparemeters dictionnary
        for hyp in hyperparameters:
            print(f"Running with hyperparameters: {hyp}")
            trained_trees = []
            moi = []  # metric of interest: i.e. F1
            # Splitting Validation from Training
            variance = np.zeros((4, folds - 1))
            for j in range(folds - 1):
                print("------- Train/Validate separation " + str(j) + " -------")
                # Split Validation from Training
                validate, train = split(train_and_validate, j * split_size, split_size)
                # print("---Test {}, Training {}, Eval {}".format(test_set.shape,train.shape,validate.shape))
                # Train Tree on Training
                tree, _ = tree_learn(
                    train, 0, tree={}, max_depth=hyp["depth"], reduction=hyp["boundary"]
                )
                # Pruning
                base_scores = evaluate(tree,validate)  # for later use
                tree, metric_scores = run_pruning(
                    tree, train, validate, base_scores[cfg.METRIC_CHOICE]
                )
                # variance[0, j] = uar
                # variance[1, j] = uap
                # variance[2, j] = f1
                # variance[3, j] = uac
                # variance = np.var(variance, axis = 1)

                moi.append(metric_scores) #metric_scores is F1 only currently as se tin run_pruning output

            # MAKE ERROR ESTIAMTE OF ALL folds for a given hyperparam
            err_all_hyperparams.append(np.mean(moi))

        # APPEND ERROR ESTIMATE, HYPERPARMS
        best_hyper = hyperparameters[np.argmax(err_all_hyperparams)]
        global_best_hyperparams.append(best_hyper)

        #FINAL evaluate with test data
        tree,_ = tree_learn(train_and_validate, 0, tree={}, max_depth=best_hyper["depth"], reduction=best_hyper["boundary"])
        base_scores = evaluate(tree, test_set)
        tree, F1_score = run_pruning(tree, train_and_validate, test_set, base_scores[cfg.METRIC_CHOICE])
        F1_hyper = evaluate(tree, test_set)[cfg.METRIC_CHOICE]
        global_F1.append(F1_hyper)

    # BEST HYPERPARAMETERS FOR ALL TEST CASES
    average_F1 = np.mean(global_F1)
        # best_tree = trained_trees[np.argmax(moi)]
        # test_score = evaluate(best_tree, test_set)
        # best_trees.append(best_tree)
        #test_scores.append(test_score)

    return global_best_hyperparams, global_F1, average_F1

def param_tuning(data, folds,test_percentage):

    global_best_hyperparams = []
    global_F1 = []
    df = data.copy()
    split_size = int(test_percentage * df.shape[0])

    hyperparameters = hyperparamters_list()
    np.random.shuffle(df)

    err_all_hyperparams = []

    # For all the sets of hyperparemeters dictionnary
    for hyp in hyperparameters:
        print(f"Running with hyperparameters: {hyp}")
        trained_trees = []
        moi = []  # metric of interest: i.e. F1

        for i in range(folds):
            print(
                "-------------------- Eval/Train separation "
                + str(i)
                + " --------------------"
            )
            # SPLIT DATAFRAME
            eval_set, train_set = split(df, i * split_size, split_size)

            # Train Tree on Training
            tree, _ = tree_learn(
                train_set, 0, tree={}, max_depth=hyp["depth"], reduction=hyp["boundary"]
            )
            # Pruning
            base_scores = evaluate(tree,eval_set)  # for later use
            tree, metric_scores = run_pruning(
                tree, train_set, eval_set, base_scores[cfg.METRIC_CHOICE]
            )

            moi.append(metric_scores) #metric_scores is F1 only currently as se tin run_pruning output

        err_all_hyperparams.append(np.mean(moi))

    # APPEND ERROR ESTIMATE, HYPERPARMS
    best_hyper = hyperparameters[np.argmax(err_all_hyperparams)]
    F1_score = np.max(err_all_hyperparams)

    return best_hyper, F1_score, err_all_hyperparams
