import numpy as np
from tree import tree_learn, run_pruning
from evaluation import evaluate
import config as cfg


def split(arr, pos, n):
    """takes an array, splits it in 2 uneven arrays.
    arr : array : The array to be split
    pos : int : the position on which the split occurs
    n : int : the number of rows to take when splitting
    the upper matrix is of size n, the lower one is (size of arr) - N
    returns upper, lower"""

    upper = arr[pos : pos + n]
    lower = np.vstack((arr[:pos], arr[pos + n :]))

    return upper, lower


def hyperparamters_list() -> list:
    """Define the different potential combinations of hyperparameters"""
    hyparameters = []
    depths = cfg.DEPTH
    boundaries = cfg.BOUNDARIES
    pruning = cfg.PRUNING
    for m in range(len(depths)):
        for n in range(len(boundaries)):
            for k in range(len(pruning)):
                hyparameters.append(
                    {"depth": depths[m], "boundary": boundaries[n], "prune": pruning[k]}
                )
    return hyparameters


def cross_validation(data: np.array, folds: int, test_percentage: float) -> tuple:
    """Conduct cross validation
        data : array : this is the complete dataset used
        folds : int : how many folds are used during cross validation
        test_percentage : float : the percentage of the dataset to be used for testing
        returns "we'll see"""

    # copy data to avoid damaging the dataset
    df = data.copy()
    split_size = int(test_percentage * df.shape[0])

    # Collecting the list of hyperparameters
    hyperparameters = hyperparamters_list()

    # Shuffle data
    np.random.shuffle(df)

    # Initialise score trackers
    global_best_hyperparams = []
    global_scores = []

    # Splitting Test from Validation and Training
    for i in range(folds):
        print(
            "-------------------- Test/Train separation "
            + str(i)
            + " --------------------"
        )
        test_set, train_and_validate = split(df, i * split_size, split_size)
        # Splitting Validation from Training
        err_all_hyperparams = []
        # For all the sets of hyperparemeters dictionnary
        for hyp in hyperparameters:
            print(f"Running with hyperparameters: {hyp}")
            moi = []  # metric of interest: i.e. F1
            # Splitting Validation from Training
            for j in range(folds - 1):
                print("------- Train/Validate separation " + str(j) + " -------")
                # Split Validation from Training
                validate, train = split(train_and_validate, j * split_size, split_size)

                # Train Tree on Training
                tree, _ = tree_learn(
                    train, 0, tree={}, max_depth=hyp["depth"], reduction=hyp["boundary"]
                )
                # Pruning
                if hyp["prune"]:
                    base_scores = evaluate(tree, validate)  # for later use
                    tree, metric_scores = run_pruning(
                        tree, train, validate, base_scores[cfg.METRIC_CHOICE]
                    )
                else:
                    metric_scores = evaluate(tree, validate)[cfg.METRIC_CHOICE]

                moi.append(
                    metric_scores
                )  # metric_scores is F1 only currently as se tin run_pruning output

            # MAKE ERROR ESTIAMTE OF ALL folds for a given hyperparam
            err_all_hyperparams.append(np.mean(moi))

        # APPEND ERROR ESTIMATE, HYPERPARMS
        best_hyper = hyperparameters[np.argmax(err_all_hyperparams)]
        global_best_hyperparams.append(best_hyper)

        # FINAL evaluate with test data
        tree, _ = tree_learn(
            train_and_validate,
            0,
            tree={},
            max_depth=best_hyper["depth"],
            reduction=best_hyper["boundary"],
        )

        if best_hyper["prune"]:
            base_scores = evaluate(tree, test_set)
            tree, _ = run_pruning(
                tree, train_and_validate, test_set, base_scores[cfg.METRIC_CHOICE]
            )

        score_hyper = evaluate(tree, test_set)  # [cfg.METRIC_CHOICE]

        global_scores.append(score_hyper)

    # BEST HYPERPARAMETERS FOR ALL TEST CASES
    base_dict = {}
    for key in global_scores[0].keys():
        base_dict[key] = 0
        for i in range(np.size(global_scores)):
            base_dict[key] += global_scores[i][key]
        base_dict[key] = base_dict[key] / (i + 1)

    return global_best_hyperparams, global_scores, base_dict


def param_tuning(data: np.array, folds: int, test_percentage: float) -> tuple:
    """Run parameter tuning."""
    df = data.copy()
    split_size = int(test_percentage * df.shape[0])

    hyperparameters = hyperparamters_list()
    np.random.shuffle(df)

    err_all_hyperparams = []
    err_all_hyperparams_var = []

    # For all the sets of hyperparemeters dictionnary
    for hyp in hyperparameters:
        print(f"Running with hyperparameters: {hyp}")
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
            base_scores = evaluate(tree, eval_set)  # for later use
            if hyp['prune']:
                tree, _ = run_pruning(
                    tree, train_set, eval_set, base_scores[cfg.METRIC_CHOICE]
                )
            metric_scores = evaluate(tree, eval_set) # [cfg.METRIC_CHOICE]

            moi.append(
                metric_scores
            )  # metric_scores is F1 only currently as se tin run_pruning output

        base_dict = {}

        arr_scores = np.zeros((4,np.size(moi)))
        for index, key in enumerate("uar uap f1 uac".split()):
            base_dict[key] = 0
            for i in range(np.size(moi)):
                arr_scores[index,i] = moi[i][key]
        variances = np.var(arr_scores,axis = 1)
        means = np.mean(arr_scores,axis = 1)

        #for key in "uar uap f1 uac".split():
        err_all_hyperparams.append(means)
        err_all_hyperparams_var.append(variances)

        err = np.array(err_all_hyperparams)
        var = np.array(err_all_hyperparams_var)

        best_hyper = cv.hyperparamters_list()[np.argmax(err[:,2])]

    return best_hyper, err, var
