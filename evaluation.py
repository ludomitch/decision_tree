import numpy as np
from tree import predict


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


def cross_validation(tree, data, folds, test_percentage):
    """method : cross_validation : we'll explain once it's finished
        tree : dictionary of dictionaries: this is the decision tree
        data : array : this is the complete dataset used
        folds : int : how many folds are used during cross validation
        test_percentage : float : the percentage of the dataset to be used for testing
        returns "we'll see"""
    df = data.copy()
    errorList = []
    split_size = int(test_percentage * df.shape[0])

    luap, luar, lf1, luac = [], [], [], []

    # index = np.arange(df.shape[0])
    # np.hstack((df, index)) # adding an index before shuffling
    np.random.shuffle(df)

    # Splitting Test from Validation and Training
    for i in range(folds):
        test_set, train_and_validate = split(df, i * split_size, split_size)
        # print("Test, Val")
        # print("split Position:"+str(i*split_size))
        uar = 0
        uap = 0
        uac = 0
        f1 = 0

        # Splitting Validation from Training
        for j in range(folds - 1):
            validate, train = split(train_and_validate, j * split_size, split_size)
            tuar, tuap, tf1, tuac = evaluate(tree, validate, 4)  # for later use
            uar += tuar
            uap += tuap
            f1 += tf1
            uac += tuac

            # Train tree on train
            # validate
            # print("Val, train")
            # print("split Position:"+str(j*split_size))
            # compute errors
        # errorList.append()
        luar.append(uar / j)
        luap.append(uap / j)
        lf1.append(f1 / j)
        luac.append(uac / j)
    # luar = np.mean(luar)
    # luap = np.mean(luap)
    # luac = np.mean(luac)
    # lf1 = np.mean(lf1)

    print("uar = ", luar, end="\n\n")
    print("uap = ", luap, end="\n\n")
    print("Classification Rate = ", luac, end="\n\n")
    print("f1 = ", lf1, end="\n\n")

    return np.mean(lf1) # return f1 fr the moment.


def evaluate(tree, validation_set, nb_labels):
    """evaluate : method : evaluates the accuracy of the predictions made by a tree
    tree : dictionnary of dictionnaries : this is the decision tree
    validation_set : array : dataset used for validation
    nb_labels : int : number of labels to classify
    returns uar, uap, f1 and the classification rate (see lecture notes)"""

    confusion_matrix = np.zeros((nb_labels, nb_labels))
    for i in range(validation_set.shape[0]):
        if predict(tree, validation_set[i, :]) == validation_set[i, -1]:
            confusion_matrix[
                int(validation_set[i, -1]) - 1, int(validation_set[i, -1]) - 1
            ] += 1

        elif predict(tree, validation_set[i, :]) != validation_set[i, -1]:
            confusion_matrix[
                int(validation_set[i, -1]) - 1,
                int(predict(tree, validation_set[i, :])) - 1,
            ] += 1

    recall_vect = np.zeros((1, nb_labels))
    prec_vect = np.zeros((1, nb_labels))
    classification_rate = np.zeros((1, nb_labels))

    for i in range(confusion_matrix.shape[0]):
        recall_vect[0, i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        prec_vect[0, i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        classification_rate[0, i] = np.trace(confusion_matrix) / (
            np.trace(confusion_matrix)
            + np.sum(confusion_matrix[:, i])
            + np.sum(confusion_matrix[i, :])
            - 2 * confusion_matrix[i, i]
        )

    uar = np.mean(recall_vect)
    uap = np.mean(prec_vect)
    f1 = 2 * (uar * uap) / (uar + uap)
    uaclassifcation_rate = np.mean(classification_rate)

    # print("Recall [Room 1 | Room 2 | Room 3 | Room 4]")
    # print(recall_vect, end= "uar ="+str(uar) +"\n\n")
    # print("Precision [Room 1 | Room 2 | Room 3 | Room 4]")
    # print(prec_vect, end= "uap ="+str(uap) +"\n\n")
    # print("Classification Rate [Room 1 | Room 2 | Room 3 | Room 4]")
    # print(classification_rate, end= "Average ="+str(uar) +"\n\n")

    # print(confusion_matrix)

    return uar, uap, f1, uaclassifcation_rate
