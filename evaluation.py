import numpy as np
from tree import predict, tree_learn

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


def evaluate(tree, validation_set, nb_labels):
    """evaluate : method : evaluates the accuracy of the predictions made by a tree
    tree : dictionnary of dictionnaries : this is the decision tree
    validation_set : array : dataset used for validation
    nb_labels : int : number of labels to classify
    returns uar, uap, f1 and the classification rate (see lecture notes)"""

    # Create the confusion matrix
    confusion_matrix = np.zeros((nb_labels, nb_labels))
    
    for i in range(validation_set.shape[0]): # For each line from the validation set
        
        if predict(tree, validation_set[i, :]) == validation_set[i, -1]: # If the tree predicted the label correctly
            confusion_matrix[
                int(validation_set[i, -1]) - 1, int(validation_set[i, -1]) - 1
            ] += 1 # At position (correct label, correct label) we add 1

        elif predict(tree, validation_set[i, :]) != validation_set[i, -1]: # If the tree predicted wrongly
            confusion_matrix[
                int(validation_set[i, -1]) - 1,
                int(predict(tree, validation_set[i, :])) - 1,
            ] += 1 # At position (correct label, predicted label) we add 1

    # Initializing Recall, Precision and Classification rate
    recall_vect = np.zeros((1, nb_labels))
    prec_vect = np.zeros((1, nb_labels))
    classification_rate = np.zeros((1, nb_labels))

    for i in range(confusion_matrix.shape[0]):
        recall_vect[0, i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) # Recall = TP/(TP + FN)
        prec_vect[0, i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]) # Precision = TP/(TP + FP)
        classification_rate[0, i] = np.trace(confusion_matrix) / (
            np.trace(confusion_matrix)
            + np.sum(confusion_matrix[:, i])
            + np.sum(confusion_matrix[i, :])
            - 2 * confusion_matrix[i, i]
        ) # Classification rate = (TP + TN)/(TP + TN + FP + FN)

    uar = np.mean(recall_vect) # Unweighted Average Recall
    uap = np.mean(prec_vect) # Unweighted Average Precision
    f1 = 2 * (uar * uap) / (uar + uap) # Compute F1
    uaclassifcation_rate = np.mean(classification_rate) # Unweighted Averate Classification Rate

    return uar, uap, f1, uaclassifcation_rate
