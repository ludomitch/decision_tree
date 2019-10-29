import numpy as np


def predict(tree, data):
    """method : predict
    tree : dictionnary of dictionnaries : this is the decision tree
    data : numpy array of floats : this is the data which is used to predict an outcome
    returns the predicted label"""
    if isinstance(tree, float):
        return tree
    if data[tree["attribute"]] > tree["value"]:
        return predict(tree["left"], data)
    else:
        return predict(tree["right"], data)
    return tree

def evaluate(tree, validation_set, nb_labels=4):
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