import numpy as np

#####
# method : split : takes an array, splits it in 2 uneven arrays.
# A : array : The array to be split
# pos : int : the position on which the split occurs
# N : int : the number of rows to take when splitting
# the upper matrix is of size N, the lower one is (size of A) - N
# returns upper, lower
#####
#
def split(A, pos, N):
    upper = A[pos:pos+N]
    lower = np.vstack((A[:pos],A[pos+N:]))

    return upper, lower

#####
# method : crossValidation : we'll explain once it's finished
# data : array : this is the complete dataset used
# folds : int : how many folds are used during cross validation
# testPercentage : float : the percentage of the dataset to be used for testing
# returns "we'll see"
#####

def crossValidation(data, folds, testPercentage):

    df = data.copy()
    errorList = []
    splitSize = int(testPercentage * df.shape[0])

    #index = np.arange(df.shape[0])
    #np.hstack((df, index)) # adding an index before shuffling
    np.random.shuffle(df)

    # Splitting Test from Validation and Training
    for i in range(folds):
        testSet, trainAndValidate = split(df, i*splitSize, splitSize)
        print("Test, Val")
        print("split Position:"+str(i*splitSize))

        # Splitting Validation from Training
        for j in range(folds - 1):
            validate, train = split(trainAndValidate, j*splitSize, splitSize)
            # Train tree on train
            # validate
            print("Val, train")
            print("split Position:"+str(j*splitSize))
            # compute errors
        #errorList.append()

    return

#####
# evaluate : method : evaluates the accuracy of the predictions made by a tree
# tree : dictionnary of dictionnaries : this is the decision tree
# validationSet : array : dataset used for validation
# returns the error/accuracy i guess
#####
def evaluate(tree, validationSet):

    confusionMatrix = {"T":0, "F":0}
    Rooms = {"room 1": confusionMatrix, "room 2":confusionMatrix, "room 3": confusionMatrix, "room 4": confusionMatrix}

    for i in range(validationSet.shape[0]):

        if (predict(tree, validationSet[i:]) == validationSet[i,-1]):
            Rooms["room "+str(validationSet[i,-1])]["T"] +=1

        elif (predict(tree, validationSet[i:]) != validationSet[i,-1]):
            Rooms["room "+str(predict(tree, validationSet[i:]))]["F"] +=1
