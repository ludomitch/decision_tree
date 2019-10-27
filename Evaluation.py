import numpy as np

#####
# method : split : takes an array, splits it in 2 uneven arrays.
# A : array : The array to be split
# pos : int : the position on which the split occurs
# N : int : the number of rows to take when splitting
# the upper matrix is of size N, the lower one is (size of A) - N
# returns upper, lower
#####
def split(A, pos, N):
    upper = A[pos:pos+N]
    lower = np.vstack((A[:pos],A[pos+N:]))
    
    return upper, lower

#####
# method : crossValidation : we'll explain once it's finished
# tree : dictionary of dictionaries: this is the decision tree
# data : array : this is the complete dataset used
# folds : int : how many folds are used during cross validation
# testPercentage : float : the percentage of the dataset to be used for testing
# returns "we'll see"
#####
def crossValidation(tree, data, folds, testPercentage):
    
    df = data.copy()
    errorList = []
    splitSize = int(testPercentage * df.shape[0])
    
    LUAP, LUAR, LF1, LUAC = [], [], [], []
    
    #index = np.arange(df.shape[0])
    #np.hstack((df, index)) # adding an index before shuffling
    np.random.shuffle(df)
    
    # Splitting Test from Validation and Training
    for i in range(folds):
        testSet, trainAndValidate = split(df, i*splitSize, splitSize)
        #print("Test, Val")
        #print("split Position:"+str(i*splitSize))
        UAR = 0
        UAP = 0
        UAC = 0
        F1 = 0
        
        # Splitting Validation from Training
        for j in range(folds - 1):
            validate, train = split(trainAndValidate, j*splitSize, splitSize)
            TUAR, TUAP, TF1, TUAC = evaluate(tree, validate, 4) # for later use
            UAR += TUAR
            UAP += TUAP
            F1 += TF1
            UAC += TUAC
            
            # Train tree on train
            # validate
            #print("Val, train")
            #print("split Position:"+str(j*splitSize))
            # compute errors
        #errorList.append()
        LUAR.append(UAR/j)
        LUAP.append(UAP/j)
        LF1.append(F1/j)
        LUAC.append(UAC/j)
    #LUAR = np.mean(LUAR)
    #LUAP = np.mean(LUAP)
    #LUAC = np.mean(LUAC)
    #LF1 = np.mean(LF1)
    
    print("UAR = ", LUAR ,end ="\n\n")
    print("UAP = ", LUAP ,end ="\n\n")
    print("Classification Rate = ", LUAC ,end ="\n\n")
    print("F1 = ", LF1 ,end ="\n\n")
    
    return

#####
# evaluate : method : evaluates the accuracy of the predictions made by a tree
# tree : dictionnary of dictionnaries : this is the decision tree
# validationSet : array : dataset used for validation
# nbLabels : int : number of labels to classify
# returns UAR, UAP, F1 and the classification rate (see lecture notes)
#####
def evaluate(tree, validationSet, nbLabels):
    from Tree import predict
    
    ConfusedMatrix = np.zeros((nbLabels,nbLabels))
    for i in range(validationSet.shape[0]):
        if (predict(tree, validationSet[i,:]) == validationSet[i,-1]):
            ConfusedMatrix[int(validationSet[i,-1]) -1 ,int(validationSet[i,-1]) -1] +=1
            
        elif (predict(tree, validationSet[i,:]) != validationSet[i,-1]):
            ConfusedMatrix[int(validationSet[i,-1]) -1, int(predict(tree, validationSet[i,:])) -1] +=1
    
    RecallVect = np.zeros((1, nbLabels))
    PrecVect = np.zeros((1, nbLabels))
    Classifiacfjdsfsdoifjrw = np.zeros((1,nbLabels))
    
    for i in range(ConfusedMatrix.shape[0]):
        RecallVect[0,i] = ConfusedMatrix[i,i]/np.sum(ConfusedMatrix[i,:])
        PrecVect[0,i] = ConfusedMatrix[i,i]/np.sum(ConfusedMatrix[:,i])
        Classifiacfjdsfsdoifjrw[0,i] = np.trace(ConfusedMatrix) / (np.trace(ConfusedMatrix) + np.sum(ConfusedMatrix[:,i]) + np.sum(ConfusedMatrix[i,:]) - 2*ConfusedMatrix[i,i])
        
    UAR = np.mean(RecallVect)
    UAP = np.mean(PrecVect)
    F1 = 2 * (UAR * UAP) / (UAR + UAP)
    UAClassifiacfjdsfsdoifjrw = np.mean(Classifiacfjdsfsdoifjrw)
    
    #print("Recall [Room 1 | Room 2 | Room 3 | Room 4]")
    #print(RecallVect, end= "UAR ="+str(UAR) +"\n\n")
    #print("Precision [Room 1 | Room 2 | Room 3 | Room 4]")
    #print(PrecVect, end= "UAP ="+str(UAP) +"\n\n")
    #print("Classification Rate [Room 1 | Room 2 | Room 3 | Room 4]")
    #print(Classifiacfjdsfsdoifjrw, end= "Average ="+str(UAR) +"\n\n")
    
    #print(ConfusedMatrix)
    
    return UAR, UAP, F1, UAClassifiacfjdsfsdoifjrw
    
        