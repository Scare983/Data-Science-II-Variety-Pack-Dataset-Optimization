from keras import Sequential, optimizers, callbacks
from keras import losses
from keras.layers import Dense
#from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import math


DEBUG=False
np.random.seed(0)
def debugPrint( sentence):
    if DEBUG:
        print(sentence)

def buildModel( modelName,nfeat,act_func='tanh',optimzer=optimizers.adam(lr=0.1)):
    model = Sequential()
    if modelName == 'perceptron':
        model.add(Dense(8,input_dim=nfeat,activation=act_func, use_bias=True))
        model.add(Dense(1))  #1st Hidden Layer - 3 neuron

    if modelName == 'neuralnet_3L':
        model.add(Dense(13,input_dim=nfeat,activation=act_func, use_bias=True))               #1st Hidden Layer
        model.add(Dense(5,activation=act_func))
        model.add(Dense(1))                               #Output Layer
    if modelName == 'neuralnet_4L':
        model.add(Dense(5,input_dim=nfeat,activation=act_func, use_bias=True))               #1st Hidden Layer
        model.add(Dense(7,activation=act_func))
        model.add(Dense(10,activation=act_func))   #2nd Hidden Layer
        model.add(Dense(1))                               #Output Layer
    model.compile(loss=losses.mean_squared_error, optimizer=optimzer)

    return model

def forwardSelect( XIndices, colsToUse, xVals, yVals, modelParams, previousRBase):

    # param:  XIndices is an array of all incices of the X attributes
    # param: colsTouse is ab array of selected columns that the model should use for the base.
    # param: modelToUse is the model to predict with.  it should be passed with the object instantiated
    initList = XIndices
    featuresToUse = colsToUse
    # find what index we need to test.  e.i what is not already in the model
    featuresToTest = list(set(initList) - set(featuresToUse))

    # analyzer will the index of attribute that was tested, and the new r-value the model created with the added attribute
    analyzer = pd.Series(index=featuresToTest)
    debugPrint("Features Index to use: {}".format(featuresToUse))
    debugPrint("features Index to test {}".format(featuresToTest))
    debugPrint(xVals.iloc[:,[*featuresToUse]].shape)
    debugPrint(analyzer.index)
    # Get all the columns we need
    xInput = xVals.loc[:,[*featuresToUse]]

    # have to build
    myBaseModel = buildModel(modelParams['model'], len(xInput.columns), modelParams['activation'], modelParams['optimizer'])
    # yVals = yVals.tolist()
    myBaseModel.fit(x=xInput, y=yVals, epochs=modelParams['epochs'], verbose=0)

    #myBaseModel = modelToUse.fit(xInput, yVals)
    # myBaseRSquare = myBaseModel.score(xVals.iloc[:,[*featuresToUse]], yVals.tolist())
    # adjBase = adj_r2(myBaseModel, yVals.tolist(), myBaseModel.predict(xVals.iloc[:,[*featuresToUse]]))

    myBaseRSquare = previousRBase

    #debugPrint(r2_score(yVals.tolist(),myBaseModel.predict(xVals.iloc[:,[*featuresToUse]])))
    # debugPrint("Base Column List:\t{}\nRsquare value of base Model:\t{}".format(colsToUse, myBaseRSquare) )
    # feature to test is list of indices.
    for newColumn in featuresToTest:
        # add the feature we want to test to the base X attribtues  we are already using
        combinedX = featuresToUse + [newColumn]
        debugPrint("CombinedX indeces:{}".format(combinedX))
        # create model with the indices we want to experiment with.

        myModel = buildModel(modelParams['model'], len(combinedX), modelParams['activation'], modelParams['optimizer'])
        myModel.fit(xVals.iloc[:,[*combinedX]], yVals, epochs=modelParams['epochs'], verbose=0) #modelToUse.fit(xVals.iloc[:,[*combinedX]], yVals)
        # add rsquare valeus at location of the index we are experimenting with.

        analyzer.loc[newColumn] = r2_score(yVals,myModel.predict(xVals.loc[:,[*combinedX]])) # this gets rsquare value.
        debugPrint(analyzer.index)
    #if there is a max value in anaylzer.  Let us add that index to the list if the rsquare is better than the current models.
    if (analyzer.max()):
        if (analyzer.max() > myBaseRSquare):
            # index of what column to add that gave best r-value, the actual r-value, rsquar-bar value, and rcv value
            return analyzer.idxmax(), analyzer.max()
    # else no added features is better than base model
    return -1, None

def forwardSelectAll( xAtts, yAtts, modelParams):

    xIndices = list( (range(0,len(xAtts.columns))))
    rSqVals = [0] # R^2, R^2 Bar, R^2 cv
    rCvVals = [0]
    rBarVals = [0]
    cols = [0] # Question?:  does this value represent the 1's column.
    # going to iterate through each value in xIndexArray and pass to forwardSelectMethod to determine rVals
    numFeatures = [1]
    for i in range(0, len(xAtts.columns)):
        myY = yAtts
        next_jIdx, next_j = forwardSelect(xIndices, cols, xAtts, myY, modelParams, rSqVals[-1])
        if(next_jIdx ==-1):
            debugPrint("No other columns to add.")
            break # means we found all columns that are significant.
        cols.append(next_jIdx)
        numFeatures.append(numFeatures[-1]+1)
        rSqVals.append(next_j)# calcualte rsquare, rsquarebar, and rcv here.


        ## KFOLD R Value
        kfold = KFold(n_splits=5, shuffle=True)
        cvScoreArray = []
        for train, test in kfold.split(xAtts.iloc[:,[*cols]], yAtts):
            model = buildModel(modelParams['model'], numFeatures[-1], modelParams['activation'], modelParams['optimizer'])
            model.fit(xAtts.iloc[train, [*cols]], yAtts.iloc[train], verbose=0, epochs=modelParams['epochs'])
            cvScoreArray.append(r2_score(yAtts.iloc[test], model.predict(xAtts.iloc[test,[*cols]])))




        rCvVals.append(sum(cvScoreArray)/len(cvScoreArray))

        rBarVals.append(adj_r2(next_j, numFeatures[-1], len(yAtts)))

        debugPrint("Next index to add to column list:\t{}\nRsquare value with this index added is:\t{}".format(next_jIdx, next_j))
    return cols, numFeatures, rSqVals, rCvVals, rBarVals


def adj_r2(rSquare, numIndependentVar, numSamples):
    return 1-(1-rSquare)*(numSamples -1)/ (numSamples - numIndependentVar - 1)



def r2_score(y, x):

    x = x.flatten()

    zx = (x-np.mean(x))/np.std(x, ddof=1)
    zy = (y-np.mean(y))/np.std(y, ddof=1)
    r = np.sum(zx*zy)/(len(x)-1)
    return round(r**2,4 )

# Utility method
def graphAttsVRVals(rsq, rCv, rbar, numParams, item,  modelParams, columnName):
    import matplotlib.pyplot as plt
    plt.plot(numParams, rsq, label="R^2")
    plt.plot( numParams, rCv, label="rCv")
    plt.plot(numParams, rbar,  label="rAdj")
    plt.xlabel("Number params")
    plt.ylabel("RVals")
    plt.title("RVals for Y={}".format(columnName))
    plt.legend()
    plt.savefig("./Concrete/AllResults/Images/{}{}{}Epochs{}Eta{}Y{}.jpg".format(item, modelParams['model'], modelParams['activation'], modelParams['epochs'], modelParams['lr'], columnName))
    plt.clf()


if __name__ == '__main__':
    learningRate = .01
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate, clipnorm=1.), "SGD" : optimizers.sgd(lr=learningRate, clipnorm=1.), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.9, clipnorm=1.) }
    activationArray = ['tanh', 'sigmoid', 'linear', 'relu']
    modelsToUse = ['perceptron', 'neuralnet_3L', 'neuralnet_4L']
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    data = pd.read_csv("../Data/new.csv")
    data.columns = list(range(1, len(data.columns) +1))
    yVals = data.loc[:,data.columns[-1]]
    xVals = data.iloc[:,:-1]
    #TODO: ASK IF I NEED TO ADD ONE's COLUMNS
    xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))], allow_duplicates=True)
    xVals.fillna(xVals.mean(), inplace=True)
    currentParams = {'model': modelsToUse[0], 'optimizer': optimizerDict["SGD"], 'activation': activationArray[2], 'epochs':1000}
    theModel = buildModel(modelsToUse[0], len(xVals.columns), activationArray[0],  optimizerDict["SGD"])
    myCols, myNumFeat, rVals, rcv, radj = forwardSelectAll(theModel, xVals, yVals, currentParams)