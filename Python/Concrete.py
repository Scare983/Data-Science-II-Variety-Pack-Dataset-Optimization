from keras import optimizers
import importlib
import pandas as pd
import ConcreteModel
from sklearn.metrics import r2_score
from keras import callbacks
from sklearn import preprocessing
import numpy as np
np.random.seed(0)
import sys
importlib.import_module("ConcreteModel")

activationArray = ['selu', 'elu', 'softmax', 'sigmoid', 'linear']
modelsToUse = ['perceptron', 'neuralnet_3L', 'neuralnet_4L']
# will have to make several models with all of the differnet types.  Need to make a method that will graph and save results and makes ure keras compatiable with the forwardSelect.
data = pd.read_csv("../Data/boston.csv")
data.columns = list(range(1, len(data.columns) +1))
allYVals = data[data.columns[-3:]] #concrete has 3 outputs,
xVals = data.iloc[:,:-3] # remove those three outputs from x inputs
#xVals = pd.DataFrame(preprocessing.normalize(xVals))
allYVals = pd.DataFrame(preprocessing.normalize(allYVals))
from sklearn.preprocessing import normalize
#TODO: ASK IF I NEED TO ADD ONE's COLUMNS
xVals.columns = list(range(1,len(xVals.columns)+1))
xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))])
def tryPerceptron(learningRate, epochs):
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=.01), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    for (columnName, columnData) in allYVals.iteritems():
        yVals = allYVals[columnName]


        for item in optimizerDict.keys():
            for i in range(0, len(activationArray)):
                print("\tcolumnIndexTOPredict={} {} {} {} ".format(columnName,item, activationArray[i], modelsToUse[0]))
                #another for loop for models after this
                currentParams = {'model': modelsToUse[0], 'optimizer': optimizerDict[item], 'activation': activationArray[i], 'epochs':epochs, 'lr': learningRate}
                if (item == "SGD" or item == "SGD_Moment") and activationArray[i] == 'linear':
                    if item == "SGD_Moment":
                        currentParams['optimizer'] = optimizers.sgd(lr=learningRate, momentum=.5, clipnorm=1.)
                    else:
                        currentParams['optimizer'] = optimizers.sgd(lr=.05, clipvalue=.4)
                #theModel = Model.buildModel(modelsToUse[0], len(xVals.columns), activationArray[i], optimizerDict[item])

                myCols, myNumFeat, rVals, rCV, rAdj = ConcreteModel.forwardSelectAll(xVals, yVals, currentParams)
                #base case with all features.
                #theModel.fit(xVals, yVals, epochs =200,verbose=0)
                #print("\t\tRValues of trained model with all features:{}".format(r2_score(yVals, theModel.predict(xVals))))
                print("\t\tColumn Index used:{}".format(myCols))
                print("\t\tNumFeatures forwardSelect Chose:{}".format(myNumFeat))
                print("\t\t rCVVales:  {}".format(rCV))
                print("\t\t rBarVaues:  {}".format(rAdj))
                ConcreteModel.graphAttsVRVals(rVals, rCV, rAdj, myNumFeat, item, currentParams, columnName)

def tryNeural3L(learningRate, epochs):

    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    for (columnName, columnData) in allYVals.iteritems():
        yVals = allYVals[columnName]
        for item in optimizerDict.keys():
            for i in range(0, len(activationArray)):
                print("\tcolumnIndexTOPredict={} {} {} {} ".format(columnName,item, activationArray[i], modelsToUse[1]))
                #another for loop for models after this
                currentParams = {'model': modelsToUse[1], 'optimizer': optimizerDict[item], 'activation': activationArray[i], 'epochs':epochs, 'lr': learningRate}
                if (item == "SGD" or item == "SGD_Moment") and activationArray[i] == 'linear':
                    if item == "SGD_Moment":
                        currentParams['optimizer'] = optimizers.sgd(lr=learningRate, momentum=.5, clipnorm=1.)
                    else:
                        currentParams['optimizer'] = optimizers.sgd(lr=learningRate, clipnorm=1.)


                myCols, myNumFeat, rVals, rCV, rAdj = ConcreteModel.forwardSelectAll(xVals, yVals, currentParams)
                #base case with all features.

                print("\t\tColumn Index used:{}".format(myCols))
                print("\t\tNumFeatures forwardSelect Chose:{}".format(myNumFeat))
                print("\t\t rCVVales:  {}".format(rCV))
                print("\t\t rBarVaues:  {}".format(rAdj))
                ConcreteModel.graphAttsVRVals(rVals, rCV, rAdj, myNumFeat, item, currentParams, columnName)

def tryNeuralXL(learningRate, epochs):

    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    for (columnName, columnData) in allYVals.iteritems():
        yVals = allYVals[columnName]
        for item in optimizerDict.keys():
            for i in range(0, len(activationArray)):
                print("\tcolumnIndexTOPredict={} {} {} {} ".format(columnName,item, activationArray[i], modelsToUse[2]))
                #another for loop for models after this
                currentParams = {'model': modelsToUse[2], 'optimizer': optimizerDict[item], 'activation': activationArray[i], 'epochs':epochs, 'lr': learningRate}
                if (item == "SGD" or item == "SGD_Moment") and activationArray[i] == 'linear':
                    if item == "SGD_Moment":
                        currentParams['optimizer'] = optimizers.sgd(lr=learningRate, momentum=.5, clipnorm=1.)
                    else:
                        currentParams['optimizer'] = optimizers.sgd(lr=learningRate, clipnorm=1.)

                myCols, myNumFeat, rVals, rCV, rAdj = ConcreteModel.forwardSelectAll(xVals, yVals, currentParams)
                #base case with all features.

                print("\t\tColumn Index used:{}".format(myCols))
                print("\t\tNumFeatures forwardSelect Chose:{}".format(myNumFeat))
                print("\t\t rCVVales:  {}".format(rCV))
                print("\t\t rBarVaues:  {}".format(rAdj))
                ConcreteModel.graphAttsVRVals(rVals, rCV, rAdj, myNumFeat, item, currentParams, columnName)

tryPerceptron(float(sys.argv[1]),int(sys.argv[2]))
tryNeural3L(float(sys.argv[1]),int(sys.argv[2]))
tryNeuralXL(float(sys.argv[1]),int(sys.argv[2]))
