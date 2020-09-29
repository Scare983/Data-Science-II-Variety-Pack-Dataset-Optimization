from keras import optimizers
import importlib
import pandas as pd
import GeneralModel
from sklearn.metrics import r2_score
from sklearn import preprocessing

import numpy as np
import sys


np.random.seed(0)
importlib.import_module("GeneralModel")

activationArray = ['selu', 'elu', 'softmax', 'sigmoid', 'linear']

modelsToUse = ['perceptron', 'neuralnet_3L', 'neuralnet_4L']

# will have to make several models with all of the differnet types.  Need to make a method that will graph and save results and makes ure keras compatiable with the forwardSelect.



def tryPerceptron(fName, learningRate, epochs):
    data = pd.read_csv("../Data/{}.csv".format(fName))
    data.columns = list(range(1, len(data.columns) +1))
    yVals = data.iloc[:,-1] #concrete has 3 outputs,
    xVals = data.iloc[:,:-1] # remove those three outputs from x inputs
    xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))], allow_duplicates=True)
    yVals = pd.DataFrame(preprocessing.scale(yVals.to_numpy().reshape(-1,1)))
    yVals = yVals.iloc[:,-1]
    # I FORGOT   TO SCALLE
    xVals = pd.DataFrame(preprocessing.scale(xVals))
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.2) }
    for item in optimizerDict.keys():
        for i in range(0, len(activationArray)):
            print("\t{} {} {} ".format(item, activationArray[i], modelsToUse[0]))
            #another for loop for models after this
            currentParams = {'model': modelsToUse[0], 'optimizer': optimizerDict[item], 'activation': activationArray[i], 'epochs':epochs, 'lr': learningRate}
            if (item == "SGD" or item == "SGD_Moment") and activationArray[i] == 'linear':
                if item == "SGD_Moment":
                    currentParams['optimizer'] = optimizers.sgd(lr=learningRate, momentum=.5, clipnorm=1.)
                else:
                    currentParams['optimizer'] = optimizers.sgd(lr=learningRate, clipnorm=1.)

            myCols, myNumFeat, rVals, rCV, rAdj = GeneralModel.forwardSelectAll(xVals, yVals, currentParams)
            #base case with all features.

            print("\t\tColumn Index used:{}".format(myCols))
            print("\t\tNumFeatures forwardSelect Chose:{}".format(myNumFeat))
            print("\t\t rCVVales:  {}".format(rCV))
            print("\t\t rBarVaues:  {}".format(rAdj))
            GeneralModel.graphAttsVRVals(rVals, rCV, rAdj, myNumFeat, item, currentParams, 0, fName)

def tryNeural3L(fName, learningRate, epochs):
    data = pd.read_csv("../Data/{}.csv".format(fName))
    data.columns = list(range(1, len(data.columns) +1))
    yVals = data.iloc[:,-1] #concrete has 3 outputs,
    xVals = data.iloc[:,:-1] # remove those three outputs from x inputs
    xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))], allow_duplicates=True)
    yVals = pd.DataFrame(preprocessing.scale(yVals.to_numpy().reshape(-1,1)))
    yVals = yVals.iloc[:,-1]
    xVals = pd.DataFrame(preprocessing.scale(xVals))
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    for item in optimizerDict.keys():
        for i in range(0, len(activationArray)):
            print("{} {} {} ".format(item, activationArray[i], modelsToUse[1]))
            #another for loop for models after this
            currentParams = {'model': modelsToUse[1], 'optimizer': optimizerDict[item], 'activation': activationArray[i], 'epochs':epochs, 'lr': learningRate }
            if (item == "SGD" or item == "SGD_Moment") and activationArray[i] == 'linear':
                if item == "SGD_Moment":
                    currentParams['optimizer'] = optimizers.sgd(lr=learningRate, momentum=.5, clipnorm=1.)
                else:
                    currentParams['optimizer'] = optimizers.sgd(lr=learningRate, clipnorm=1.)
            myCols, myNumFeat, rVals, rCV, rAdj = GeneralModel.forwardSelectAll(xVals, yVals, currentParams)
            #base case with all features.
            print("\t\tColumn Index used:{}".format(myCols))
            print("\t\tNumFeatures forwardSelect Chose:{}".format(myNumFeat))
            print("\t\t rCVVales:  {}".format(rCV))
            print("\t\t rBarVaues:  {}".format(rAdj))
            GeneralModel.graphAttsVRVals(rVals, rCV, rAdj, myNumFeat, item, currentParams, 0, fName)

def tryNeuralXL(fName, learningRate, epochs):
    data = pd.read_csv("../Data/{}.csv".format(fName))
    data.columns = list(range(1, len(data.columns) +1))
    yVals = data.iloc[:,-1] #concrete has 3 outputs,
    xVals = data.iloc[:,:-1] # remove those three outputs from x inputs
    xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))], allow_duplicates=True)
    yVals = pd.DataFrame(preprocessing.scale(yVals.to_numpy().reshape(-1,1)))
    yVals = yVals.iloc[:,-1]
    xVals = pd.DataFrame(preprocessing.scale(xVals))


    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    for item in optimizerDict.keys():
        for i in range(0, len(activationArray)):
            print("{} {} {} ".format(item, activationArray[i], modelsToUse[2]))
            #another for loop for models after this
            currentParams = {'model': modelsToUse[2], 'optimizer': optimizerDict[item], 'activation': activationArray[i], 'epochs':epochs, 'lr': learningRate}
            if (item == "SGD" or item == "SGD_Moment") and activationArray[i] == 'linear':
                if item == "SGD_Moment":
                    currentParams['optimizer'] = optimizers.sgd(lr=learningRate, momentum=.5, clipnorm=1.)
                else:
                    currentParams['optimizer'] = optimizers.sgd(lr=learningRate, clipnorm=1.)
            myCols, myNumFeat, rVals, rCV, rAdj = GeneralModel.forwardSelectAll(xVals, yVals, currentParams)
            #base case with all features.


            print("\t\tColumn Index used:{}".format(myCols))
            print("\t\tNumFeatures forwardSelect Chose:{}".format(myNumFeat))
            print("\t\t rCVVales:  {}".format(rCV))
            print("\t\t rBarVaues:  {}".format(rAdj))
            GeneralModel.graphAttsVRVals(rVals, rCV, rAdj, myNumFeat, item, currentParams, 0, fName)
#tryPerceptron()


if sys.argv[1] == 'perceptron':
    print('here')
    tryPerceptron(sys.argv[2],float(sys.argv[3]), int(sys.argv[4]))
elif sys.argv[1] == '3L':
    tryNeural3L(sys.argv[2],float(sys.argv[3]), int(sys.argv[4]))
elif sys.argv[1] == 'XL':
    tryNeuralXL(sys.argv[2],float(sys.argv[3]), int(sys.argv[4]))