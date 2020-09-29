from keras import optimizers
import importlib
import pandas as pd
import GeneralModel
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
importlib.import_module("GeneralModel")
from sklearn import preprocessing

activationArray = ['linear', 'selu','softmax',  'sigmoid', 'elu']
modelsToUse = ['perceptron', 'neuralnet_3L', 'neuralnet_4L']
# will have to make several models with all of the differnet types.  Need to make a method that will graph and save results and makes ure keras compatiable with the forwardSelect.

data = pd.read_csv("../Data/auto-mpg.csv")
data.columns = list(range(1, len(data.columns) +1))
yVals = data.iloc[:,-1] #concrete has 3 outputs,
xVals = data.iloc[:,:-1] # remove those three outputs from x inputs
xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))], allow_duplicates=True)
yVals = pd.DataFrame(preprocessing.scale(yVals.to_numpy().reshape(-1,1)))
allYVals = yVals.iloc[:,-1]

#xVals = pd.DataFrame(preprocessing.scale(xVals))
def tryPerceptron():
    learningRate = .1
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate, clipnorm=.5), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    yVal1 = allYVals
    colsToUse = [0, 4, 6, 3, 1]

    yVal1Param = {'model': modelsToUse[0], 'optName': "Adam", 'optimizer': optimizerDict['Adam'], 'activation': activationArray[3], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse]], "yVals":yVal1, 'num':0}
    for myParams in [yVal1Param]:
        myXLength = len(myParams['xVals'].columns)
        myModel =  GeneralModel.buildModel(myParams['model'], myXLength, myParams['activation'], myParams['optimizer'])
        myHistory = myModel.fit(myParams['xVals'], myParams['yVals'], epochs=200, validation_split=.2, verbose=0)

        plt.plot(myHistory.history['loss'], label="train")
        plt.plot(myHistory.history['val_loss'], label="test")
        plt.title('Model Loss')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./Mpg/BestResults/Images/{}{}{}{}".format(myParams['optName'], myParams['model'],myParams['activation'], myParams['num']))
        plt.clf()

#theModel = Model.buildModel(modelsToUse[0], len(xVals.columns), activationArray[i], optimizerDict[item])
def tryNeural3L():
    learningRate = .1
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate, clipnorm=.5), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    yVal1 = allYVals
    colsToUse = [0, 4, 6, 5, 3]

    yVal1Param = {'model': modelsToUse[1], 'optName': "SGD", 'optimizer': optimizerDict['SGD'], 'activation': activationArray[0], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse]], "yVals":yVal1, 'num':0}
    for myParams in [yVal1Param]:
        myXLength = len(myParams['xVals'].columns)
        myModel =  GeneralModel.buildModel(myParams['model'], myXLength, myParams['activation'], myParams['optimizer'])
        myHistory = myModel.fit(myParams['xVals'], myParams['yVals'], epochs=200, validation_split=.2, verbose=0)

        plt.plot(myHistory.history['loss'], label="train")
        plt.plot(myHistory.history['val_loss'], label="test")
        plt.title('Model Loss')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./Mpg/BestResults/Images/{}{}{}{}".format(myParams['optName'], myParams['model'],myParams['activation'], myParams['num']))
        plt.clf()
def tryNeuralXL():
    learningRate = .1
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=learningRate, clipnorm=.5), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    yVal1 = allYVals
    colsToUse = [0, 4, 6]

    yVal1Param = {'model': modelsToUse[2], 'optName': "Adam", 'optimizer': optimizerDict['Adam'], 'activation': activationArray[0], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse]], "yVals":yVal1, 'num':0}
    for myParams in [yVal1Param]:
        myXLength = len(myParams['xVals'].columns)
        myModel =  GeneralModel.buildModel(myParams['model'], myXLength, myParams['activation'], myParams['optimizer'])

        myHistory = myModel.fit(myParams['xVals'], myParams['yVals'], epochs=200, validation_split=.2, verbose=0)
        plt.plot(myHistory.history['loss'], label="train")
        plt.plot(myHistory.history['val_loss'], label="test")
        plt.title('Model Loss')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./Mpg/BestResults/Images/{}{}{}{}".format(myParams['optName'], myParams['model'],myParams['activation'], myParams['num']))
        plt.clf()
tryPerceptron()
tryNeuralXL()
tryNeural3L()