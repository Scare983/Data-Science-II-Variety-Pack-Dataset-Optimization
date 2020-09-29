from keras import optimizers
import importlib
import pandas as pd
import ConcreteModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(0)
importlib.import_module("ConcreteModel")
from sklearn import preprocessing

activationArray = ['linear', 'selu','softmax',  'sigmoid', 'elu']
modelsToUse = ['perceptron', 'neuralnet_3L', 'neuralnet_4L']
# will have to make several models with all of the differnet types.  Need to make a method that will graph and save results and makes ure keras compatiable with the forwardSelect.

data = pd.read_csv("../Data/Concrete.csv")
data.columns = list(range(1, len(data.columns) +1))
allYVals = data[data.columns[-3:]] #concrete has 3 outputs,
xVals = data.iloc[:,:-3] # remove those three outputs from x inputs
xVals = pd.DataFrame(preprocessing.normalize(xVals))
from sklearn.preprocessing import normalize
#TODO: ASK IF I NEED TO ADD ONE's COLUMNS
xVals.columns = list(range(1,len(xVals.columns)+1))
xVals.insert(loc=0, column=0, value=[1 for x in range(0,len(xVals.values))])
allYVals = pd.DataFrame(preprocessing.normalize(allYVals))
def tryPerceptron():
    learningRate = .1 # Adam
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=.15, clipnorm=.5), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    yVal1 = allYVals.iloc[:,-3]
    yVal2 = allYVals.iloc[:,-2]
    yVal3 = allYVals.iloc[:,-1]
    colsToUse1 = [0, 2, 3]
    colsToUse2 = [0, 4, 2, 3, 1]
    colsToUse3 = [0, 4, 2, 1, 3, 5]
    yVal1Param = {'model': modelsToUse[0], 'optName': "SGD_Moment", 'optimizer': optimizerDict['SGD_Moment'], 'activation': activationArray[2], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse1]], "yVals":yVal1, 'num':0}
    yVal2Param = {'model': modelsToUse[0], 'optName': "Adam", 'optimizer': optimizerDict['Adam'], 'activation': activationArray[4], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse2]], "yVals": yVal2 , 'num':1}
    yVal3Param = {'model': modelsToUse[0], 'optName': "Adam",'optimizer': optimizerDict['Adam'], 'activation': activationArray[2], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse3]], 'yVals': yVal3,'num':2}
    for myParams in [yVal1Param, yVal2Param,yVal3Param ]:
        myXLength = len(myParams['xVals'].columns)
        myModel =  ConcreteModel.buildModel(myParams['model'], myXLength,myParams['activation'], myParams['optimizer'] )

        myHistory = myModel.fit(myParams['xVals'], myParams['yVals'], epochs=200, validation_split=.2, verbose=0)
        plt.plot(myHistory.history['loss'], label="train")
        plt.plot(myHistory.history['val_loss'], label="test")
        plt.title('Model Loss')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./Concrete/BestResults/Images/{}{}{}{}{}".format(myParams['model'], myParams['optName'], myParams['model'],myParams['activation'], myParams['num']))
        plt.clf()


#theModel = Model.buildModel(modelsToUse[0], len(xVals.columns), activationArray[i], optimizerDict[item])
def tryNeural3L():
    learningRate = .1 # Adam
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=.15, clipnorm=.5), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    yVal1 = allYVals.iloc[:,-3]
    yVal2 = allYVals.iloc[:,-2]
    yVal3 = allYVals.iloc[:,-1]
    colsToUse1 = [0, 2, 4, 3]
    colsToUse2 = [0, 4, 2, 3, 1, 5]
    colsToUse3 = [0, 4, 1, 3, 2, 5]
    yVal1Param = {'model': modelsToUse[1], 'optName': "Adam", 'optimizer': optimizerDict['Adam'], 'activation': activationArray[2], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse1]], "yVals":yVal1, 'num':0}
    yVal2Param = {'model': modelsToUse[1], 'optName': "Adam", 'optimizer': optimizerDict['Adam'], 'activation': activationArray[0], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse2]], "yVals": yVal2 , 'num':1}
    yVal3Param = {'model': modelsToUse[1], 'optName': "Adam",'optimizer': optimizerDict['Adam'], 'activation': activationArray[0], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse3]], 'yVals': yVal3,'num':2}
    for myParams in [yVal1Param, yVal2Param,yVal3Param ]:
        myXLength = len(myParams['xVals'].columns)
        myModel =  ConcreteModel.buildModel(myParams['model'], myXLength,myParams['activation'], myParams['optimizer'] )

        myHistory = myModel.fit(myParams['xVals'], myParams['yVals'], epochs=200, validation_split=.2, verbose=0)
        plt.plot(myHistory.history['loss'], label="train")
        plt.plot(myHistory.history['val_loss'], label="test")
        plt.title('Model Loss')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./Concrete/BestResults/Images/{}{}{}{}{}".format(myParams['model'], myParams['optName'], myParams['model'],myParams['activation'], myParams['num']))
        plt.clf()

def tryNeuralXL():
    learningRate = .1 # Adam
    optimizerDict = {"Adam": optimizers.adam(lr=learningRate), "SGD" : optimizers.sgd(lr=.15, clipnorm=.5), "SGD_Moment": optimizers.sgd(lr=learningRate, momentum=.5) }
    yVal1 = allYVals.iloc[:,-3]
    yVal2 = allYVals.iloc[:,-2]
    yVal3 = allYVals.iloc[:,-1]
    colsToUse1 = [0, 3, 2, 4, 1, 7, 6]
    colsToUse2 = [0, 4, 2, 3, 1, 5]
    colsToUse3 = [0, 4, 2, 1, 3]
    yVal1Param = {'model': modelsToUse[2], 'optName': "SGD_Moment", 'optimizer': optimizerDict['SGD_Moment'], 'activation': activationArray[4], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse1]], "yVals":yVal1, 'num':0}
    yVal2Param = {'model': modelsToUse[2], 'optName': "SGD_Moment", 'optimizer': optimizerDict['SGD_Moment'], 'activation': activationArray[0], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse2]], "yVals": yVal2 , 'num':1}
    yVal3Param = {'model': modelsToUse[2], 'optName': "SGD_Moment",'optimizer': optimizerDict['SGD_Moment'], 'activation': activationArray[0], 'epochs':200, 'xVals': xVals.loc[:, [*colsToUse3]], 'yVals': yVal3,'num':2}
    for myParams in [yVal1Param, yVal2Param,yVal3Param ]:
        myXLength = len(myParams['xVals'].columns)
        myModel =  ConcreteModel.buildModel(myParams['model'], myXLength,myParams['activation'], myParams['optimizer'] )

        myHistory = myModel.fit(myParams['xVals'], myParams['yVals'], epochs=200, validation_split=.2, verbose=0)
        plt.plot(myHistory.history['loss'], label="train")
        plt.plot(myHistory.history['val_loss'], label="test")
        plt.title('Model Loss')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend()
        plt.savefig("./Concrete/BestResults/Images/{}{}{}{}{}".format(myParams['model'], myParams['optName'], myParams['model'],myParams['activation'], myParams['num']))
        plt.clf()
tryPerceptron()
tryNeural3L()
tryNeuralXL()