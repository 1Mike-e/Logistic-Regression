from __future__ import division
import numpy as np
import classalgorithms as algs

def splitdataset(dataset, trainsize=80000, testsize=20000, testfile=None):
    randindices = np.random.randint(0, dataset.shape[0], trainsize + testsize)
    numinputs = dataset.shape[1] - 1
    Xtrain = dataset[randindices[0:trainsize], 0:numinputs]
    ytrain = dataset[randindices[0:trainsize], numinputs]
    Xtest = dataset[randindices[trainsize:], 0:numinputs]
    ytest = dataset[randindices[trainsize:], numinputs]

    if testfile is not None:
        testdataset = np.genfromtxt(testfile, delimiter=',')
        Xtest = testdataset[:, 0:numinputs]
        ytest = testdataset[:, numinputs]

    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0], 1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0], 1))))
    return ((Xtrain, ytrain), (Xtest, ytest))

def getaccuracy(ytest, predictions):
    correct = sum(ytest[i] == predictions[i] for i in range(len(ytest)))
    return (correct / float(len(ytest))) * 100.0

def loadsusy():
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')
    return splitdataset(dataset)

def loadmadelon():
    datasettrain = np.genfromtxt('madelon/madelon_train.data', delimiter=' ')
    trainlab = np.genfromtxt('madelon/madelon_train.labels', delimiter=' ')
    trainlab[trainlab == -1] = 0
    trainsetx = np.hstack((datasettrain, np.ones((datasettrain.shape[0],1))))
    trainset = (trainsetx, trainlab)

    datasettest = np.genfromtxt('madelon/madelon_valid.data', delimiter=' ')
    testlab = np.genfromtxt('madelon/madelon_valid.labels', delimiter=' ')
    testlab[testlab == -1] = 0
    testsetx = np.hstack((datasettest, np.ones((datasettest.shape[0],1))))
    testset = (testsetx, testlab)

    return trainset, testset

if __name__ == '__main__':
    trainset, testset = loadsusy()
    print('Running on train={0} and test={1} samples'.format(trainset[0].shape[0], testset[0].shape[0]))

    nnparams = {'ni': trainset[0].shape[1], 'nh': 64, 'no': 1}
    classalgs = {
        'Random': algs.Classifier(),
        'Linear Regression': algs.LinearRegressionClass(),
        'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
        'Naive Bayes Ones': algs.NaiveBayes(),
        'Logistic Regression (Batch)': algs.LogitReg({'stochastic': False}),
        'Logistic Regression (Stochastic)': algs.LogitReg({'stochastic': True}),
        'Neural Network': algs.NeuralNet(nnparams)
    }

    for learnername, learner in classalgs.items():
        print('Running learner =', learnername)
        learner.learn(trainset[0], trainset[1])
        predictions = learner.predict(testset[0])
        accuracy = getaccuracy(testset[1], predictions)
        print('Accuracy for', learnername + ':', accuracy)