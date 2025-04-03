from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}
    """
    def __init__(self, params=None):
        pass

    def learn(self, Xtrain, ytrain):
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    """
    def __init__(self, params=None):
        self.weights = None
        self.regwgt = params.get('regwgt', 0.01) if params else 0.01

    def learn(self, Xtrain, ytrain):
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        numsamples = Xtrain.shape[0]
        reg = self.regwgt * np.identity(Xtrain.shape[1])
        self.weights = np.linalg.inv(np.dot(Xtrain.T, Xtrain)/numsamples + reg)
        self.weights = np.dot(self.weights, np.dot(Xtrain.T, yt))/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest <= 0] = 0
        return ytest

class NaiveBayes(Classifier):
    def __init__(self, params=None):
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']

class LogitReg(Classifier):
    """
    Logistic Regression with support for batch and stochastic gradient descent
    """
    def __init__(self, params=None):
        self.weights = None
        self.stepsize = 0.00001
        self.reps = 300
        self.batch_size = 32
        self.stochastic = params.get('stochastic', False) if params else False

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def learn(self, Xtrain, ytrain):
        n, d = Xtrain.shape
        self.weights = np.zeros(d)

        for epoch in range(self.reps):
            indices = np.random.permutation(n)

            if self.stochastic:
                for i in range(n):
                    xi = Xtrain[i]
                    yi = ytrain[i]
                    pred = self.sigmoid(np.dot(self.weights, xi))
                    grad = (pred - yi) * xi
                    self.weights -= self.stepsize * grad
            else:

                batch_size = self.batch_size
                num_batches = n // batch_size

                for i in range(0, n, batch_size):  
                    batch_indices = indices[i:i + batch_size]
                    Xbatch = Xtrain[batch_indices]
                    ybatch = ytrain[batch_indices]

                    preds = self.sigmoid(np.dot(Xbatch, self.weights))
                    grad = np.dot(Xbatch.T, (preds - ybatch)) / batch_size
                    self.weights -= self.stepsize * grad

    def predict(self, Xtest):
        probs = self.sigmoid(np.dot(Xtest, self.weights))
        return utils.threshold_probs(probs)

class NeuralNet(Classifier):
    def __init__(self, params=None):
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid
        self.stepsize = 0.01
        self.reps = 5
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp,:], ytrain[samp])

    def evaluate(self, inputs):
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        ah = self.transfer(np.dot(self.wi, inputs))
        ao = self.transfer(np.dot(self.wo, ah))
        return (ah, ao)

    def update(self, inp, out):
        pass  # Not implemented