from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class MyLinearRegression(BaseEstimator):
    def __init__(self, batch_size = -1, n_epochs = 50, shuffle = False, holdout_size = 0.2, l2 = 100, learning_rate = 0.12, decay = 1, standardize = 3):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.holdout_size = holdout_size
        self.l2 = l2
        self.learning_rate = learning_rate
        self.decay = decay
        self.standardize = standardize
    
    
    def __standardization(self, X):
        if self.standardize == 1:
            mean = np.mean(X, axis = 0, dtype = np.float64)
            std = np.std(X, axis = 0, dtype = np.float64)
            meanPrepared = np.tile(mean,(X.shape[0], 1))
            stdPrepared = np.tile(std, (X.shape[0], 1))
            X -= meanPrepared
            X /= stdPrepared
            return X
        elif self.standardize == 0:
            mean = np.mean(X, axis = 0, dtype = np.float64)
            maxi = np.amax(X, axis = 0)
            mini = np.amin(X, axis = 0)
            rangeOf = maxi - mini
            rangeOf[rangeOf == 0.] += 1
            meanPrepared = np.tile(mean,(X.shape[0], 1))
            rangePrepared = np.tile(rangeOf, (X.shape[0], 1))
            X -= meanPrepared
            X /= rangePrepared
            return X
        else:
            return X
    
    def __holdouter(self, X, y):
        if len(X) < 4:
            return X, y
        else:
            holdOutIt = int(np.floor(len(X) * self.holdout_size))
            self.__heldOut = [X[0 : holdOutIt + 1, :], y[0 : holdOutIt + 1]]
            return X[holdOutIt + 1 : len(X), :], y[holdOutIt + 1 : len(y)]
    
    def __batcher(self, X, y):
        self.__batchesX = []
        self.__batchesY = []
        sizeB = self.batch_size
        if self.batch_size == -1:
            sizeB = len(X)
        index1 = 0
        while index1 < X.shape[0]:
            self.__batchesX.append(X[index1 : min(index1 + sizeB, X.shape[0]), :])
            self.__batchesY.append(y[index1 : min(index1 + sizeB, y.shape[0])])
            index1 += sizeB
    
    
    def __shuffler(self, x, y):
        assert len(x) == len(y)
        if self.shuffle:
            p = np.random.permutation(len(x))
            return x[p], y[p]
        else:
            return x, y
    
    
    def __cost(self, X, y):
        costs = X.dot(self.__prediction) - y
        costs = costs**2
        costs /= len(X)
        return np.sum(costs)
    
    
    def __updateWeights(self, X, y):
        alpha = self.learning_rate
        m = X.shape[0]
        theta = self.__prediction
        self.__prediction = theta - ((alpha / m) * ((X.T).dot(X.dot(theta) - y)))
        
        theta = self.__prediction
        thetaZero = theta[0]
        multiply = alpha * self.l2 / m
        self.__prediction = theta - multiply * theta
        self.__prediction[0] = thetaZero
    
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = (np.array(X)).astype(np.float64)
        y = (np.array(y)).astype(np.float64)
        self.__prediction = np.ones(X.shape[1])
        X = self.__standardization(X)
        X, y = self.__holdouter(X, y)
        
        for i in range(self.n_epochs):
            X, y = self.__shuffler(X, y)
            self.__batcher(X, y)
            for j in range(len(self.__batchesX)):
                self.__updateWeights(self.__batchesX[j], self.__batchesY[j])
            self.learning_rate *= self.decay
            
        return self
        
        
    def predict(self, X):
        X = check_array(X)
        X = np.array(X).astype(np.float64)
        X = self.__standardization(X)
        return X.dot(self.__prediction)
