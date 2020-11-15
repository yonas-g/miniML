import numpy as np
from optimizer import Gradient

# each model shoudl have fit and predict


class LinearRegression:
    '''
    LinearRegression
    '''
    # def fit()
    # def predict()

    def __cost(self, X, y, theta):
        m, n = self.X.shape
        h = X.dot(theta)

        J = 1/(2*m)*np.sum((h - y)**2)

        grad = X.T.dot(h - y)

        return J, grad


class LogisticRegression:
    '''
    LogisticRegression
    '''
    # def fit()
    # def predict()

    def sigmoid(self, z):
        '''
        A sigmoid function is a mathematical function having a characteristic 
        "S"-shaped curve or sigmoid curve. 

        1.0 / (1.0 + np.exp(-z))
        '''
        return 1.0 / (1.0 + np.exp(-z))

    def __cost(self, X, y, theta):
        m, n = X.shape

        h = self.sigmoid(X.dot(theta))

        J = 1/m*(-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

        # include 1/m?
        grad = X.T.dot(h-y)

        return J, grad
