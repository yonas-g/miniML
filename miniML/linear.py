import numpy as np
from optimizer import Gradient

# each model shoudl have fit and predict


class LinearRegression:
    '''
    LinearRegression
    '''
    # def fit()
    # def predict()

    def cost(self, X, y, theta):
        m, n = self.X.shape
        h = X.dot(theta)

        J = 1/(2*m)*np.sum((h - y)**2)

        grad = X.T.dot(X.dot(theta) - y)

        return J, grad


class LogisticRegression:
    '''
    LogisticRegression
    '''
    # def fit()
    # def predict()
