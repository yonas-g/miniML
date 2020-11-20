import numpy as np
from optim import SDG

# each model should have fit and predict
# regularized


class LinearRegression:
    '''
    LinearRegression
    '''

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        theta_best_svd, residuals, rank, s = np.linalg.lstsq(
            X, y, rcond=1e-6)

        self.theta = theta_best_svd

    def predict(self, X):
        return X.dot(self.theta)

    def __cost(self, X, y, theta):
        m, n = X.shape
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
