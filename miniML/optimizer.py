import numpy as np


class Gradient:
    '''
    gradient
    '''
    # demo
    def gradientDescent(X, y, theta, learning_rate, epoch, fit_intercept=False):
        if fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)

        m, n = X.shape
        J_hist = []

        y = np.array(y).reshape(m, 1)

        theta = np.array(theta).reshape(n, 1)

        for iter in range(epoch):
            h = sigmoid(X.dot(theta))
            d = h - y
            #         gradient = (1/m)*X.T.dot(h-y)
            # we pass __cost as parameter
            cost, gradient = __cost(X, y, theta)

            theta = theta - (learning_rate/m * gradient)

            J_hist.append(cost)

        return theta, J_hist
