import numpy as np

'''
TODO
- predict
'''


class CollaborativeFiltering:
    def __init__(self, n_features, n_iters, lr, lamb):
        self.n_features = n_features
        self.n_iters = n_iters
        self.lamb = lamb
        self.lr = lr
        #
        self.X = None
        self.theta = None
        self.y = None
        self.r = None
        #
        self.cost_hist = []

    def fit(self, y, r):
        '''
        y contains a matrix of item vs user.
        r is an indicator
        '''
        m, n = y.shape
        # think about bias val 1
        self.X = np.random.rand(m, self.n_features)
        self.theta = np.random.rand(n, self.n_features)
        self.y = y
        self.r = r

        self.__minimize()

    def predict(self):
        '''
        How is the input formatted
        '''
        return 0

    def __cost(self, X, theta, y, r, lamb):
        '''
        Args:
        X: feature matrix of items
        lamb: lambda
        '''
        J = 1/2*np.sum(r * (X.dot(theta.T) - y)**2) + lamb / \
            2 * np.sum(theta**2) + np.sum(X**2)

        X_grad = (r * (X.dot(theta.T) - y)).dot(theta) + lamb * X
        Theta_grad = (r * (X.dot(theta.T) - y)).T.dot(X) + lamb * theta

        return J, X_grad, Theta_grad

    def __minimize(self):
        '''
        minimizes X and Theta values
        '''

        for i in range(self.n_iters):

            J, X_grad, Theta_grad = self.__cost(
                self.X, self.theta, self.y, self.r, self.lamb)

            self.X = self.X - lr * X_grad
            self.theta = self.theta - lr * Theta_grad

            self.cost_hist.append(J)

    @property
    def history(self):
        return {
            'Cost': self.cost_hist,
            'n_features': n_features,
            'n_iters': n_iters,
            'lambda': lamb,
            'lr': lr
        }
