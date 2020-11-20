import numpy as np

# SGD, RMSprop, and Adam implemented


class GradientDescent:
    '''
    gradient
    '''

    def __init__(self, cost, lr, n_iters):
        '''
        cost: function used to get the cost and gradient
        lr: learning rate
        n_iters: number of iterations
        '''
        self.cost = cost
        self.lr = lr
        self.n_iters = n_iters
        #
        self.theta = None
        self.costHist = []

    def optimize(self, X, y, fit_intercept=False):
        '''
        called to find optimal value of theta.
        if fit_intercept is `True` column axis of ones will be added to X
        '''
        if fit_intercept:
            X = np.c_[np.ones((X.shape[1], 1)), X]

        m, n = X.shape
        self.theta = np.random.rand(n, 1)

        for iter in range(self.n_iters):
            j, grad = self.cost(X, y, self.theta)
            self.costHist.append(j)
            self.theta = self.theta - self.lr/m * grad

        return self.theta

    @property
    def cost_hist_(self):
        return self.costHist

    @property
    def theta_(self):
        return self.theta
