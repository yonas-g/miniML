import numpy as np

# SGD, RMSprop, and Adam implemented


class GradientDescent:
    '''
    batch GradientDescent
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


class SGD:
    def __init__(self, cost, lr, epoch):
        '''
        cost: function used to get the cost and gradient
        lr: learning rate
        epoch
        '''
        self.cost = cost
        self.lr = lr
        self.epoch = epoch
        #
        self.theta = None
        self.costHist = []
        # learning schedule params
        self.t0, self.t1 = 5, 50

    def optimize(self, X, y, fit_intercept=False):
        '''
        Note that since instances are picked randomly, some instances may be picked
        several times per epoch, while others may not be picked at all. If you want to
        be sure that the algorithm goes through every instance at each epoch, another
        approach is to shuffle the training set (making sure to shuffle the input
        features and the labels jointly), then go through it instance by instance, then
        shuffle it again, and so on. However, this approach generally converges more
        slowly.
        '''
        if fit_intercept:
            X = np.c_[np.ones((X.shape[1], 1)), X]

        m, n = X.shape
        self.theta = np.random.rand(n, 1)

        for iter in range(self.epoch):
            for i in range(m):
                # get single random instance
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]

                j, grad = self.cost(xi, yi, theta)
                self.costHist.append(j)
                #
                self.lr = self.__learning_schedule(self.epoch * m + i)

                self.theta = self.theta - self.lr * grad

    def __learning_schedule(self, lr):
        '''
        scheduler used when gradually reducing learning rate
        '''
        return self.t0 / (lr + self.t1)
