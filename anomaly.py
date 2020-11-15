import numpy as np


class Gaussian:
    def __init__(self, epsilon=None):
        self.epsilon = epsilon
        self.mu = None
        # covariance_matrix in the case of Multivariate Gaussian
        self.sigma2 = None
        #
        self.X_val = None
        self.y_val = None

    def predict(self, X):
        if self.epsilon == None and (self.X_val is None and self.y_val is None):
            raise Exception(
                'Epsilon Not Set. Initialize with epsilon or fit with X_val and y_val to find optimal value based on the highest F1 score using training data')
            return

        if self.epsilon == None and (self.X_val is not None and self.y_val is not None):
            self.epsilon, _ = self.selectThreshold()

        p = self.predict_probability(X)

        # 1 anomaly, 0 not
        return (p < self.epsilon).astype(int)

    def selectThreshold(self):
        bestF1 = 0
        bestEpsilon = 0
        p_val = self.predict_probability(self.X_val)

        stepsize = (np.max(p_val) - np.min(p_val)) / 1000

        for epsilon in np.arange(np.min(p_val), np.max(p_val), stepsize):
            predictions = (p_val < epsilon).astype(int)

            tp = np.sum((predictions == 1) & (self.y_val.flatten() == 1))
            fp = np.sum((predictions == 1) & (self.y_val.flatten() == 0))
            fn = np.sum((predictions == 0) & (self.y_val.flatten() == 1))

            prec = tp / (tp+fp)
            rec = tp / (tp+fn)
            F1 = (2 * prec * rec) / (prec + rec)

            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon

        return bestEpsilon, bestF1

    @property
    def params(self):
        return {
            'Mean': self.mu,
            'Sigma2': self.sigma2,
            'Epsilon': self.epsilon
        }


class NormalGaussian(Gaussian):
    '''
    Normal Gaussian Distribution for anomaly detection
    '''

    def __init__(self, epsilon=None):
        Gaussian.__init__(self, epsilon)

    def fit(self, X, X_val=None, y_val=None):
        self.X_val = X_val
        self.y_val = y_val

        X = np.array(X)
        m, n = X.shape

        self.mu = np.sum(X, axis=0)/m
        self.sigma2 = np.var(X, axis=0)

    def predict_probability(self, X):
        # we can use Multivariate gaussian by setting sigma2 diagonal matrix
        X = (X - self.mu)**2
        p = 1/(np.sqrt(2*np.pi)*np.sqrt(self.sigma2))*np.exp(-X/2*self.sigma2)

        return np.prod(p, axis=1)


class MultivariateGaussian(Gaussian):
    '''
    Multivariate Gaussian Distribution for anomaly detection.
    Computes the probabilitydensity function of the examples X 
    under the multivariate gaussian distribution with parameters mu and convariance matrix
    '''

    def __init__(self, epsilon=None):
        Gaussian.__init__(self, epsilon)

    def fit(self, X, X_val=None, y_val=None):
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)
        X = np.array(X)

        m, n = X.shape

        self.mu = np.sum(X, axis=0)/m
        self.sigma2 = ((X - self.mu).T).dot(X - self.mu)/m

    def predict_probability(self, X):
        X = X - self.mu
        m, n = X.shape

        p = 1/((2 * np.pi)**(n/2) * np.linalg.det(self.sigma2)**0.5) *\
            np.exp(-0.5 * np.sum(X * X.dot(np.linalg.pinv(self.sigma2)), axis=1))

        return p
