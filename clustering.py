import numpy as np

'''
TODO
- KMeans: iterate until there is not change is cluster_centers rather than n_iters
'''


class KMeans:
    def __init__(self, n_clusters, n_iters=500):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.cluster_centers = np.zeros(n_clusters)

    def fit(self, X):
        steroid_index = np.random.randint(X.shape[0], size=self.n_clusters)
        self.cluster_centers = X[steroid_index]
        c = np.zeros(X.shape[0])

        for i in range(self.n_iters):

            for item in range(X.shape[0]):
                closest = np.argmin(
                    np.sum(np.square(X[item]-self.cluster_centers), axis=1))
                c[item] = closest

            for k in range(self.n_clusters):
                k_vals = X[np.where(c == k)]
                self.cluster_centers[k] = np.sum(
                    k_vals, axis=0)/k_vals.shape[0]

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            closest = np.argmin(
                np.sum(np.square(X[i]-self.cluster_centers), axis=1))
            preds.append(closest)
        return preds

    @property
    def cluster_centers_(self):
        return self.cluster_centers


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        #
        e = X.T.dot(X)/X.shape[0]

        u, s, v = np.linalg.svd(e)
        self.u = u
        self.s = s
        self.v = v

        u_reduce = u[:, :self.n_components]

        z = X.dot(u_reduce)

        self.z = z
        self.X_approx = z.dot(u_reduce.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.z

    @property
    def X_approx(self):
        '''
        X values after approximation with n_components dimension
        '''
        return self.X_approx

    @property
    def loss(self):
        '''
        returns variance loss after approximation.
        0.01(1%) loss is preferable
        '''
        return 1 - (np.sum(self.s[:self.n_components]) / np.sum(self.s))
