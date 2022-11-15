import numpy as np


class ClfEstimator:
    def __init__(self):
        self._metric = {"accuracy": self.accuracy}
    
    @property
    def metric(self):
        return self._metric

    def accuracy(self, y, y_hat):
        return np.count_nonzero(y.reshape(-1) == y_hat.reshape(-1)) / len(y)


class PCA:
    def __init__(self):
        self._param = None
        self.target = None

    def fit(self, X, target_dim=1):
        X = X - X.mean(axis=0)
        cov = np.cov(X.T) / X.shape[0]
        V, U = np.linalg.eig(cov)
        idx = V.argsort()[::-1]
        V = V[idx]
        U = U[:,idx]
        self.target = target_dim
        self.param = U[:, :target_dim]

    def transform(self, X):
        return X @ self.param[:, :self.target]
    
    def fit_transform(self, X, target_dim=1):
        self.fit(X, target_dim)
        return self.transform(X - X.mean(axis=0))


def grad_finite_diff(f, X, y, w, eps=1e-8):
    grad = list()
    for i in range(len(w)):
        e_i = np.zeros_like(w)
        e_i[i] = eps
        grad.append(f(X, y, w + e_i))
    grad = np.array(grad)

    return (grad - f(X, y, w)) / eps
