import numpy as np
import scipy as sc


class BaseSmoothOracle:
    def func(self, w):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        if isinstance(X, sc.sparse._csr.csr_matrix):
            X = X.toarray()
        
        M = y.reshape(-1, 1) * (X @ w.reshape(-1, 1))
        losses = np.logaddexp(0, -M)

        Loss_term = 1 / X.shape[0] * np.sum(losses)
        L2_term = self.l2_coef / 2 * (w @ w)

        return Loss_term + L2_term

    def grad(self, X, y, w):
        if isinstance(X, sc.sparse._csr.csr_matrix):
            X = X.toarray()
        M = y.reshape(-1, 1) * (X @ w.reshape(-1, 1))
        M = np.clip(M, -100, 100)
        mul = X * y.reshape(-1, 1)
        Loss_term = - 1 / X.shape[0] * np.sum(mul /
                                              (1 + np.exp(M)), axis=0)
        L2_term = self.l2_coef * w

        return Loss_term + L2_term
