import numpy as np
import scipy as sc

# testing numerical grad implementation
def f(X, y, w):
    return np.sum(X)

def g(X, y, w):
    return np.sum(w)

def h(X, y, w):
    return w[0]**2 + w[2]

def phi(X, y, w):
    return 0.5 * (w @ w)

def psi(X, y, w):
    return np.linalg.norm(5 * w) - w[0]

test_functions = [f, g, h, phi, psi]
ans_functions = [
    np.array([0, 0, 0, 0]),
    np.array([1, 1, 1, 1]),
    np.array([2, 0, 1, 0]),
    np.array([1, 1, 1, 1]),
    np.array([1.5, 2.5, 2.5, 2.5])
]

# testing BinaryLogistic grad
X1 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
y1 = np.array([1, -1, 1])
w1 = np.array([3.01, -1.2, 0.03])

X2 = np.array([
    [1, 0],
    [8, 0],
    [7, 8]
])
y2 = np.array([1, 1, -1])
w2 = np.array([3.02, -0.2])

X3 = np.array([
    [1.2],
    [-8.2],
    [7.33],
    [5.66]
])
y3 = np.array([-1, 1, -1, -1])
w3 = np.array([3.02])

X4 = np.random.random((1_000, 2))
X4[:, 0] = X4[:, 0] - 0.5
X4[:, 1] = X4[:, 1] - 0.5
y4 = (X4[:, 1] > 0) * 2 - 1
w4 = np.array([-1.0, 1.0])

# test scipy.sparse._csr.csr_matrix
X5 = np.random.random((1_000, 13))
y5 = X5[:, 1] > 0.76
X5 = sc.sparse._csr.csr_matrix(X5)
w5 = np.random.random(13)

test_binary = [
    (X1, y1, w1),
    (X2, y2, w2),
    (X3, y3, w3),
    (X4, y4, w4),
    (X5, y5, w5)
]
