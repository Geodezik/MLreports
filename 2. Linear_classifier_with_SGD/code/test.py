import numpy as np
from tests import *
from utils import grad_finite_diff
from utils import PCA
from oracles import BinaryLogistic
from optimization import *

# test grad_finite diff
eps = 1e-6
X = 0
y = 0
w = np.ones(4)
print("Testing grad_finite_diff...")
for F, ans in zip(test_functions, ans_functions):
    res = grad_finite_diff(F, X, y, w)
    assert np.linalg.norm(res - ans) < eps
print("Tests passed!\n")

# test binary_logistic grad
print("Testing binary_logistic_grad (np.array and scipy.sparse._csr.csr_matrix)...")
oracle = BinaryLogistic(1)
for X, y, w in test_binary:
    true = grad_finite_diff(oracle.func, X, y, w)
    implemented = oracle.grad(X, y, w)
    assert np.linalg.norm(true - implemented) < eps
print("Tests passed!\n")

# test SGD
clf = MinibatchGDClassifier(loss_function='binary_logistic', batch_size = 5000, l2_coef=0.1)
print("Testing SGD (validation set accuracy)...")
for i in range(5):
    X = np.random.random((100_000, 2))
    X[:, 0] = X[:, 0] - 0.5
    X[:, 1] = X[:, 1] - 0.5
    y = (X[:, 1] > 0) * 2 - 1

    hist = clf.fit(X, y, val_split=0.2, show_progress=False,
                   trace=True, log_freq=0.5, train_subset_size=0.25)

    assert hist["train_accuracy"][0] < hist["train_accuracy"][-1]
    assert hist["val_accuracy"][0] < hist["val_accuracy"][-1]

    print("Train accuracy", *hist["train_accuracy"])
    print("Validation accuracy", *hist["val_accuracy"])
    print()
print("Accuracy is increasing!\n")

# test PCA
print("Testing PCA...")
pca = PCA()
pca.fit_transform(np.array([[0.5, 1], [0, 0]]))
print("PCA test passed!")
