import numpy as np


def euclidean_distance(X, Y):
    return np.sqrt(np.linalg.norm(X, axis=-1)[:, None] ** 2 +
                   np.linalg.norm(Y, axis=-1)[None, :] ** 2 - (2.0 * X) @ Y.T)


def cosine_distance(X, Y):
    return 1 - (X @ Y.T) / (np.linalg.norm(X, axis=-1)[:, None] *
                            np.linalg.norm(Y, axis=-1)[None, :])
