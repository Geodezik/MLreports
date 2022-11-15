import numpy as np
from distances import euclidean_distance, cosine_distance
import sklearn.neighbors


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size=1000):
        self.k = k

        assert strategy in {"my_own", "brute", "kd_tree", "ball_tree"}
        assert metric in {"euclidean", "cosine"}
        if metric == "euclidean":
            self.metric = euclidean_distance
        else:
            self.metric = cosine_distance
        if strategy != "my_own":
            self.knn = sklearn.neighbors.NearestNeighbors(
                n_neighbors=k,
                algorithm=strategy,
                metric=metric
                )
        else:
            self.knn = None

        self.weights = weights
        if test_block_size:
            self.test_block_size = test_block_size
        else:
            self.test_block_size = None
        self.X = None
        self.y = None
        self.unique_classes = None

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        self.y = y
        if self.knn:
            self.knn.fit(X)
        else:
            self.X = X

    def find_kneighbors(self, X, return_distance=False):
        if self.knn:
            return self.knn.kneighbors(X, return_distance=return_distance)
        else:
            distances = self.metric(X, self.X)
            indices = distances.argpartition(self.k)[:, :self.k]
            sort_ind = np.take_along_axis(distances,
                                          indices, axis=1).argsort(axis=1)
            indices = np.take_along_axis(indices, sort_ind, axis=1)
            if return_distance:
                sort_dist = np.take_along_axis(distances, indices, axis=1)
                return sort_dist, indices
            else:
                return indices

    # based on sklearn version
    def weighted_mode(self, a, w, possible_classes, axis=0):
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)
        for c in possible_classes:
            template = np.zeros(a.shape)
            ind = (a == c)
            template[ind] = w[ind]
            counts = np.expand_dims(np.sum(template, axis), axis)
            mostfrequent = np.where(counts > oldcounts, c, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent
        return mostfrequent, oldcounts

    def predict(self, X):
        answers = list()
        tbs = self.test_block_size if self.test_block_size else len(X)
        for i in range(0, len(X), tbs):
            if self.weights:
                distances, nn = self.find_kneighbors(X[i:(i + tbs)],
                                                     return_distance=True)
                eps = float("1e-5")
                weights = 1 / (distances + eps)
            else:
                nn = self.find_kneighbors(X[i:(i + tbs)],
                                          return_distance=False)
                weights = np.ones_like(nn)

            nn_classes = self.y[nn]
            block_ans = self.weighted_mode(nn_classes, weights,
                                           self.unique_classes, axis=1)[0]
            answers.append(block_ans)
        return np.concatenate(answers).ravel()


'''
X = np.array([
    [1.0, 1.12],
    [2.12, 5.43],
    [0.91, 2.11],
    [2.42, 0.99],
    [0.67, 1.11],
    [1.11, 5.67],
    [0.11, 3.21],
    [2.01, 5.02],

    [-1.5, 1.8],
    [-1.1, 0.99],
    [-3.56, 3.01],
    [-2.67, 2.55],
    [-1.88, 0.67],
    [-5.0, 6.66],
    [-6.11, 1.1],

    [-1.21, -1.0],
    [-1.0, -1.0],
    [-2.4, -1.22],
    [-1.0, -4.12],
    [-0.23, -0.99],
    [-2.33, -.2],

    [1.21, -0.89],
    [2.78, -1.77],
    [3.0, -4.01],
    [2.6, -2.55],
    [1.88, -1.6],
    [1.04, -3.6],
    [3.19, -3.11]
])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
             2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4])

Z = np.repeat(np.array([[1, 0.97], [-1, -0.11], [-12, -6]]), 1, axis=0)
clf = KNNClassifier(
    k=5,
    strategy="my_own",
    metric="euclidean",
    weights=True,
    test_block_size=None
    )

clf.fit(X, y)
print(clf.predict(X))
#clf.predict(Z)
#print(clf.predict(Z))
'''
