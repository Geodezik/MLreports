import numpy as np
from sklearn.model_selection import KFold as sklearn_kfold
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    extra_elems_counter = n % n_folds
    d_step = extra_elems_counter
    step = n // n_folds
    cv = list()
    for i in range(n_folds):
        val = np.arange(step + bool(extra_elems_counter)) + i * step + \
                                            (d_step - extra_elems_counter)
        train = np.concatenate((np.arange(val[0]), np.arange(val[-1] + 1, n)))
        if extra_elems_counter:
            extra_elems_counter -= 1
        cv.append((train, val))
    return cv


# based on sklearn version
def weighted_mode(a, w, axis=0):
    possible_classes = np.unique(a)
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


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    assert score in {"accuracy"}

    if cv is None:
        cv = kfold(len(X), n_folds=3)

    ans = dict()
    for i, fold in enumerate(cv):
        X_train, y_train = X[fold[0]], y[fold[0]]
        X_test, y_test = X[fold[1]], y[fold[1]]

        kwargs["k"] = max(k_list)
        clf = KNNClassifier(**kwargs)
        clf.fit(X_train, y_train)

        if kwargs['weights']:
            dist, indices = clf.find_kneighbors(X_test, return_distance=True)
            eps = float("1e-5")
            weights = 1 / (dist + eps)
        else:
            indices = clf.find_kneighbors(X_test, return_distance=False)
            weights = np.ones_like(indices)

        testlen = len(X_test)
        classes = y_train[indices]

        for k in k_list:
            preds = weighted_mode(classes[:, :k],
                                  weights[:, :k], axis=1)[0].reshape(-1)
            if k not in ans:
                ans[k] = np.zeros(len(cv))
            ans[k][i] = np.mean(preds == y_test)

    return ans


'''
n = 1234567
n_folds = 1000

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

X = np.repeat(X, 228, axis=0)
y = np.repeat(y, 228)

print(knn_cross_val_score(X, y,
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
"accuracy", None, strategy="my_own", metric="euclidean", weights=True))
#for elem1, elem2 in zip(kfold(n, n_folds), cv.split(X)):
    #print(elem1)
    #print(elem2)
    #assert np.all(elem1[1] == elem2[1])
    #assert np.all(elem1[0] == elem2[0])
'''
