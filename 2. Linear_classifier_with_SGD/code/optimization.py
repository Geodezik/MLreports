from oracles import BinaryLogistic
from time import time
from utils import ClfEstimator
import numpy as np
import scipy as sc
import math


# Has mimimal functionality for passing tests, use MinibatchGD for
# more features
class GDClassifier:
    def __init__(
        self, loss_function="binary_logistic", step_alpha=1,
        step_beta=0, tolerance=1e-5, max_iter=1000, **kwargs
    ):
        if loss_function == "binary_logistic":
            self._oracle = BinaryLogistic(**kwargs)
        else:
            raise NotImplementedError

        self._alpha = step_alpha
        self._beta = step_beta
        self._tolerance = tolerance
        self._max_iter = max_iter
        self._w = None
        self.estimator = ClfEstimator()

    def _GD_step(self, X, y, epoch):
        grad = self._oracle.grad(X, y, self._w)
        n = self._alpha / (epoch ** self._beta)
        self._w = self._w - n * grad

    def fit(self, X, y, w_0=None, trace=False):
        if w_0 is None:
            self._w = np.random.random(X.shape[1]) * 2 - 1
        else:
            self._w = w_0

        func = self._oracle.func(X, y, self._w)
        if trace:
            history = {"time": [0.0], "func": [func]}

        epoch_start = 0.0
        for epoch in range(1, self._max_iter + 1):
            epoch_start = time()
            self._GD_step(X, y, epoch)
            func = self._oracle.func(X, y, self._w)

            if trace:
                history["time"].append(time() - epoch_start)
                history["func"].append(func)

            if (epoch > 1) and (abs(func - func_prev) < self._tolerance):
                break
            func_prev = func

        if trace:
            return history

    def predict(self, X):
        if self._w is None:
            return RuntimeError("Fit method should be called before predicting")
        return np.sign(X @ self._w.reshape(-1, 1))

    def predict_proba(self, X):
        if self._w is None:
            return RuntimeError("Fit method should be called before predicting")
        if isinstance(X, sc.sparse._csr.csr_matrix):
            X = X.toarray()
        pred = X @ self._w.reshape(-1, 1)
        pred[pred < -100.0] = -100.0
        return 1 / (1 + np.exp(-pred))

    def get_objective(self, X, y):
        if self._w is None:
            return RuntimeError("Weight vector should be initialized in fit method before calling this method")
        return self._oracle.func(X, y, self._w)

    def get_gradient(self, X, y):
        if self._w is None:
            return RuntimeError("Weight vector should be initialized in fit method before calling this method")
        return self._oracle.grad(X, y, self.w)

    def get_weights(self):
        if self._w is None:
            return RuntimeError("Weight vector should be initialized in fit method before calling this method")
        return self._w

    @property
    def metric(self):
        return self.estimator.metric


class MinibatchGDClassifier(GDClassifier):
    def __init__(
        self, loss_function="binary_logistic", batch_size=1, step_alpha=1,
        step_beta=0, tolerance=1e-6, max_iter=1000, random_seed=42, **kwargs
    ):
        super().__init__(loss_function, step_alpha, step_beta, tolerance,
                         max_iter, **kwargs)
        self._seed = random_seed
        self.batch_size = batch_size
        self._cur_pos = 0
        self._seen_objects = 0
        self._epoch = 0
        self.random_indices = None

    def _resample(self, length):
        self.random_indices = np.arange(length)
        np.random.shuffle(self.random_indices)

    def _GD_step(self, X, y, iteration):
        start = self._cur_pos
        stop = self._cur_pos + self.batch_size
        grad = self._oracle.grad(X[start:stop],
                                 y[start:stop],
                                 self._w)
        n = self._alpha / (iteration ** self._beta)
        self._w = self._w - n * grad
        self._cur_pos += self.batch_size
        if self._cur_pos >= X.shape[0]:
            self._cur_pos = 0
            self._epoch += 1
            self._resample(X.shape[0])
        self._seen_objects += self.batch_size

    def fit(self, X, y, X_val=None, y_val=None, val_split=None, w_0=None, trace=False,
            show_progress=True, train_subset_size=0.0, log_freq=1.0):
        # log freq approximately shows how many times you'll iterate through the
        # whole set before updating
        if isinstance(X, sc.sparse._csr.csr_matrix):
            X = X.toarray()
        if (val_split is not None):
            if (X_val is not None) or (y_val is not None):
                print("Warning: X, y were splitted to replace passed X_val and y_val")
            to = round(X.shape[0] * val_split)
            X_val = X[:to]
            y_val = y[:to]
            X = X[to:]
            y = y[to:]
        else:
            to = 0
        if ((X_val is None) and not (y_val is None)) or \
           ((y_val is None) and not (X_val is None)):
            raise RuntimeError("One of validation set's parts wasn't passed into fit")
        elif (X_val is not None) and (y_val is not None):
            validation_flag = True
        else:
            validation_flag = False
        if self.batch_size == "full":
            self.batch_size = X.shape[0] - to
        np.random.seed(self._seed)
        self._resample(X.shape[0])

        if w_0 is None:
            self._w = np.random.random(X.shape[1]) * 2 - 1
        else:
            self._w = w_0

        if trace:
            func = self._oracle.func(X, y, self._w)
            old_weights = self._w.copy()
            history = {"time": [0.0], "func": [func], "epoch_num": [0.0],
                       "weights_diff": [0.0]}
            if validation_flag:
                val_loss = self._oracle.func(X_val, y_val, self._w)
                history["val_func"] = [val_loss]
                val_accuracy = self.metric["accuracy"](y_val, self.predict(X_val))
                history["val_accuracy"] = [val_accuracy]
            if train_subset_size:
                X_subset = X[:round(X.shape[0] * train_subset_size)]
                y_subset = y[:round(X.shape[0] * train_subset_size)]
                train_accuracy = self.metric["accuracy"](y_subset, self.predict(X_subset))
                history["train_accuracy"] = [train_accuracy]
            if show_progress:
                if validation_flag:
                    print(f"Epoch 0.0 (iteration 0), train {train_accuracy}, validation {val_accuracy}")
                else:
                    print(f"Epoch 0.0 (iteration 0), train {train_accuracy}")

        start = time()
        for iteration in range(1, self._max_iter + 1):
            self._GD_step(X, y, iteration)
            func = self._oracle.func(X, y, self._w)

            if trace and (self._seen_objects / X.shape[0] >= log_freq):
                history["time"].append(time() - start)
                history["func"].append(func)
                history["epoch_num"].append(round(history["epoch_num"][-1] +
                                            self._seen_objects / X.shape[0], 2))
                history["weights_diff"].append(np.linalg.norm(self._w - old_weights))
                if train_subset_size:
                    train_accuracy = self.metric["accuracy"](y_subset, self.predict(X_subset))
                    history["train_accuracy"].append(train_accuracy)
                if validation_flag:
                    val_loss = self._oracle.func(X_val, y_val, self._w)
                    history["val_func"].append(val_loss)
                    val_accuracy = self.metric["accuracy"](y_val, self.predict(X_val))
                    history["val_accuracy"].append(val_accuracy)
                if show_progress:
                    approx_epoch = history["epoch_num"][-1]
                    if validation_flag:
                        print(f"Epoch {approx_epoch} (iteration {iteration}), train {train_accuracy}, validation {val_accuracy}")
                    else:
                        print(f"Epoch {approx_epoch} (iteration {iteration}), train {train_accuracy}")
                old_weights = self._w.copy()
                self._seen_objects = 0
                start = time()

            if (iteration > 1) and (abs(func - func_prev) < self._tolerance):
                print("Tolerance threshold was hit")
                break
            func_prev = func
        else:
            print("Maximal iteration was reached")

        if trace:
            return history
