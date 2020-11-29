from linreg.matrix import *
from math import sqrt


def sign(x):
    if x < 0.:
        return -1.
    elif x == 0.:
        return 0.
    else:
        return 1.


class SgdOptimizer:
    def __init__(self, weights, *args, lr=0.01, l1_penalty=0., l2_penalty=0., **kwargs):
        self.lr = lr
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.weights = weights
        self.eps = 1e-8

    def _get_grad(self, x, y_true, y_pred):
        x_t = get_transposed(x)  # X.T
        v = [pred - truth for pred, truth in zip(y_pred, y_true)]
        g = mult_matrix_vector(x_t, v)  # g = X.T * v

        g = [el / len(x) for el in g]  # mean gradient

        # L1 and L2 regularisation
        return [gr + self.l1_penalty * sign(w) + self.l2_penalty * w for gr, w in zip(g, self.weights)]

    def fit_batch(self, x, y_true, y_pred):
        gradient = self._get_grad(x, y_true, y_pred)
        self._update_w(gradient)
        return self

    def _update_w(self, gradient):
        assert len(gradient) == len(self.weights),\
            f"Gradient and weights size must be equal, but {len(gradient)} != {len(self.weights)}"
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gradient[i]
        return self


class AdaGradOptimizer(SgdOptimizer):
    def __init__(self, weights, *args, lr=0.01, l1_penalty=0., l2_penalty=0., **kwargs):
        super().__init__(weights, *args, lr=lr, l1_penalty=l1_penalty, l2_penalty=l2_penalty, **kwargs)
        self.cum_grad = [0 for _ in range(len(weights))]

    def fit_batch(self, x, y_true, y_pred):
        gradient = self._get_grad(x, y_true, y_pred)

        for i in range(len(gradient)):
            self.cum_grad[i] += gradient[i] ** 2

        gradient = [g / sqrt(c + self.eps) for g, c in zip(gradient, self.cum_grad)]

        self._update_w(gradient)
        return self


class RmsPropOptimizer(SgdOptimizer):
    def __init__(self, weights, *args, lr=0.01, l1_penalty=0., l2_penalty=0., beta=0.9, **kwargs):
        super().__init__(weights, *args, lr=lr, l1_penalty=l1_penalty, l2_penalty=l2_penalty, **kwargs)
        self.cum_grad = [0 for _ in range(len(weights))]
        self.beta = beta

    def fit_batch(self, x, y_true, y_pred):
        gradient = self._get_grad(x, y_true, y_pred)

        for i in range(len(gradient)):
            self.cum_grad[i] = self.beta * self.cum_grad[i] + (1 - self.beta) * gradient[i] ** 2

        gradient = [g / sqrt(c + self.eps) for g, c in zip(gradient, self.cum_grad)]

        self._update_w(gradient)
        return self


class AdamOptimizer(SgdOptimizer):
    def __init__(self, weights, *args, lr=0.01, l1_penalty=0., l2_penalty=0., beta1=0.9, beta2=0.999, **kwargs):
        super().__init__(weights, *args, lr=lr, l1_penalty=l1_penalty, l2_penalty=l2_penalty, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.cum_grad = [0 for _ in range(len(weights))]
        self.moment = [0 for _ in range(len(weights))]

    def fit_batch(self, x, y_true, y_pred):
        gradient = self._get_grad(x, y_true, y_pred)

        for i in range(len(gradient)):
            self.moment[i] = self.beta1 * self.moment[i] + (1. - self.beta1) * gradient[i]
            self.cum_grad[i] = self.beta2 * self.cum_grad[i] + (1. - self.beta2) * gradient[i] ** 2

        gradient = [v / sqrt(c + self.eps) for v, c in zip(self.moment, self.cum_grad)]

        self._update_w(gradient)
        return self
