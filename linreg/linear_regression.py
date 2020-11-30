from linreg.metrics import *
from linreg.optimizers import *
import copy
from tqdm.notebook import tqdm


class LinearRegression:
    def __init__(self, *args, l1_penalty: float = 0., l2_penalty: float = 0., verbose=2, **kwargs):
        self.w = None
        self.cache: list = []
        self.velocity: list = []
        self.l1_penalty: float = l1_penalty
        self.l2_penalty: float = l2_penalty
        self.loss_list: list = list()
        self.r2_list: list = list()
        self.optimizer = None
        self.verbose = verbose

    def _print_debug(self, msg, verbose):
        if verbose < self.verbose:
            print(msg)

    def predict(self, x, bias=True):
        assert self.w, f"w is None, use fit first!"
        x_train = copy.deepcopy(x)
        if bias:
            for i in range(len(x_train)):
                x_train[i].append(1)

        for i in range(len(x_train)):
            assert len(self.w) == len(x_train[i]), f"Column {i} has dim {len(x_train[i])} but {len(self.w)} required."

        return mult_matrix_vector(x_train, self.w)

    def fit(self, x, y, loss_func=mse, optimizer=SgdOptimizer, lr=0.01, iterations=200, bias=True, *args, **kwargs):
        x_train = copy.deepcopy(x)
        assert len(y) == len(x_train), f"X has dim {len(x_train)} but {len(y)} required."

        if bias:
            for i in range(len(x_train)):
                x_train[i].append(1)

        n, m = len(x), len(x[0])  # shapes
        if not self.w:  # init weights
            self.w = [1 for _ in range(m + 1)]

        self.optimizer = optimizer(self.w, *args, lr=lr,
                                   l1_penalty=self.l1_penalty,
                                   l2_penalty=self.l2_penalty,
                                   **kwargs)

        for it in tqdm(range(iterations)):
            y_pred = self.predict(x_train, bias=False)
            self.loss_list.append(loss_func(y, y_pred))
            self.r2_list.append(r2(y, y_pred))

            self.optimizer.fit_batch(x_train, y, y_pred)

            self._print_debug(f'iteration: {it}, loss: {self.loss_list[-1]:.6f}, r2: {self.r2_list[-1]:.6f}',
                              verbose=1)
