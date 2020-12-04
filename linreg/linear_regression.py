from linreg.metrics import *
from linreg.optimizers import *
import copy
from tqdm.notebook import tqdm


class LinearRegression:
    def __init__(self, optimizer: SgdOptimizer, l1_penalty: float = 0., l2_penalty: float = 0., loss=mse, verbose=2):
        self.w = None
        self.l1_penalty: float = l1_penalty
        self.l2_penalty: float = l2_penalty
        self.loss = loss
        self.loss_list: list = list()
        self.r2_list: list = list()
        self.optimizer = optimizer
        self.verbose = verbose

    def _print_debug(self, msg, verbose):
        if verbose < self.verbose:
            print(msg)

    def reset(self):
        self.loss_list = list()
        self.r2_list = list()
        if self.w:
            self.w = [1 for _ in self.w]

    def predict(self, x, bias=True):
        assert self.w, f"w is None, use fit first!"
        x_train = copy.deepcopy(x)
        if bias:
            for i in range(len(x_train)):
                x_train[i].append(1)

        for i in range(len(x_train)):
            assert len(self.w) == len(x_train[i]), f"Column {i} has dim {len(x_train[i])} but {len(self.w)} required."

        return mult_matrix_vector(x_train, self.w)

    def fit(self, x, y, iterations=200, bias=True):
        x_train = copy.deepcopy(x)
        assert len(y) == len(x_train), f"X has dim {len(x_train)} but {len(y)} required."

        if bias:
            for i in range(len(x_train)):
                x_train[i].append(1)

        n, m = len(x), len(x[0])  # shapes
        if not self.w:  # init weights
            self.w = [1 for _ in range(m + 1)]

        self.optimizer.set_weights(self.w)

        for it in tqdm(range(iterations)):
            y_pred = self.predict(x_train, bias=False)
            self.loss_list.append(self.loss(y, y_pred))
            self.r2_list.append(r2(y, y_pred))

            self.optimizer.fit_batch(x_train, y, y_pred)

            self._print_debug(f'Iteration: {it}, loss: {self.loss_list[-1]:.6f}, r2: {self.r2_list[-1]:.6f}',
                              verbose=1)
