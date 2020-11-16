import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_)


if __name__ == '__main__':
    # simple 1-d test
    np.random.seed(0)

    n_train = 500
    n_test = 50

    x_train = np.stack((np.random.rand(n_train), np.ones(n_train)), axis=1)
    y_train = np.stack((2 + 3 * x_train[:, 0] + np.random.rand(n_train),), axis=1)

    x_test = np.stack((np.random.rand(n_test), np.ones(n_test)), axis=1)
    y_test = np.stack((2 + 3 * x_test[:, 0] + np.random.rand(n_test),), axis=1)

    model = LinearRegressionUsingGD()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    plt.scatter(x_test[:, 0], y_test, s=10)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x_test[:, 0], y_pred, color='r')
    plt.show()
