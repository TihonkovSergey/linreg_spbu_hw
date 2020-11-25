import numpy as np


# Linear Regression
class LinearRegression:

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate

        self.iterations = iterations

    def fit(self, X, Y):
        # no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)

        # calculate gradients
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m

        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW

        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b


class RidgeRegression:

    def __init__(self, learning_rate, iterations, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty

    def fit(self, X, Y):
        #  no_of_training_examples, no_of_features
        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros(self.n)

        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)

        # calculate gradients
        dW = (- (2 * (self.X.T).dot(self.Y - Y_pred)) +
              (2 * self.l2_penalty * self.W)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b


class LassoRegression:

    def __init__(self, learning_rate, iterations, l1_penalty):

        self.learning_rate = learning_rate

        self.iterations = iterations

        self.l1_penalty = l1_penalty

    def fit(self, X, Y):
        #  no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # weight initialization

        self.W = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):

        Y_pred = self.predict(self.X)

        # calculate gradients
        dW = np.zeros(self.n)

        for j in range(self.n):

            if self.W[j] > 0:

                dW[j] = (- (2 * (self.X[:, j]).dot(self.Y - Y_pred))

                         + self.l1_penalty) / self.m

            else:

                dW[j] = (- (2 * (self.X[:, j]).dot(self.Y - Y_pred))

                         - self.l1_penalty) / self.m

        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW

        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )
    def predict(self, X):

        return X.dot(self.W) + self.b