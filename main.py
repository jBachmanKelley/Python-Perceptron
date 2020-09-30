# John Kelley, 09/29/2020, CMSC 409: Artificial Intelligence, Milos Manic #
import numpy as np


class Perceptron:
    # This method will train the perceptron: Max iterations are 5000, the error threshold should be passed in
    def train(self, x, y, max_iterations=5000):
        n_samples = x.shape[0]
        n_features = x.shape[1]

        # Add 1 for the bias term
        self.weights = np.zeros((n_features + 1))

        # Add column of 1s
        x = np.concatenate([x, np.ones((n_samples, 1))], axis=1)

        for i in range(max_iterations):
            for j in range(n_samples):
                if y[j] * np.dot(self.weights, x[j, :]) <= 0:
                    self.weights += y[j] * x[j, :]

    # This method will predict labels for passsed data and handle unlabeled data
    def label(self, X):
        if not hasattr(self, 'weights'):
            print("The data hasn't been trained yet")
            return

        n_samples = X.shape[10]
        # Add column of 1s
        X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)

        return y

    # This method calculates the accuracy of our model using______
    def assess(self, X, y):
        pred_y = self.label(X)

        return np.mean(y == pred_y)


