# John Kelley, 09/29/2020, CMSC 409: Artificial Intelligence, Milos Manic #
import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Perceptron:
    # Class Datafields: None Needed At The Moment

    # This method initializes the instance, loads the CSV, and runs the perceptron
    def __init__(self):
        self.data = self.read_CSV("GroupB.csv")
        self.X = self.data[:, [0, 1]]
        self.real_label = self.data[:, [2]]
        for i in range(self.real_label.shape[0]):
            if self.real_label[i] == 0:
                self.real_label[i] = -1
        print("Finished CSV")
        self.train(0.1)
        self.display(self.label())

    # This method will train the perceptron: Max iterations are 5000, the error threshold should be passed in
    def train(self, alpha, max_iterations=5000):

        # Alhpe value CORRECT THIS LATTER

        numOfSamples = self.X.shape[0]
        numOfFeatures = self.X.shape[1]

        # Add 1 term for the offset term
        self.weights = np.zeros((numOfFeatures + 1))

        # Add column of 1s to the 2-D array X
        self.X = np.concatenate([self.X, np.ones((numOfSamples, 1))], axis=1)

        # NEED TO DO: Implement a way to do a ranom % of entries
        for i in range(max_iterations):
            for j in range(numOfSamples):
                # The dot product detects if there is a difference and thus a need to update weights,
                # By multiplying by label, we determine if pred_label is not actual label
                if self.real_label[j] * np.dot(self.weights, self.X[j, :]) <= 0:  # The labels are 0,1 so we need to adjust
                    self.weights += alpha * (self.real_label[j] * self.X[j, :])

    # This method will predict labels for passsed data and handle unlabeled data
    def label(self):
        if not hasattr(self, 'weights'):
            print("The data hasn't been trained yet")
            return
        self.X = self.X[:, [0, 1]]
        numOfSamples = self.X.shape[0]
        # Add column of 1s
        self.X = np.concatenate([self.X, np.ones((numOfSamples, 1))], axis=1)
        y = np.matmul(self.X, self.weights)

        # Adjust label vector
        y = np.vectorize(lambda val: 1 if val > 0 else 0)(y)

        return y

    # This method calculates the accuracy of our model using Total Error NEED TO CHANGE TO TOTAL ERROR
    def assess(self, X, y):
        pred_y = self.label(X)

        # Check for Episilon value and set max_iterations to 500 if triggered

        return np.mean(y == pred_y)

    # This method imports the CSV and creates a numpy data structure
    def read_CSV(self, target):
        print("Started CSV")
        return np.genfromtxt(target, delimiter=",", skip_header=1)

    # This method displays the results in plot form
    def display(self, y):
        #col = np.where(self.y[:] == 0, 'k', 'r')
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y)
        plt.show()


# Main Script
perc = Perceptron()
