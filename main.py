# John Kelley, 09/29/2020, CMSC 409: Artificial Intelligence, Milos Manic #
import csv
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Perceptron:
    # Class Datafields: None Needed At The Moment

    # This method initializes the instance, loads the CSV, and runs the perceptron
    def __init__(self):
        # Import the data and separate it into input and real_labels
        self.data = self.read_CSV("GroupA.csv")
        self.real_label = self.data[:, [2]]

        # Normalize the data and set a learning rate (alpha)
        self.X = self.data[:, [0, 1]]
        self.X = self.X / np.linalg.norm(self.X)
        self.alpha = 0.1

        # Train the neuron, confirm using real_labels, display
        self.train()
        self.display(self.label())

    # This method will train the perceptron: Max iterations are 5000, the error threshold should be passed in(NOT DONE YET)
    def train(self, testPercentage = 0.75, max_iterations=5000):

        numOfSamples = self.X.shape[0]
        numOfFeatures = self.X.shape[1]

        # Add 1 weight for the offset term
        self.weights = np.zeros((numOfFeatures + 1))

        # Initialize neuron offset between -0.5 and 0.5 for the 2-D array X
        self.X = np.concatenate([self.X, np.random.rand(numOfSamples, 1) - 0.5], axis=1)

        # Create a random array of row-numbers for access order
        accessOrder = []
        for x in range(numOfSamples):
            temp = 0
            while temp in accessOrder:
                temp = random.randint(0, numOfSamples - 1)
            accessOrder.append(temp)

        for i in range(max_iterations):
            # Initialize T.E to 0
            totalError = 0

            # Access the input data using the randomly ordered, unique values of accessOrder, calc sum and adjust weight
            for j in range(numOfSamples):
                index = accessOrder[j]
                # The dot product detects if there is a difference and thus a need to update weights,
                s = np.dot(self.X[index, :], self.weights)
                self.weights += self.X[index, :] * self.alpha * (self.real_label[index] - np.sign(s))
                totalError += (self.real_label[index] - np.sign(s)) * (self.real_label[index] - np.sign(s))
            # This is where we can have a convergence if-statement which returns
            if totalError < np.power(10.0, 1):
                print("Convergence")
                return

        print(self.weights)

    # This method will predict labels for passsed data and handle unlabeled data
    def label(self):
        if not hasattr(self, 'weights'):
            print("The data hasn't been trained yet")
            return

        # Activation function
        #y = 1/(1 + np.exp(-1 * self.alpha * (np.dot(self.X[:, [1, 2]], self.weights[1:]) + self.weights[0]))) Soft Threshold
        y = np.sign(np.dot(self.X, self.weights))

        # Adjust label vector to 0 or 1
        for i in range(y.shape[0]):
            if y[i] >= 1:
                y[i] = 1
            else:
                y[i] = 0

        return y

    # This method calculates the accuracy of our model using Total Error NEED TO CHANGE TO TOTAL ERROR
    def assess(self, X, y):
        pred_y = self.label(X)

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
