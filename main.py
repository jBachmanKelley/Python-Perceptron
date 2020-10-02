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
        self.X0 = self.data[:, [0]]
        self.X1 = self.data[:, [1]]
        for x in range(self.X0.shape[0]):
            # value - min divide by range
            self.X0[x, 0] = (self.X0[x, 0] - np.min(self.X0)) / (np.max(self.X0) - np.min(self.X0))
            self.X1[x, 0] = (self.X1[x, 0] - np.min(self.X1)) / (np.max(self.X1) - np.min(self.X1))
        self.X = np.concatenate([self.X0, self.X1], axis=1)

        self.alpha = 0.01

        # Train the neuron, confirm using real_labels, display
        self.train()
        self.display(self.label())

    # This method will train the perceptron: Max iterations are 5000, the error threshold should be passed in(NOT DONE YET)
    def train(self, testPercentage = 0.75, max_iterations=100):

        numOfSamples = self.X.shape[0]
        numOfFeatures = self.X.shape[1]

        # Add 1 weight for the offset term
        self.weights = np.ones((numOfFeatures + 1))

        # Initialize neuron offset between -0.5 and 0.5 for the 2-D array X
        self.X = np.concatenate([self.X, np.random.rand(numOfSamples, 1) - 0.5], axis=1)

        # Create a random array of row-numbers for access order
        accessOrder = []
        for x in range(numOfSamples):
            temp = random.randint(0, numOfSamples - 1)
            while temp in accessOrder:
                temp = random.randint(0, numOfSamples - 1)
            accessOrder.append(temp)

        for i in range(max_iterations):
            # Initialize T.E to 0
            totalError = 0

            # Access the input data using the randomly ordered, unique values of accessOrder, calc sum and adjust weight
            for j in range(numOfSamples):
                index = accessOrder[j]
                # The dot product gives the 'net',
                net = np.dot(self.X[index, :], self.weights)

                # Activation function fires if net > threshold, threshold = -bias
                threshold = -1 * self.weights[2] + 1
                if (net > threshold):
                    out = 1
                else:
                    out = 0
                self.weights += self.X[index, :] * self.alpha * (self.real_label[index] - out)
                totalError += (self.real_label[index] - out) * (self.real_label[index] - out)
                # print(f"\nNet is: {net}\nThreshold is: {threshold}\nOut is : {out}\nTotal Error is: {totalError}")
            # This is where we can have a convergence if-statement which returns
            if totalError < np.power(10.0, -5):
                print("Convergence")
                return

        print(self.weights)

    # This method will predict labels for passsed data and handle unlabeled data
    def label(self):
        if not hasattr(self, 'weights'):
            print("The data hasn't been trained yet")
            return

        # Activation function
        # y = 1/(1 + np.exp(-1 * self.alpha * (np.dot(self.X[:, [1, 2]], self.weights[1:]) + self.weights[0]))) Soft Threshold
        out = np.dot(self.X, self.weights)
        threshold = -1 * self.weights[2] + 1

        # Activation function
        for i in range(out.shape[0]):
            if out[i] > threshold:
                out[i] = 1
            else:
                out[i] = 0

        return out

    # This method calculates the accuracy of our model using Total Error NEED TO CHANGE TO TOTAL ERROR
    def assess(self, X, y):
        pred_y = self.label(X)

        return np.mean(y == pred_y)

    # This method imports the CSV and creates a numpy data structure
    def read_CSV(self, target):
        print("Started CSV")
        return np.genfromtxt(target, delimiter=",", skip_header=2)

    # This method displays the results in plot form
    def display(self, y):
        # Plot points
        for i in range(y.shape[0]):
            if i != 0:
                if y[i] == 0:
                    plt.scatter(self.X[i, 0], self.X[i, 1], c='b') # Small is Blue
                else:
                    plt.scatter(self.X[i, 0], self.X[i, 1], c='g') # Big is Green
        # Plot line
        # Plt.Line2D
        np.savetxt("GroupAResult.csv", self.X, delimiter=',')
        plt.show()


# Main Script
perc = Perceptron()
