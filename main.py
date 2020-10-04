# John Kelley, 09/29/2020, CMSC 409: Artificial Intelligence, Milos Manic #
import csv
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Perceptron:
    # Class Datafields: None Needed At The Moment

    # This method initializes the instance, loads the CSV, and runs the perceptron
    def __init__(self, target, percent, epsilon):
        # Configuration Settings
        self.target = target
        self.percentTrain = percent
        self.epsilon = epsilon

        # Import the data and separate it into input and real_labels
        self.data = self.read_CSV(f"{self.target}.csv")
        self.real_label = self.data[:, [2]]

        # Normalize the data and set a learning rate (alpha)
        self.X0 = self.data[:, [0]]
        self.X1 = self.data[:, [1]]
        for x in range(self.X0.shape[0]):
            # value - min divide by range
            self.X0[x, 0] = (self.X0[x, 0] - np.min(self.X0)) / (np.max(self.X0) - np.min(self.X0))
            self.X1[x, 0] = (self.X1[x, 0] - np.min(self.X1)) / (np.max(self.X1) - np.min(self.X1))
        self.X = np.concatenate([self.X0, self.X1], axis=1)

        self.alpha = 0.05

        # Train the neuron, confirm using real_labels, display
        self.train()
        self.display(self.label())

    # This method will train the perceptron: Max iterations are 5000, the error threshold should be passed in(NOT DONE YET)
    def train(self, max_iterations=500):

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
            for j in range(round(numOfSamples * self.percentTrain)):
                index = accessOrder[j]
                # The dot product gives the 'net',
                net = np.dot(self.X[index, :], self.weights)

                # HARD
                # Activation function fires if net > threshold, threshold = -bias
                # threshold = -1 * self.weights[2] + 1
                # if (net > threshold):
                #    out = 1
                # else:
                #     out = 0

                # SOFT
                # Activation function fires if out > 0.5
                out = self.sigmoid(net, 1, 0)

                self.weights += self.X[index, :] * self.alpha * (self.real_label[index] - np.tanh(out))
                totalError += (self.real_label[index] -  np.tanh(out)) * (self.real_label[index] -  np.tanh(out))
                # print(f"\nNet is: {net}\nThreshold is: {threshold}\nOut is : {out}\nTotal Error is: {totalError}")
            # This is where we can have a convergence if-statement which returns
            if totalError < np.power(10.0, self.epsilon):
                print("Convergence")
                return

        print(self.weights)

    # This method will predict labels for passsed data and handle unlabeled data
    def label(self):
        if not hasattr(self, 'weights'):
            print("The data hasn't been trained yet")
            return

        net = np.dot(self.X, self.weights)
        # threshold = -1 * self.weights[2] + 1

        # Hard Activation function
        # for i in range(net.shape[0]):
        #    if net[i] > threshold:
        #        net[i] = 1
        #        self.X[i, 2] = 1
        #    else:
        #        net[i] = 0
        #        self.X[i, 2] = 0

        # Soft Activation Function
        for i in range(net.shape[0]):
            if  self.sigmoid(net[i], 1, 0) > 0.5:
                net[i] = 1
                self.X[i, 2] = 1
            else:
                net[i] = 0
                self.X[i, 2] = 0

        return net

    # This method imports the CSV and creates a numpy data structure
    def read_CSV(self, target):
        print("Started CSV")
        return np.genfromtxt(target, delimiter=",")

    # This method displays the results in plot form
    def display(self, y):
        # Plot points
        for i in range(y.shape[0]):
            if i != 0:
                if y[i] == 0:
                    plt.scatter(self.X[i, 0], self.X[i, 1], c='b')  # Small is Blue
                else:
                    plt.scatter(self.X[i, 0], self.X[i, 1], c='g')  # Big is Green

        np.savetxt(f"{self.target}_{self.percentTrain}_Result_Soft.csv", self.X, delimiter=',')
        np.savetxt(f"{self.target}_{self.percentTrain}_Weights_Soft.csv", self.weights, delimiter=',')
        plt.show()

    # This method performs the sigmoid
    def sigmoid(self, net, k, b):
        return 1 / (1 + np.exp(-(k * net + b)))


# Main Script
# perc1 = Perceptron("GroupA", 0.75, -5)
# perc2 = Perceptron("GroupA", 0.25, -5)
# perc3 = Perceptron("GroupB", 0.75, 2)
# perc4 = Perceptron("GroupB", 0.25, 2)
# perc5 = Perceptron("GroupC", 0.75, 2)
# perc6 = Perceptron("GroupC", 0.25, 2)
