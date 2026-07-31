import numpy as np
from activation_functions import *

"""

This is the upgraded version of the simple cat image classifier that only used logistic regression

This will be a 5 layer Neural Network that runs ReLU for each hidden layer, and a Sigmoid for
the output layer

"""

# define the Neural Network
class NN:

    # initial function called upon instance creation
    def __init__(self, layer_dim, lr=0.001, epochs=2500):
        self.lr = lr
        self.epochs = epochs
        self.layers = len(layer_dim) - 1

        self.W = {}
        self.b = {}

    # initialize the parameters for each layer
    def init_params(self, layer_dim):

        for i in range(self.layers):
            self.W["W" + str(i+1)] = np.random.randn(layer_dim[i], layer_dim[i-1]) * 0.01
            self.b["b" + str(i+1)] = np.zeros((layer_dim[i], 1))

    def weighted_sum(self, A_prev, curr_w, curr_b):
        Z = np.dot(curr_w, A_prev) + curr_b

        return Z

    def activation(self, Z, function):
        pass