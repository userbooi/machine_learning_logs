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
    def __init__(self, layer_dim, lr=0.001, epochs=2500, lambd=None, keep_prob=None):
        self.lr = lr
        self.epochs = epochs
        self.layers = len(layer_dim) - 1
        self.layer_dim = layer_dim
        self.lambd=lambd
        self.keep_prob = keep_prob

        self.W = {}
        self.b = {}

        self.init_params()

        self.epsilon = 1e-15

    # initialize the parameters for each layer
    def init_params(self):

        # use He initialization for the ReLU activation layers
        for i in range(self.layers-1):
            self.W[f"W{i+1}"] = np.random.randn(self.layer_dim[i + 1], self.layer_dim[i]) * np.sqrt(2 / self.layer_dim[i])
            self.b[f"b{i+1}"] = np.zeros((self.layer_dim[i+1], 1))

        # use Xavier initialization for the sigmoid output layer
        self.W[f"W{self.layers}"] = np.random.randn(self.layer_dim[self.layers], self.layer_dim[self.layers-1]) * np.sqrt(1 / self.layer_dim[self.layers-1])
        self.b[f"b{self.layers}"] = np.zeros((self.layer_dim[self.layers], 1))


    # helper function to only compute the weighted sums
    def weighted_sum(self, A_prev, curr_w, curr_b):
        Z = np.dot(curr_w, A_prev) + curr_b

        return Z

    # helper function to compute the activation function
    def activation(self, Z, function):
        A = None

        if function == "relu":
            A = relu(Z)
        elif function == "sigmoid":
            A = sigmoid(Z)

        return A

    # helper function to compute the entire forward propagation
    def forward(self, X):

        A_prev = X
        caches = []

        # loop through each layer
        for l in range(self.layers-1):

            # calculate the weighted sum
            Z = self.weighted_sum(A_prev, self.W[f"W{l+1}"], self.b[f"b{l+1}"])
            A = self.activation(Z, "relu")

            # add the Z and A to the cache
            caches.append((Z, A_prev))
            # set the A for the next layer
            A_prev = A

        # calculate the sigomid for the output layer
        Z = self.weighted_sum(A_prev, self.W[f"W{self.layers}"], self.b[f"b{self.layers}"])
        AL = self.activation(Z, "sigmoid")

        # add the final cache
        caches.append((Z, A_prev))

        return AL, caches

    # calculate the cost function - using BCE Cost function
    def calc_cost(self, AL, y, m):

        log_part = np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL))
        cost = -1. / m * np.nansum(log_part)

        return np.squeeze(cost)

    def calc_cost_L2_regularized(self, AL, y, m):

        log_part = np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL))
        cost = -1. / m * np.nansum(log_part)

        return np.squeeze(cost)

    # helper function to calculate the gradients for the parameters
    def gradient_parameters(self, A_prev, dZ, curr_w, m):
        dw = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(curr_w.T, dZ)

        return dA_prev, dw, db

    # helper function to calculate the gradient for the activation function
    def gradient_activation(self, dA, curr_w, m, cache, function):
        dZ = None
        Z, A_prev = cache

        if function == "relu":
            dZ = relu_gradients(dA, Z)
        elif function == "sigmoid":
            dZ = sigmoid_gradients(dA, Z)

        dA_prev, dw, db = self.gradient_parameters(A_prev, dZ, curr_w, m)

        return dA_prev, dw, db

    # helper function to calculate the entire backward propagation
    def backward(self, AL, caches, y, m):

        # initialize the gradient cache
        grads = {}
        y = y.reshape(AL.shape)
        # clip the AL
        AL = np.clip(AL, self.epsilon, 1-self.epsilon)

        # calculate the gradient from the BCE Loss
        dAL = -(np.divide(y, AL) - np.divide(1.-y, 1.-AL))
        # put it into the sigmoid output layer gradient calculation
        grads[f"dA{self.layers-1}"], grads[f"dW{self.layers}"], grads[f"db{self.layers}"] = self.gradient_activation(
            dAL, self.W[f"W{self.layers}"], m, caches[self.layers-1], "sigmoid"
        )

        # loop through all the hidden RELU layers backwards
        for l in range(self.layers-1, 0, -1):

            # calculate the gradients from the activation
            grads[f"dA{l-1}"], grads[f"dW{l}"], grads[f"db{l}"] = self.gradient_activation(
                grads[f"dA{l}"], self.W[f"W{l}"], m, caches[l-1], "relu"
            )

        return grads

    # helper function to update the parameters
    def update_parameters(self, grads):

        # loop through all the gradients and update
        for l in range(self.layers):
            self.W[f"W{l+1}"] -= self.lr * grads[f"dW{l+1}"]
            self.b[f"b{l+1}"] -= self.lr * grads[f"db{l+1}"]

    # the fit function that trains the model
    def fit(self, X, y, print_cost=False):

        costs = []
        m = y.shape[1]

        # loop through all the iterations
        for _ in range(self.epochs):

            # calculate the forward pass
            AL, caches = self.forward(X)

            # calculate the cost
            cost = self.calc_cost(AL, y, m)
            if _ % 100 == 0:
                costs.append(cost)
                if print_cost:
                    print(cost)

            # calculate the backward pass
            grads = self.backward(AL, caches, y, m)

            # update the parameters
            self.update_parameters(grads)

        return costs

    # the function to predict the test data
    def predict(self, X):
        # forward pass
        AL, caches = self.forward(X)

        # convert the answer to 0/1
        for i in range(AL.shape[1]):
            if AL[0, i] > 0.5:
                AL[0, i] = 1
            else:
                AL[0, i] = 0

        return AL
