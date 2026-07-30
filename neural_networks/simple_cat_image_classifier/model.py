import numpy as np

"""
for this binary classification, logistic regression will be implemented

***NOTES

X is the input where each column is a new sample, and each row is a feature in the current column sample
W will also be a (n, 1) array
"""

# define the Logistic Regression Class
class LogisticRegression:

    # initialize the parameters
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.costs = []

    # helper function to set w and b to default values
    def initialize_parameters(self, n_features):
        self.w = np.zeros((n_features, 1))
        self.b = 0.0

    # helper function to calculate the sigmoid function
    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))

    # the fit function that performs the forward and backward passes to find w and b
    def fit(self, X, y):

        # set basic information
        n_features, n_samples = X.shape
        # call the initialize function
        self.initialize_parameters(n_features)

        # iterate through the epochs to correct parameters
        for _ in range(self.epochs):
            # ========= forward pass =========
            # weighted sum
            z = np.dot(self.w.T, X) + self.b
            # sigmoid
            y_pred = self.sigmoid(z)

            # calculate the cost using BCE
            cost = -np.mean(y * np.log(y_pred) + (1-y) * np.log(y_pred))
            # check to see if it will be added to the costs attribute
            if _ % 100 == 0:
                self.costs.append(cost)

            # ======= backpropagation ========
            # calculate the gradients
            dw = 1/n_samples * np.dot(X, (y_pred-y).T)
            db = np.mean(y_pred-y)
            # update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    # write the predict function
    def predict(self, X):

        # ========= only forward pass =========
        z = np.dot(self.w.T, X) + self.b
        y_pred = self.sigmoid(z)
        y_pred.reshape(1, y_pred.shape[0])

        # interpret the values
        for i in range(y_pred.shape[1]):
            if y_pred[0, i] > 0.5:
                y_pred[0, i] = 1
            else:
                y_pred[0, i] = 0

        # return the value
        return y_pred

    # the get parameters function returns the w and b
    def get_parameters(self):
        return self.w, self.b
