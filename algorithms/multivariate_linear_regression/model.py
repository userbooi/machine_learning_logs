import numpy as np

'''
'lr' is the learning rate (the amount of the gradient that will be used to change the weight and bias)

formula for multivariate lienar regression:
    y = w1 * x1 + w2 * x2 + ... + wn * xn + b

where w is the weight (slope)
      b is the bias (intercept)

basic concept of linear regression:
    - finds the line of best fit from scattered points
    - updates the weight (w) and bias (b) until its the most accurate
        - uses the MSE (Mean Square Error) loss function
    - uses gradient descent to find the w and b that yields the minimum MSE

'''

class MultivariateLinearRegression:

    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            dw = 1/n_samples * np.dot(y_pred - y, X)
            db = (y_pred - y).mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

