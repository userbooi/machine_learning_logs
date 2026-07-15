import numpy as np

'''
'a' is the learning rate (the amount of the gradient that will be used to change the weight and bias)

formula for linear regression: y_pred = w * x + b
where w is the weight (slope)
      b is the bias (intercept)
      
basic concept of linear regression:
    - finds the line of best fit from scattered points
    - updates the weight (w) and bias (b) until its the most accurate
        - uses the MSE (Mean Square Error) loss function
    - uses gradient descent to find the w and b that yields the minimum MSE

'''

class LinearRegression:

    def __init__(self, a=0.001, epochs=1000):
        self.a = a
        self.epochs = epochs
        self.w = 0
        self.b = 0

        self.output_default()

    # training function that updates the default parameters into the correct ones based on the
    # features (X) and targets (y)
    def fit(self, X, y):

        for _ in range(self.epochs):
            preds = self.w * X.ravel() + self.b
            dw = ((preds-y) * X.ravel()).mean()
            db = (preds-y).mean()
            self.w -= self.a * dw
            self.b -= self.b * db

    # predict values of the tests
    def predict(self, X):
        preds = self.w * X.ravel() + self.b
        return preds

    def output_default(self):
        print((f"information regarding default model:\n"
              f"\tmodel name\t\t: Linear Regression\n"
              f"\tlearning rate (a)\t: {self.a}\n"
              f"\tweight\t\t\t: {self.w}\n"
              f"\tbias\t\t\t: {self.b}\n"
              f"\tepochs\t\t\t: {self.epochs}").title())

