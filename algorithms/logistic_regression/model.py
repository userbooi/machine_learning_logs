import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def sigmoid_function(self, X):
        """
        sigmoid = 1 / (1 + e^(-X))
        """
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            # calculate the weighted linear function
            y_lin = np.dot(X, self.w) + self.b
            y_preds = self.sigmoid_function(y_lin)

            """
            using BSE Loss Function:
            L = -1/n * sigma(y*log(y_pred) + (1-y)*log(1-y_pred))
            """

            dw = 1/n_samples * np.dot(y_preds-y, X)
            db = np.mean(y_preds - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        y_lin = np.dot(X, self.w) + self.b
        y_preds = self.sigmoid_function(y_lin)
        y_preds = np.array([1 if pred > 0.5 else 0 for pred in y_preds])

        return y_preds
