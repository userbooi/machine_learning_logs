import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from model import LinearRegression

def MSE(y, pred):
    return ((pred-y)**2).mean()

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# # X is a 2D numpy array with shape (total_samples, 1)
# print(X)
# print(type(X))
# # y is a 1d numpy array
# print(y)
# print(type(y))

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(MSE(y_test, pred))

all_pred = model.predict(X)

cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label="Train")
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label="Test")
plt.plot(X, all_pred, color="black", linewidth=2, label="Prediction")
plt.legend()
plt.show()
