import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# # X is a 2D numpy array with shape (total_samples, 1)
# print(X)
# print(type(X))
# # y is a 1d numpy array
# print(y)
# print(type(y))

