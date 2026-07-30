from data_loader import data_loader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# # set the dimensions of the pixel
# pixel_dims = 128
#
# # load the data
# X_raw, y, list_class = data_loader()
#
# # check some images
# # plt.imshow(X_raw[9])
# # plt.show()
#
# # view the data
# # print(type(X_raw), X_raw.shape)
# # print(type(y), y.shape)
#
# # flatten and normalize the colors - each column will be a single sample, and the rows will be the colors
# X_flat = X_raw.reshape(-1, X_raw.shape[0])
# X = X_flat / 255
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# test
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.reshape(1, y_train.shape[0]), y_test.reshape(1, y_test.shape[0])

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred.ravel(), y_test.ravel()))
model.show_costs()
