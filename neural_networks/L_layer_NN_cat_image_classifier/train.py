from data_loader import data_loader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import NN
from sklearn.metrics import accuracy_score
from PIL import Image
from joblib import dump

# set the dimensions of the pixel
pixel_dims = 128

# load the data
X_raw, y, list_class = data_loader()

# check some images
# plt.imshow(X_raw[52])
# plt.show()

# view the data
# print(type(X_raw), X_raw.shape)
# print(type(y), y.shape)

# flatten and normalize the colors - each row will be a single sample, and the columns will be the colors
X_flat = X_raw.reshape(X_raw.shape[0], -1)
mew = np.mean(X_flat, axis=0, keepdims=True)
sigma = np.mean(np.square(X_flat), axis=0, keepdims=True)
X = (X_flat - mew) / sigma

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# transform the data into the ones that will work with the model
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.reshape(1, y_train.shape[0]), y_test.reshape(1, y_test.shape[0])

layers_dims = [X_train.shape[0], 20, 7, 5, 1]

'''
not regularized

testing - 50%
training - 100%
'''
# model = NN(layers_dims, mew, sigma, lr=0.0075)
# model.fit(X_train, y_train, print_cost=True)
#
# # save the model - non-regularized
# dump(model, "model/cat_model.joblib")
# print("model successfully saved to 'cat_model.joblib'")

'''
L2 regularized

testing - 68.75%
training - 100%
'''
model = NN(layers_dims, mew, sigma, lr=0.0075, lambd=0.075)
model.fit(X_train, y_train, print_cost=True, regularization="L2")

# # save the model - L2 regularized
# dump(model, "model/cat_model_L2_regularized.joblib")
# print("model successfully saved to 'cat_model_L2_regularized.joblib'")

'''
Dropout regularized

testing - 50%
training - 100%
'''
# model = NN(layers_dims, mew, sigma, lr=0.0075, keep_prob=0.75)
# model.fit(X_train, y_train, print_cost=True, regularization="dropout")
#
# # save the model - dropout regularized
# dump(model, "model/cat_model_dropout_regularized.joblib")
# print("model successfully saved to 'cat_model_dropout_regularized.joblib'")

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print(accuracy_score(y_pred.ravel(), y_test.ravel()))
print(accuracy_score(y_pred_train.ravel(), y_train.ravel()))
