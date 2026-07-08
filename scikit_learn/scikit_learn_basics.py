import numpy as np
import pandas as pd
import scipy as sp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

'''
Some terms:
    - FEATURES are the characteristics used to describe data (X)
        - these are used to make predictions of the data
    - TARGETS are the values that we want to predict (y)
        - they are the correct answer that the algorithm use to compare its 
          predictions to minimize error
          
data is split into two categories to evaluate models:
    - Training Set
        - used to train the model
    - Testing set
        - used to evaluate how well the model generalizes
'''

#=============================loading the iris dataset===============================
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# print(iris)
# print(X)
# print(y)
#
# print("Feature names:", feature_names)
# print("Target names:", target_names)
#
# print("\nType of X is:", type(X))
# print("\nType of iris:", type(iris))
#
# print("\nFirst 5 rows of X:\n", X[:5])

#================================splitting datasets========================

# split the data into 60% training and 40% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# check the shapes to ensure the correct proportions
print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape:", y_test.shape)
