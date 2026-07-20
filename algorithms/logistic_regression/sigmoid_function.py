import numpy as np
from math import e
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

"""
The sigmoid function is an "S" shaped function that has x e R and y e (0, 1)
it is helpful in classification problems as it can convert any real number into a probability between 0 and 1 for classification.

the function is:

sigmoid = 1/(1 + e^(-x))

its derivative is:

sigmoid' = e^(-x)/(1+e^(-x))^2 = (sigmoid)(1-sigmoid)

"""

def sigmoid(X):
    return 1 / (1 + e**(-X))

def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))

X = np.linspace(-10, 10, 200)
y = sigmoid(X)
y_prime = sigmoid_derivative(X)

plt.figure(figsize=(8, 6))
plt.plot(X, y, color="blue", linewidth=2, label="Sigmoid")
plt.plot(X, y_prime, color="red", linewidth=2, linestyle="--", label="Sigmoid Derivative")
plt.legend()
plt.show()
