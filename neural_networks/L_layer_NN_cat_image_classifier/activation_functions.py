import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return max(0, Z)

