import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_gradients(dA, Z):
    sig_Z = sigmoid(Z)
    return np.multiply(dA, np.multiply(sig_Z, (1-sig_Z)))

def relu_gradients(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

