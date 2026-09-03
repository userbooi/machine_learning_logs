import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

def logistic_regression():

    X = Input(input_shape=(30, ))
    output = Dense(1, activation="relu")(X)

    model = Model(inputs=X, outputs=output)

    return model

