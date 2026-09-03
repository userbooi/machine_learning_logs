import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

def logistic_regression():

    X = Input(shape=(30, ))
    output = Dense(1, activation="sigmoid")(X)

    model = Model(inputs=X, outputs=output)

    return model

