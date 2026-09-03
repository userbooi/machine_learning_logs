import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

from model import *

# bc_dataset = load_breast_cancer()
# print(bc_dataset.data)
# print(bc_dataset.target)

# load the dataset
X_raw, y_raw = load_breast_cancer(return_X_y=True)

# split it into train, dev, test
X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# print(y_train.shape, y_dev.shape, y_test.shape)

AUTOTUNE = tf.data.AUTOTUNE

# create the tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(32).prefetch(AUTOTUNE)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, y_dev))
dev_dataset = dev_dataset.batch(32).prefetch(AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(32).prefetch(AUTOTUNE)

# create the model
model = logistic_regression()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=500, validation_data=dev_dataset)
# print(history.history)
