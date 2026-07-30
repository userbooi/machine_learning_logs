from data_loader import data_loader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


X_raw, y, list_class = data_loader()

print(type(X_raw), X_raw.shape)
print(type(y), y.shape)
