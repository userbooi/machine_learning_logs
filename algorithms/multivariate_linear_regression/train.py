import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from model import MultivariateLinearRegression

def MSE(y_pred, y):
    return ((y_pred - y)**2).mean()

cali_housing = fetch_california_housing()

X = pd.DataFrame(cali_housing.data, columns=cali_housing.feature_names)
y = pd.Series(cali_housing.target)
X = X[["MedInc", "AveRooms"]].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_train.shape, y_train.shape)

model = MultivariateLinearRegression(lr=0.003, epochs=10000)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(f"Bias: {model.b}")
print(f"Weights: {model.w}")
print(MSE(y_test_pred, y_test))

# ============================================= visualization =================================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test[:, 0], X_test[:, 1],
           y_test, color='blue', label='Actual Data')

x1_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
x2_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)

z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
ax.plot_surface(x1, x2, z, color='red', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('Median Income')
ax.set_ylabel('Average Rooms')
ax.set_zlabel('House Price')
ax.set_title('Multiple Linear Regression Best Fit Line (3D)')

plt.show()

