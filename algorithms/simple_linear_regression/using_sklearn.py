from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def MSE(pred, y):
    return ((pred-y)**2).mean()

X, y = make_regression(n_samples=300, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"MSE Value: {MSE(pred, y_test)}")

all_pred = model.predict(X)

# the visualization
cmap = plt.get_cmap("viridis")
plt.figure(figsize=(10, 10))
plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label="Train")
plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label="Test")
plt.plot(X, all_pred, color="black", linewidth=2, label="Prediction")
plt.legend()
plt.show()
