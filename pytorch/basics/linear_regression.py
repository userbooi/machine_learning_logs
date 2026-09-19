import torch
from torch import nn, optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# for the simple model f = 2 * x

# create the data
X_numpy, y_numpy = datasets.make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
# print(X_numpy.shape, y_numpy.shape)

# must be a 2D array where each row is a new test sample
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
Y = Y.view((Y.shape[0], 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# view the data
# print(X, Y)

n_samples, n_features = X.shape

'''
lets say we needed a custom linear regression model
'''
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)

# model = nn.Linear(n_features, n_features) # built in
model = LinearRegression(n_features, n_features) # custom linear regression

lr = 0.01
epochs = 1000

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# print(list(model.parameters()))

# training loop
for epoch in range(epochs):

    # prediction - forward pass
    Y_hat = model(X_train)

    # calculate loss
    loss = criterion(Y_train, Y_hat)

    # calculate gradients
    loss.backward()

    # update the parameter
    optimizer.step()

    # clear the gradients so that it doesn't affect the next loop
    optimizer.zero_grad()

    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f"Epoch {epoch+1}: \n\tw = {w[0].item()}\n\tb = {b[0].item()}\n\tloss: {loss}")

print("\n")

Y_pred = model(X_test).detach()
with torch.no_grad():
    loss = criterion(Y_pred, Y_test)
    print(f"loss = {loss}")

plt.figure(figsize=(8, 8))
plt.scatter(X_train.numpy(), Y_train.numpy(), c="blue", s=10, label="training")
plt.scatter(X_test.numpy(), Y_test.numpy(), c="red", s=10, label="testing")
plt.plot(X_test.numpy(), Y_pred.numpy(), c="black", linewidth=5.0, label="regression")
plt.legend()
plt.show()
