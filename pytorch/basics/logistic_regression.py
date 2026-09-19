import torch
from torch import nn, optim
from torch.nn.functional import normalize
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X_numpy, y_numpy = load_breast_cancer(return_X_y=True)
X, y = torch.from_numpy(X_numpy.astype(np.float32)), torch.from_numpy(y_numpy.astype(np.float32))
# view the dataset
# print(X_numpy.shape, y_numpy.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# view the split datasets
# print(X_train.shape, y_train.shape)

# normalize
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

# reshape the y tensors
y_train = y_train.view((y_train.shape[0], 1))
y_test = y_test.view((y_test.shape[0], 1))

'''
or use StandardScalar from sklearn

from sklearn.preprocessing import StandardScalar

sc = StandardScalar() # like a layer
X_train = sc.fit_transform(X_train) # calculates the mean and std
X_test = sc.transform(X_test) # simply normalize using the mean and std from training
'''

class LogisticRegression(nn.Module):

    def __init__(self, input_features):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, X):
        y_pred = torch.sigmoid(self.linear(X))
        return y_pred

input_features = X_train.shape[1]
model = LogisticRegression(input_features)

criterion = nn.BCELoss()

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
epochs = 200

for epoch in range(epochs):

    # forward pass
    y_pred = model(X_train_normalized)

    # loss
    loss = criterion(y_pred, y_train)

    # back propagation
    loss.backward()

    # update parameters
    optimizer.step()

    # reset the gradients
    optimizer.zero_grad()

    # log some of the logs
    if epoch % 10 == 0:
        print(f"Epoch {epoch} -- loss: {loss}")

with torch.no_grad():
    y_pred = model(X_test_normalized)
    # convert to zero or 1
    y_pred_decoded = y_pred.round()

    acc = y_pred_decoded.eq(y_test).sum() / y_test.shape[0]
    print(f"\nAccuracy: {acc*100:.4f}%")

