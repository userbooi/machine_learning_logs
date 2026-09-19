'''
this program trains a model that will train a hand-drawn digits recognizer

it will use the MNIST dataset for training

this implementation flattens the pixel grayscale values, but modern digit recognizer use
CNNs
'''

import torch
from torch import nn
from torch.optim import Adam
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# hyper parameters
input_size = 784 # 28x28
hidden_size = 100 # randomly chosen
num_classes = 10 # digits from 0 - 9
num_epochs = 2
batch_size = 64
learning_rate = 0.001

# import the MNIST dataset
'''
the root is where the dataset will be stored. It is also where pytorch will look to check if there exists
an MNIST set already

transform gives transformations to the dataset already

download=True specifies that if there is no MNIST in /data, to download one
'''
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=False)

# load the data through the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# check one batch
# examples = iter(train_loader)
# # the dimensions of the samples are (batch_size, channels, height, width)
# # the dimensions of the labels are (batch_size)
# samples, labels = next(examples)
# print(samples.shape, labels.shape)
#
# # visualize the images
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i, 0], cmap='gray')
# plt.show()

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        # no softmax layer

    def forward(self, x):

        out = self.flatten(x)
        # pass it through the layers
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)

        return out

model = NeuralNetwork(input_size, hidden_size, num_classes)

# loss and criterion
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# training loop
# number of batches
total_batches = len(train_loader)

# loop through the epochs
for epoch in range(num_epochs):
    # loop through each batch
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):

        # push the images and labels to the device
        # if GPU is available, it will run on GPU, otherwise, the CPU
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        y_pred = model(images)

        # calculate the loss
        loss = criterion(y_pred, labels)
        total_loss += loss.item() * labels.shape[0]

        # backward
        loss.backward()

        # update the parameters
        optimizer.step()

        # clear gradient
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(f"epoch {epoch}/{num_epochs}, batch {i}/{total_batches}:\n\tloss: {total_loss/(i*batch_size)}")

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # push the images and labels to the device
        # if GPU is available, it will run on GPU, otherwise, the CPU
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        y_pred = model(images)
        # value and index - value is probability, index is the class
        _, predictions = torch.max(y_pred, dim=1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    print(f"accuracy: {100 * n_correct / n_samples}%")
