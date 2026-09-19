import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from torch.optim import Adam
from torch.nn import functional
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 4
batch_size = 64
learning_rate = 0.001

# dataset has PILImage images with pixel values of range [0, 1]
# transform and normalize the images to range to [-1, 1]
'''
if trying to augment data to increase training samples, load three MNIST dataset and transforms them
then concatenate them at the end

original = MNIST(..., transform=transforms.ToTensor())
rotated = MNIST(..., transform=transforms.RandomRotation())
translate = MNIST(..., transform=transforms.RandomAffine())

train_dataset = ConcatDataset([original, rotated, translate])
'''

transform = transforms.Compose(
    [transforms.ToTensor(),
     # each value represents the mean and std of one channel
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# import the dataset - CIFAR10 has images of size 32
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

# print(len(train_dataset))

# load the data using DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define the classes that CIFAR10 dataset detects
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Conv2d( input_channels, num_filters/output_channels, kernel_size, stride, padding )
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # shrinks by half

        self.conv2 = nn.Conv2d(6, 16, 5)

        # flatten layer
        # global average pool (called AdaptiveAvgPool in pytorch)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # the pooling layer is used twice
        # a better way to implement the FC layers is not to flatten, but globalAveragePool
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        # no softmax layer

        # activation layers will be implemented through torch or functional

    def forward(self, x):
        # convolutions
        output = functional.relu(self.conv1(x))
        output = self.pool(output)
        output = functional.relu(self.conv2(output))
        output = self.pool(output)

        # flatten
        output = self.global_average_pool(output)
        output = self.flatten(output)

        # fully connected layers
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = functional.relu(self.fc3(output))

        return output

model = CNN(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
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
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print(f"epoch {epoch}/{num_epochs}, batch {i}/{total_batches}:\n\tloss: {total_loss/(i*batch_size)}")

# test
with torch.no_grad():
    # general accuracy
    n_correct = 0
    n_samples = 0

    # class specific accuracy
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

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

        # get infor on each class
        for i in range(min(batch_size, len(labels))):
            label = labels[i]
            pred = predictions[i]

            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    print(f"accuracy: {100 * n_correct / n_samples}%\n")
    for i in range(len(n_class_correct)):
        print(f"{classes[i]} Accuracy: {100 * n_class_correct[i] / n_class_samples[i]}")
