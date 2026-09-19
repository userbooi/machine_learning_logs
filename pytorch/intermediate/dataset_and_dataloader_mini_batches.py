import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

'''
minibatch gradient descent works better than normal gradient descent

these batches can also be applied to more effective optimization algorithms like ADAM

pytorch handles these through Datasets and Dataloader

Clarification on terminology:
    - epoch = 1 forward and backward pass of ALL training samples
    - batch_size = number of training samples in one forward & backward pass
    - number of iterations = number of passes, each pass using [batch_size] number of samples

    e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
'''

'''
implement a custom dataset with batch sizes and stuff

this will load the wine dataset. There are 3 categories of wine in this dataset
the first column will have the target category and all the other are the features
'''
class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt("./data/wine.csv", delimiter=",", skiprows=1, dtype=np.float32)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # (n_samples, 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

'''
create the dataset
'''
dataset = WineDataset()

'''
load the dataset with dataloader
'''
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2) # num_workers makes it faster

# training loop
epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass
        # calculate loss
        # backward pass
        # update variables
        # clear gradients
        pass

'''
built-in datasets from pytorch
'''
# gray scale handwritten digit dataset consisting of 70000 images
torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco
