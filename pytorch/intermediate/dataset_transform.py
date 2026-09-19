import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
                               
the torchvision.transform packages are mainly comprised of preprocessing and augmentation functions used to
transform images in general
'''

'''
transforms the MNIST dataset (which consists of PIL images), into tensors of shape

dataset = torchvision.datasets.MNIST(
    root='./data', transform=torchvision.transforms.ToTensor()
)
'''


'''
implement a custom dataset with batch sizes and stuff

this will load the wine dataset. There are 3 categories of wine in this dataset
the first column will have the target category and all the other are the features
'''
class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt("./data/wine.csv", delimiter=",", skiprows=1, dtype=np.float32)

        # leave this as a numpy array
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # (n_samples, 1)
        self.n_samples = xy.shape[0]

        self.transform=transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            # apply the transform to the data
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

# custom ToTensor class
# for educational purposes - in practical use, just use the torchvision.transform.ToTensor
class ToTensor:
    def __call__(self, sample):
        input, targets = sample
        return torch.from_numpy(input.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))

# multiply the values of the input
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor

        return inputs, target

'''
create the dataset
'''
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data

print(type(features), type(labels))
print(features)

print()

composed = torchvision.transforms.Compose([
    ToTensor(),
    MulTransform(2)
])

dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)
