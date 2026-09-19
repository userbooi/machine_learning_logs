'''
transfer learning with a pre-trained ResNet-18
'''

import torch
from torch import nn, optim
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the weights
weights = models.ResNet18_Weights.DEFAULT
# mean = weights.transforms().mean
# std = weights.transforms().std
preprocessing = weights.transforms()

# view the preprocessing
# print(preprocessing)

# create the transforms
custom_transforms = {
    "train": transforms.Compose([
        # include randomness
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(preprocessing.mean, preprocessing.std)
    ]),
    "val": preprocessing
}

# print(custom_transforms["train"])

# create data from a folder clearly labeled
data_dir = 'hymenoptera_data'
sets = ['train', 'val']

# create a dictionary to contain the datasets for each set
image_dataset = {x: datasets.ImageFolder(f"./{data_dir}/{x}", transform=custom_transforms[x])
                 for x in sets}

# print(image_dataset["val"][3])

# create the dataloader
image_dataloader = {x: DataLoader(dataset=image_dataset[x], batch_size=16, shuffle=True)
                    for x in sets}

datasets_size = {x: len(image_dataset[x]) for x in sets}
class_names = image_dataset['train'].classes
# print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # checking the training/evaluation time
    since = time.time()

    # copies all the learnable parameters of the pre-trained model
    # this include the BatchNorm Parameters such as gamma and beta
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            # controls the running total for mean and std throughout the batches
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in image_dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(preds == labels)

            # update the learning rate using scheduling
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datasets_size[phase]
            epoch_acc = running_corrects.double() / datasets_size[phase]

            print(f'{phase} Loss: {round(epoch_loss, 4)} Acc: {round(epoch_acc.item(), 4)}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# import the pretrained model
model = models.resnet18(weights=weights)

# this is for just training the top layers
# comment this out to do fine-tuning with the entire model
for param in model.parameters():
    param.requires_grad_(False)

# only has 1 fully connected layer?
# print(model)

'''
if want to add more layers to a model, you can:

model.fc = nn.Sequential(
    [
        nn.Linear(...),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(...)
    ]
)

or

class MyModel(nn.Module):
    def __init__(self, pretrained, num_classes):
        super().__init__()
        
        self.pretrained = pretrained # this keeps even the top layers
        self.extra1 = nn.Linear(...)
        ...
        self.last = nn.Linear(..., num_classes)
        
    def forward(self, x):
        out = self.pretrained(x)
        out = self.extra1(x)
        ...
        out = self.last(out)
        
        return out
        
'''

num_features_top = model.fc.in_features
# change the top layers
model.fc = nn.Linear(num_features_top, len(class_names))

model.to(device)

learning_rate = 0.001

# loss and optimization function
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# scheduler
# every 7 epochs, the learning_rate = 0.1 * learning_rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)

