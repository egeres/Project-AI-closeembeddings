
# Imports
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader
import torch.utils.data.sampler as sampler
import torchvision
import torchvision.transforms as transforms



# Define the convolutional neural network that takes a [1, 28, 28] image and outputs a [10] vector
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
        
    # Get data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    trainset = torchvision.datasets.MNIST(
        root      = './data',
        train     = True,
        download  = True,
        transform = transform,
    )
    testset  = torchvision.datasets.MNIST(
        root      = './data',
        train     = False,
        download  = True,
        transform = transform,
    )
    trainloader = data.DataLoader(
        trainset,
        batch_size = 4,
        shuffle    = True,
        num_workers= 2
    )
    testloader  = data.DataLoader(
        testset,
        batch_size = 4,
        shuffle    = False,
        num_workers= 2
    )

    # Generate tuples of isntances of the same class
    class_pairs = []
    for i in range(10):
        class_pairs.append(deque())
    
    for i in range(len(trainset)):
        class_pairs[trainset[i][1]].append(i)
        # class_pairs[trainset[i][1]].append(trainset[i][1])


    positive_examples = []
    for i in range(10):
        for j in range(2000):
            positive_examples.append((class_pairs[i].popleft(), class_pairs[i].popleft()))

    negative_examples = []
    for i in range(int(2000 / (10*10))):
        for j in range(10):
            for k in range(10):
                if j != k:
                    negative_examples.append((class_pairs[j].popleft(), class_pairs[k].popleft()))

    # Convert each list of tuples of indexes per instance into a tensor
    # positive_examples = torch.tensor(positive_examples)
    # negative_examples = torch.tensor(negative_examples)

    # Turn each index into an image
    positive_examples_left  = torch.stack([trainset[i[0]][0] for i in positive_examples])
    positive_examples_right = torch.stack([trainset[i[1]][0] for i in positive_examples])
    negative_examples_left  = torch.stack([trainset[i[0]][0] for i in negative_examples])
    negative_examples_right = torch.stack([trainset[i[1]][0] for i in negative_examples])


    # neural network takes a [1, 28, 28] image and outputs a [10] vector
    # train such that the network outputs a high value for the same class and a low value for different classes



    p = 0

    # Plot a grid
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))










