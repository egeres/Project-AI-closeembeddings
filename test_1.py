
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
    """Takes MNIST image and outputs a [10,] vector"""

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

class CosineDistance(nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()
    
    def forward(self, x, y):
        # return 1 - F.cosine_similarity(x, y, dim=1)
        return 1 - F.cosine_similarity(x, y, dim=1).mean()

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

    # Turn each index into an image
    positive_examples_left  = torch.stack([trainset[i[0]][0] for i in positive_examples])
    positive_examples_right = torch.stack([trainset[i[1]][0] for i in positive_examples])
    negative_examples_left  = torch.stack([trainset[i[0]][0] for i in negative_examples])
    negative_examples_right = torch.stack([trainset[i[1]][0] for i in negative_examples])


    print("+L", len(positive_examples_left ))
    print("+R", len(positive_examples_right))
    print("-L", len(negative_examples_left ))
    print("-R", len(negative_examples_right))


    # Batch the data into 64 chunks
    positive_examples_left  = positive_examples_left .split(64)
    positive_examples_right = positive_examples_right.split(64)
    negative_examples_left  = negative_examples_left .split(64)
    negative_examples_right = negative_examples_right.split(64)



    p = 0

    # Plot a grid
    if False:
        def imshow(img):
            img = img / 2 + 0.5
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        imshow(torchvision.utils.make_grid(images))



    # Train
    net         = Net()
    optimizer   = optim.Adam(net.parameters(), lr=0.001)
    criterion_0 = nn.CrossEntropyLoss()
    criterion_1 = CosineDistance()

    for epoch in range(2):
        running_loss = 0.0
        for step, (a, b) in enumerate(zip(positive_examples_left, positive_examples_right)):
            
            # Train itself
            output_a = net(a)
            output_b = net(b)
            optimizer.zero_grad()
            loss = criterion_1(output_a, output_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print loss
            if step % 10 == 9:
                print('[%d, %5d] loss: %.5f' % (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    with torch.no_grad():
        for i in range(10):
            output_a = net(positive_examples_left [i * 1000 // 64][0].view(-1, 1, 28, 28))
            output_b = net(positive_examples_right[i * 1000 // 64][0].view(-1, 1, 28, 28))
            print(criterion_1(output_a, output_b))
        for i in range(10):
            output_a = net(negative_examples_left [i][0].view(-1, 1, 28, 28))
            output_b = net(negative_examples_right[i][0].view(-1, 1, 28, 28))
            print(criterion_1(output_a, output_b))

    p = 0


