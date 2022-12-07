
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


class Config:
    batch_size    = 64
    num_epochs    = 10
    learning_rate = 0.001
    vector_size   = 10


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
        self.fc3   = nn.Linear(84, Config.vector_size)

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

class CosineDistance_bi(nn.Module):
    def __init__(self):
        super(CosineDistance_bi, self).__init__()
    
    def forward(self, x, y, label):
        a = F.cosine_similarity(x, y, dim=1)
        return ((1-a)*label + a*(1-label)).mean()

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
        # class_pairs[trainset[i][1]].append(trainset[i][1]) # For debuggig

    # print ("---------------")
    # for i in class_pairs:
    #     print(len(i))

    positive_examples = []
    for i in range(10):
        for j in range(1100):
            positive_examples.append((class_pairs[i].popleft(), class_pairs[i].popleft()))

    # print ("---------------")
    # for i in class_pairs:
    #     print(len(i))

    negative_examples = []
    while len(negative_examples) < 11000:
        for j in range(10):
            for k in range(10):
                if j != k:
                    negative_examples.append((class_pairs[j].popleft(), class_pairs[k].popleft()))
    negative_examples = negative_examples[:11000]

    # print ("---------------")
    # for i in class_pairs:
    #     print(len(i))
    # print ("---------------")

    # Reshuffle time
    np.random.shuffle(positive_examples)
    np.random.shuffle(negative_examples)

    # Indexes to images
    positive_examples_left  = torch.stack([trainset[i[0]][0] for i in positive_examples])
    positive_examples_right = torch.stack([trainset[i[1]][0] for i in positive_examples])
    negative_examples_left  = torch.stack([trainset[i[0]][0] for i in negative_examples])
    negative_examples_right = torch.stack([trainset[i[1]][0] for i in negative_examples])

    # Debug (?)
    print("+L", len(positive_examples_left ))
    print("+R", len(positive_examples_right))
    print("-L", len(negative_examples_left ))
    print("-R", len(negative_examples_right))

    # Batch the data
    positive_examples_left  = positive_examples_left .split(Config.batch_size)
    positive_examples_right = positive_examples_right.split(Config.batch_size)
    negative_examples_left  = negative_examples_left .split(Config.batch_size)
    negative_examples_right = negative_examples_right.split(Config.batch_size)

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
    # criterion_0 = nn.CrossEntropyLoss()
    # criterion_1 = CosineDistance()
    criterion_2 = CosineDistance_bi()

    losses = []
    for epoch in range(Config.num_epochs):
        running_loss = 0.0
        for step, (pos_l,pos_r, neg_l,neg_r) in enumerate(zip(positive_examples_left, positive_examples_right, negative_examples_left, negative_examples_right)):
            
            combined_l = torch.cat((pos_l, neg_l))
            combined_r = torch.cat((pos_r, neg_r))
            labels     = torch.cat((torch.ones(combined_l.shape[0]//2), torch.zeros(combined_r.shape[0]//2)))

            # Train itself
            output_a = net(combined_l)
            output_b = net(combined_r)
            optimizer.zero_grad()
            loss = criterion_2(output_a, output_b, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print loss
            if step % 50 == 49:
                losses.append(running_loss / 50)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 50))
                running_loss = 0.0

    # Plot losses
    plt.plot(losses)
    plt.show()

    p = 0

    with torch.no_grad():
        for i in range(10):
            output_a = net(positive_examples_left [i * 1000 // Config.batch_size][0].view(-1, 1, 28, 28))
            output_b = net(positive_examples_right[i * 1000 // Config.batch_size][0].view(-1, 1, 28, 28))
            print(np.dot(output_a[0].numpy(), output_b[0].numpy()) / (np.linalg.norm(output_a[0].numpy()) * np.linalg.norm(output_b[0].numpy())))
        print ("---------------")
        for i in range(10):
            output_a = net(negative_examples_left [i][0].view(-1, 1, 28, 28))
            output_b = net(negative_examples_right[i][0].view(-1, 1, 28, 28))
            print(np.dot(output_a[0].numpy(), output_b[0].numpy()) / (np.linalg.norm(output_a[0].numpy()) * np.linalg.norm(output_b[0].numpy())))

    p = 0
