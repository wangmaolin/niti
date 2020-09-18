import torch.nn as nn
import torch.nn.functional as F

class lenet(nn.Module):
    '''
    An example model for mnist from pytorch tutorial
    '''
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.regime = [{'epoch': 0,
                        'optimizer': 'SGD',
                        'lr': 0.01,
                        'momentum': 0.5},]

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

class lenet_cifar10(nn.Module):
    '''
    An example model for mnist from pytorch tutorial
    '''
    def __init__(self):
        super(lenet_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1250, 500)
        self.fc2 = nn.Linear(500, 10)
        self.regime = [{'epoch': 0,
                        'optimizer': 'SGD',
                        'lr': 0.01,
                        'momentum': 0.5},]

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
