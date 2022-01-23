import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 8, 3)
        self.batch1_1 = nn.BatchNorm2d(8)
        self.conv1_2 = nn.Conv2d(8, 16, 3)
        self.batch1_2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(16, 16, 3)
        self.batch2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(16, 8, 3)
        self.batch2_2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, classes)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.batch1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.batch1_2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.batch2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.batch2_2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x