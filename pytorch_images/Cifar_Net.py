from collections import OrderedDict

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform


class Cifar_Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 64, 5)),
                ("bn1", nn.BatchNorm2d(64)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(2, 2)),
            ])
        )

        self.layer2 = nn.Sequential(
            OrderedDict([
                ("conv2", nn.Conv2d(64, 64, 5)),
                ("bn2", nn.BatchNorm2d(64)),
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(2, 2)),
            ])
        )

        self.fc_layers = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(64 * 5 * 5, 384)),
                ("relu3", nn.ReLU()),
                ("fc2", nn.Linear(384, 192)),
                ("relu4", nn.ReLU()),
                ("fc3", nn.Linear(192, 10)),
            ])
        )

        self.init_Xavier()

    def init_Xavier(self):
        for name, param in self.named_parameters():
            if name.find('fc') != -1 or name.find('conv') != -1:
                if name.find('weight') != -1:
                    xavier_uniform(param)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc_layers(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
