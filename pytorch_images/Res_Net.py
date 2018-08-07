from collections import OrderedDict

import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from torch.nn.init import xavier_normal_
from torch.nn.init import kaiming_uniform_
from torch.nn.init import kaiming_normal_

from inspect import signature

inplace_setup = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, channel_in, channel_hidden, stride=1):
        super().__init__()
        # direct path in the resnet
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(channel_in, channel_hidden,
                                    kernel_size=3, stride=stride,
                                    padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(channel_hidden)),
                ("relu1", nn.ReLU(inplace=inplace_setup)),
                ("conv2", nn.Conv2d(channel_hidden, channel_hidden,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
            ])
        )

        # short cut in the resnet, if same size, do nothing,
        # if different, transform
        if stride != 1 or channel_in != self.expansion * channel_hidden:
            self.short_cut = nn.Sequential(
                OrderedDict([
                    ("conv3", nn.Conv2d(channel_in,
                                        self.expansion * channel_hidden,
                                        kernel_size=1, stride=stride, bias=False)),
                    ("bn3", nn.BatchNorm2d(self.expansion * channel_hidden))
                ])
            )
        else:
            self.short_cut = nn.Sequential()

    def forward(self, x):
        out = self.direct_path(x) + self.short_cut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, channel_in, channel_hidden, stride=1):
        super().__init__()
        # direct path in the resnet
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(channel_in, channel_hidden,
                                    kernel_size=1, bias=False)),
                ("bn1", nn.BatchNorm2d(channel_hidden)),
                ("relu1", nn.ReLU(inplace=inplace_setup)),
                ("conv2", nn.Conv2d(channel_hidden, channel_hidden,
                                    kernel_size=3, stride=stride,
                                    padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
                ("relu2", nn.ReLU(inplace=inplace_setup)),
                ("conv3", nn.Conv2d(channel_hidden,
                                    self.expansion * channel_hidden,
                                    kernel_size=1, bias=False)),
                ("bn3", nn.BatchNorm2d(self.expansion * channel_hidden)),
            ])
        )

        # short cut in the resnet, if same size, do nothing,
        # if different, transform
        if stride != 1 or channel_in != self.expansion * channel_hidden:
            self.short_cut = nn.Sequential(
                OrderedDict([
                    ("conv4", nn.Conv2d(channel_in, 
                                        self.expansion * channel_hidden,
                                        kernel_size=1, stride=stride, bias=False)),
                    ("bn4", nn.BatchNorm2d(self.expansion * channel_hidden))
                ])
            )
        else:
            self.short_cut = nn.Sequential()

    def forward(self, x):
        out = self.direct_path(x) + self.short_cut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bottleneck_channels = 64

        self.block = cfg['block']
        self.num_blocks = cfg['num_blocks']
        self.num_classes = cfg['num_classes']

        # by default, avg pooling has the same stride as kernel size
        self.main = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(3, 64, kernel_size=3,
                                    stride=1, padding=1,
                                    bias=False)),
                ("bn1", nn.BatchNorm2d(64)),
                ("relu1", nn.ReLU(inplace=inplace_setup)),
                ("layer1", self._make_layer(self.block, 64,
                                            self.num_blocks[0], stride=1)),
                ("layer2", self._make_layer(self.block, 128,
                                            self.num_blocks[1], stride=2)),
                ("layer3", self._make_layer(self.block, 256,
                                            self.num_blocks[2], stride=2)),
                ("layer4", self._make_layer(self.block, 512,
                                            self.num_blocks[3], stride=2)),
                ("pool1", self._global_average_pooling())
            ])
        )

        self.fc1 = nn.Linear(512 * self.block.expansion, self.num_classes)
        self._init_kaiming_normal()

    def _make_layer(self, block, channel_hidden, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.bottleneck_channels,
                          channel_hidden, stride))
            self.bottleneck_channels = channel_hidden * block.expansion
        return nn.Sequential(*layers)

    def grad_on(self):
        for param in self.parameters():
            param.requires_grad = True

    def grad_off(self):
        for param in self.parameters():
            param.requires_grad = False

    def _init_params(self, init_type):
        for name, param in self.named_parameters():
            if name.find('conv') != -1:
                if name.find('weight') != -1:
                    print("initializing " + name)
                    # check for kaiming init
                    if str(signature(init_type)).find('mode') != -1:
                        init_type(param, mode='fan_out')
                    else:
                        init_type(param)
                if name.find('bias') != -1:
                    print("initializing " + name)
                    init.constant_(param, 0)
            elif name.find('fc') != -1:
                if name.find('weight') != -1:
                    print("initializing " + name)
                    init.normal_(param, std=1e-3)
                if name.find('bias') != -1:
                    print("initializing " + name)
                    init.constant_(param, 0)
            elif name.find('bn'):
                if name.find('weight') != -1:
                    print("initializing " + name)
                    #init.constant_(param, 1)
                    init.uniform_(param)
                if name.find('bias') != -1:
                    print("initializing " + name)
                    init.constant_(param, 0)

    def _init_xavier_uniform(self):
        self._init_params(xavier_uniform_)

    def _init_xavier_normal(self):
        self._init_params(xavier_normal_)

    def _init_kaiming_uniform(self):
        self._init_params(kaiming_uniform_)

    def _init_kaiming_normal(self):
        self._init_params(kaiming_normal_)

    def _global_average_pooling(self):
        return nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def ResNet18(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [2, 2, 2, 2],
        'num_classes': num_classes
    }
    return ResNet(cfg)


def ResNet34(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [3, 4, 6, 3],
        'num_classes': num_classes
    }
    return ResNet(cfg)


def ResNet50(num_classes=10):
    cfg = {
        'block': Bottleneck,
        'num_blocks': [3, 4, 6, 3],
        'num_classes': num_classes
    }
    return ResNet(cfg)


def ResNet101(num_classes=10):
    cfg = {
        'block': Bottleneck,
        'num_blocks': [3, 4, 23, 3],
        'num_classes': num_classes
    }
    return ResNet(cfg)


def ResNet152(num_classes=10):
    cfg = {
        'block': Bottleneck,
        'num_blocks': [3, 8, 36, 3],
        'num_classes': num_classes
    }
    return ResNet(cfg)


def test():
    net = ResNet18()
    y = net(torch.randn(100, 3, 32, 32))
    print(y.size())

test()
