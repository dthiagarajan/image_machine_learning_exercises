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

bias_setup = False
inplace_setup = True


# original nn.Conv2d stride=1, padding=0, by default
# for conv3x3 padding=1, 1x1=0
# common conv setups
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias_setup)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=bias_setup)


def conv1x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), bias=bias_setup)


def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride,
                     padding=(1, 0), bias=bias_setup)


def conv_pseudo3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        OrderedDict([
            ("halfconv1", conv1x3(in_planes, out_planes, stride=(1, stride))),
            ("halfbn1", nn.BatchNorm2d(out_planes)),
            ("halfrelu1", nn.ReLU(inplace=inplace_setup)),
            ("halfconv2", conv3x1(out_planes, out_planes, stride=(stride, 1)))
        ])
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, channel_in, channel_hidden, stride=1):
        super().__init__()
        # pre_act in the resnet
        self.pre_act = nn.Sequential(
            OrderedDict([
                ("bn1", nn.BatchNorm2d(channel_in)),
                ("relu1", nn.ReLU(inplace=inplace_setup))
            ])
        )

        # direct path
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("conv1", conv_pseudo3x3(channel_in, channel_hidden, stride=stride)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
                ("relu2", nn.ReLU(inplace=inplace_setup)),
                ("conv2", conv_pseudo3x3(channel_hidden, channel_hidden))
            ])
        )

        # short cut in the resnet, if same size, do nothing,
        # if different, transform
        if stride != 1 or channel_in != self.expansion * channel_hidden:
            self.short_cut = nn.Sequential(
                OrderedDict([
                    ("conv3", conv1x1(channel_in, self.expansion * channel_hidden, stride=stride))
                ])
            )
        else:
            self.short_cut = nn.Sequential()

    def forward(self, x):
        out = self.pre_act(x)
        shortcut_out = self.short_cut(out)
        directpath_out = self.direct_path(out)
        out = torch.add(directpath_out, shortcut_out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, channel_in, channel_hidden, stride=1):
        super().__init__()
        # pre_act in the resnet
        self.pre_act = nn.Sequential(
            OrderedDict([
                ("bn1", nn.BatchNorm2d(channel_in)),
                ("relu1", nn.ReLU(inplace=inplace_setup))
            ])
        )

        # direct path
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("conv1", conv1x1(channel_in, channel_hidden)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
                ("relu2", nn.ReLU(inplace=inplace_setup)),
                ("conv2", conv_pseudo3x3(channel_hidden, channel_hidden, stride=stride)),
                ("bn3", nn.BatchNorm2d(channel_hidden)),
                ("relu3", nn.ReLU(inplace=inplace_setup)),
                ("conv3", conv1x1(channel_hidden, self.expansion * channel_hidden)),
            ])
        )

        # short cut in the resnet, if same size, do nothing,
        # if different, transform
        if stride != 1 or channel_in != self.expansion * channel_hidden:
            self.short_cut = nn.Sequential(
                OrderedDict([
                    ("conv4", conv1x1(channel_in, self.expansion * channel_hidden, stride=stride))
                ])
            )
        else:
            self.short_cut = nn.Sequential()

    def forward(self, x):
        out = self.pre_act(x)
        shortcut_out = self.short_cut(out)
        directpath_out = self.direct_path(out)
        out = torch.add(directpath_out, shortcut_out)
        return out


class InceptPreActResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bottleneck_channels = 64

        self.block = cfg['block']
        self.num_blocks = cfg['num_blocks']
        self.num_classes = cfg['num_classes']

        # by default, avg pooling has the same stride as kernel size
        self.main = nn.Sequential(
            OrderedDict([
                ("conv1", conv3x3(3, 64)),
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
            if name.find('fc') != -1:
                if name.find('weight') != -1:
                    print("initializing " + name)
                    init.normal_(param, std=1e-3)
                if name.find('bias') != -1:
                    print("initializing " + name)
                    init.constant_(param, 0)
            # init bn before conv, since there is bn in pseudo conv,
            # and shouldn't be initialized by the conv way
            elif name.find('bn'):
                if name.find('weight') != -1:
                    print("initializing " + name)
                    #init.constant_(param, 1)
                    init.uniform_(param)
                if name.find('bias') != -1:
                    print("initializing " + name)
                    init.constant_(param, 0)
            elif name.find('conv') != -1:
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


def InceptPreActResNet18(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [2, 2, 2, 2],
        'num_classes': num_classes
    }
    return InceptPreActResNet(cfg)


def InceptPreActResNet34(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [3, 4, 6, 3],
        'num_classes': num_classes
    }
    return InceptPreActResNet(cfg)


def InceptPreActResNet50(num_classes=10):
    cfg = {
        'block': Bottleneck,
        'num_blocks': [3, 4, 6, 3],
        'num_classes': num_classes
    }
    return InceptPreActResNet(cfg)


def InceptPreActResNet101(num_classes=10):
    cfg = {
        'block': Bottleneck,
        'num_blocks': [3, 4, 23, 3],
        'num_classes': num_classes
    }
    return InceptPreActResNet(cfg)


def InceptPreActResNet152(num_classes=10):
    cfg = {
        'block': Bottleneck,
        'num_blocks': [3, 8, 36, 3],
        'num_classes': num_classes
    }
    return InceptPreActResNet(cfg)

'''
def test():
    net = InceptPreActResNet101()
    y = net(torch.randn(100, 3, 32, 32))
    print(y.size())

test()
'''
