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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, channel_in, channel_hidden, drop_rate, preact_option, stride=1):
        super().__init__()
        self.preact_option = preact_option
        # need to preact only if in_size != out_size
        self.need_preact = (channel_in != self.expansion * channel_hidden)

        # preact path in the resnet
        self.pre_act = nn.Sequential(
            OrderedDict([
                ("bn1", nn.BatchNorm2d(channel_in)),
                ("relu1", nn.ReLU(inplace=inplace_setup))
            ])
        )

        # direct path
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("conv1", conv3x3(channel_in, channel_hidden, stride=stride)),
                ("dropout1", nn.Dropout(p=drop_rate)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
                ("relu2", nn.ReLU(inplace=inplace_setup)),
                ("conv2", conv3x3(channel_hidden, self.expansion * channel_hidden))
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
        # whether or not apply the preact
        if self.preact_option and self.need_preact:
            shortcut_out = self.short_cut(out)
        else:
            shortcut_out = self.short_cut(x)
        directpath_out = self.direct_path(out)
        out = torch.add(directpath_out, shortcut_out)
        return out


class WideResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bottleneck_channels = 16

        self.block = cfg['block']
        self.num_blocks = cfg['num_blocks']
        self.k = cfg['k']
        self.drop_rate = cfg['drop_rate']
        self.preact_option = cfg['preact_option']
        self.num_classes = cfg['num_classes']

        # calculate the output from the last layer
        self.last_layer_out_size = 64 * self.k[2] * self.block.expansion

        # by default, avg pooling has the same stride as kernel size
        self.main = nn.Sequential(
            OrderedDict([
                ("conv1", conv3x3(3, 16)),
                ("layer1", self._make_layer(self.block, 16, self.k[0],
                                            self.num_blocks[0], self.drop_rate,
                                            self.preact_option, stride=1)),
                ("layer2", self._make_layer(self.block, 32, self.k[1],
                                            self.num_blocks[1], self.drop_rate,
                                            self.preact_option, stride=2)),
                ("layer3", self._make_layer(self.block, 64, self.k[2],
                                            self.num_blocks[2], self.drop_rate,
                                            self.preact_option, stride=2)),
                ("bn4", nn.BatchNorm2d(self.last_layer_out_size)),
                # batch norm with 0.9 momentum as used in some implementation
                #("bn4", nn.BatchNorm2d(self.last_layer_out_size,
                #                       momentum=0.9)),
                ("relu4", nn.ReLU(inplace=inplace_setup)),
                ("pool4", self._global_average_pooling()),
            ])
        )

        self.fc1 = nn.Linear(self.last_layer_out_size, self.num_classes)
        self._init_kaiming_normal()

    def _make_layer(self, block, base_channel_hidden, k, num_block, drop_rate,
                    preact_option, stride):
        strides = [stride] + [1] * (num_block - 1)
        # calculate the number of filters
        channel_hidden = base_channel_hidden * k
        layers = []
        for stride in strides:
            layers.append(block(self.bottleneck_channels,
                          channel_hidden, drop_rate, preact_option, stride))
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


def WideResNet16_8(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [2, 2, 2],
        'k': [8, 8, 8],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet16_10(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [2, 2, 2],
        'k': [10, 10, 10],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet16_20(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [2, 2, 2],
        'k': [20, 20, 20],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet22_8(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [3, 3, 3],
        'k': [8, 8, 8],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet22_10(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [3, 3, 3],
        'k': [10, 10, 10],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet22_20(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [3, 3, 3],
        'k': [20, 20, 20],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet28_8(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [4, 4, 4],
        'k': [8, 8, 8],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet28_10(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [4, 4, 4],
        'k': [10, 10, 10],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet28_20(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [4, 4, 4],
        'k': [20, 20, 20],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet40_8(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [6, 6, 6],
        'k': [8, 8, 8],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet40_10(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [6, 6, 6],
        'k': [10, 10, 10],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)


def WideResNet40_20(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [6, 6, 6],
        'k': [20, 20, 20],
        'drop_rate': 0.3,
        'preact_option': True,
        'num_classes': num_classes,
    }
    return WideResNet(cfg)

'''
def test():
    net = WideResNet22_8()
    y = net(torch.randn(100, 3, 32, 32))
    print(y.size())

test()
'''
