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

    def __init__(self, channel_in, channel_hidden, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        # direct path
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("bn1", nn.BatchNorm2d(channel_in)),
                ("conv1", conv3x3(channel_in, channel_hidden, stride=stride)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
                ("relu2", nn.ReLU(inplace=inplace_setup)),
                ("conv2", conv3x3(channel_hidden, self.expansion * channel_hidden)),
                ("bn3", nn.BatchNorm2d(self.expansion * channel_hidden)),
            ])
        )

    def forward(self, x):
        direct_out = self.direct_path(x)

        if self.downsample is not None:
            shortcut_out = self.downsample(x)
            featuremap_size = shortcut_out.size()[2:4]
        else:
            shortcut_out = x
            featuremap_size = direct_out.size()[2:4]

        batch_size = direct_out.size()[0]
        residual_channel_size = direct_out.size()[1]
        shortcut_channel_size = shortcut_out.size()[1]

        if residual_channel_size != shortcut_channel_size:
            padding = torch.FloatTensor(batch_size,
                                        residual_channel_size - shortcut_channel_size,
                                        featuremap_size[0], featuremap_size[1]).fill_(0)

            if shortcut_out.is_cuda:
                padding = padding.cuda()

            shortcut_out = torch.cat((shortcut_out, padding), 1)

        out = torch.add(direct_out, shortcut_out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, channel_in, channel_hidden, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        # direct path
        self.direct_path = nn.Sequential(
            OrderedDict([
                ("bn1", nn.BatchNorm2d(channel_in)),
                ("conv1", conv1x1(channel_in, channel_hidden)),
                ("bn2", nn.BatchNorm2d(channel_hidden)),
                ("relu2", nn.ReLU(inplace=inplace_setup)),
                ("conv2", conv3x3(channel_hidden, channel_hidden, stride=stride)),
                ("bn3", nn.BatchNorm2d(channel_hidden)),
                ("relu3", nn.ReLU(inplace=inplace_setup)),
                ("conv3", conv1x1(channel_hidden, channel_hidden * self.expansion)),
                ("bn4", nn.BatchNorm2d(channel_hidden * self.expansion))
            ])
        )

    def forward(self, x):
        direct_out = self.direct_path(x)

        if self.downsample is not None:
            shortcut_out = self.downsample(x)
            featuremap_size = shortcut_out.size()[2:4]
        else:
            shortcut_out = x
            featuremap_size = direct_out.size()[2:4]

        batch_size = direct_out.size()[0]
        residual_channel_size = direct_out.size()[1]
        shortcut_channel_size = shortcut_out.size()[1]

        if residual_channel_size != shortcut_channel_size:
            padding = torch.FloatTensor(batch_size,
                                        residual_channel_size - shortcut_channel_size,
                                        featuremap_size[0], featuremap_size[1]).fill_(0)

            if shortcut_out.is_cuda:
                padding = padding.cuda()

            shortcut_out = torch.cat((shortcut_out, padding), 1)

        out = torch.add(direct_out, shortcut_out)

        return out


class PyramidNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.block = cfg['block']
        self.num_blocks = cfg['num_blocks']
        self.alpha = cfg['alpha']
        self.small_image = cfg['small_image']
        self.num_classes = cfg['num_classes']

        self.block_type = self.block.__class__.__name__

        self.addrate = self.alpha / (sum(self.num_blocks) * 1.0)

        if self.small_image:
            # input dim is similar to bottleneck_channels in resnet
            self.input_dim = 16
            # if small images, use the same num of blocks
            # for all layers
            self.main_head = nn.Sequential(
                OrderedDict([
                    ("conv1", conv3x3(3, self.input_dim, stride=1)),
                    ("bn1", nn.BatchNorm2d(self.input_dim))
                ])
            )

            self.current_dim = self.input_dim
            self.main_mid = nn.Sequential(
                OrderedDict([
                    ("layer1", self._make_layer(self.block, self.num_blocks[0])),
                    ("layer2", self._make_layer(self.block, self.num_blocks[1], stride=2)),
                    ("layer3", self._make_layer(self.block, self.num_blocks[2], stride=2))
                ])
            )

            # similar to last_layer_out_size in wideresnet
            self.output_dim = self.input_dim
            self.main_tail = nn.Sequential(
                OrderedDict([
                    ("bnn", nn.BatchNorm2d(self.output_dim)),
                    ("relun", nn.ReLU(inplace=inplace_setup)),
                    ("pooln", self._global_average_pooling()),
                ])
            )
        else:
            self.input_dim = 64

            self.main_head = nn.Sequential(
                OrderedDict([
                    ("conv1", nn.Conv2d(3, self.input_dim, kernel_size=7,
                                        stride=2, padding=3, bias=bias_setup)),
                    ("bn1", nn.BatchNorm2d(self.input_dim)),
                    ("relu1", nn.ReLU(inplace=inplace_setup)),
                    ("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                ])
            )

            self.current_dim = self.input_dim
            self.main_mid = nn.Sequential(
                OrderedDict([
                    ("layer1", self._make_layer(self.block, self.num_blocks[0])),
                    ("layer2", self._make_layer(self.block, self.num_blocks[1], stride=2)),
                    ("layer3", self._make_layer(self.block, self.num_blocks[2], stride=2)),
                    ("layer4", self._make_layer(self.block, self.num_blocks[3], stride=2))
                ])
            )

            self.output_dim = self.input_dim
            self.main_tail = nn.Sequential(
                OrderedDict([
                    ("bnn", nn.BatchNorm2d(self.output_dim)),
                    ("relun", nn.ReLU(inplace=inplace_setup)),
                    ("pooln", self._global_average_pooling()),
                ])
            )

        self.fcn = nn.Linear(self.output_dim, self.num_classes)
        self._init_kaiming_normal()

    def _make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.current_dim = self.current_dim + self.addrate
        layers.append(block(self.input_dim, int(round(self.current_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_dim = self.current_dim + self.addrate
            layers.append(block(int(round(self.current_dim)) * block.expansion,
                                int(round(temp_dim)), 1))
            self.current_dim = temp_dim
        self.input_dim = int(round(self.current_dim)) * block.expansion

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
        out = self.main_head(x)
        out = self.main_mid(out)
        out = self.main_tail(out)
        out = out.view(out.size(0), -1)
        out = self.fcn(out)
        return out


def PyramidNet110_48(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [18, 18, 18],
        'alpha': 48,
        'small_image': True,
        'num_classes': num_classes,
    }
    return PyramidNet(cfg)


def PyramidNet110_270(num_classes=10):
    cfg = {
        'block': BasicBlock,
        'num_blocks': [18, 18, 18],
        'alpha': 270,
        'small_image': True,
        'num_classes': num_classes,
    }
    return PyramidNet(cfg)

'''
def test():
    net = PyramidNet110_48()
    print(net)
    net.cuda()
    y = net(torch.randn(100, 3, 32, 32).cuda())
    print(y.size())

test()
'''
