from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x


        out = self.conv1(x)
        # print(f'x: conv1:{out.shape}, type: {out.dtype}')
        out = self.bn1(out)
        # print(f'x: bn1:{out.shape}, type: {out.dtype}')
        out = self.relu(out)
        # print(f'x: relu:{out.shape}, type: {out.dtype}')

        out = self.conv2(out)
        # print(f'x: conv2:{out.shape}, type: {out.dtype}')
        out = self.bn2(out)
        # print(f'x: conv2:{out.shape}, type: {out.dtype}')

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # print(f'x: relu:{out.shape}, type: {out.dtype}')

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, input_shape=(32,32,3),num_classes=1000, block_name='BasicBlock', fc2conv=False):
        super(ResNet, self).__init__()
        self.fc2conv = fc2conv
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        h,w, ch = input_shape
        self.num_classes = num_classes
        self.inplanes = 16
        self.conv1 = nn.Conv2d(ch, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8) if not self.fc2conv else nn.AvgPool2d(8, stride=1)
        # self.avgpool = nn.AvgPool2d(kernel_size=(h//4,w//4), stride=1)
        fc = nn.Linear(int(h * w/16) * block.expansion, num_classes)
        fc_conv = nn.Conv2d(int(h * w/16) * block.expansion, num_classes, kernel_size=1, bias=True, padding=0)
        self.fc = fc_conv if self.fc2conv else fc

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f'{"+"*30}')
        # print(f'x: shape:{x.shape}, type: {x.dtype}')
        x = self.conv1(x)
        # print(f'conv1: shape:{x.shape}, type: {x.dtype}')
        x = self.bn1(x)
        # print(f'bn1: shape:{x.shape}, type: {x.dtype}')
        x = self.relu(x)    # 32x32
        # print(f'relu: shape:{x.shape}, type: {x.dtype}')

        x = self.layer1(x)  # 32x32
        # print(f'layer1: shape:{x.shape}, type: {x.dtype}')
        x = self.layer2(x)  # 16x16
        # print(f'layer2: shape:{x.shape}, type: {x.dtype}')
        x = self.layer3(x)  # 8x8
        # print(f'layer3: shape:{x.shape}, type: {x.dtype}')

        x = self.avgpool(x)
        # print(f'avgpool: shape:{x.shape}, type: {x.dtype}')
        if not self.fc2conv: x = x.view(x.size(0), -1)
        # x = x.view(x.size(0), -1)
        # print(f'view: shape:{x.shape}, type: {x.dtype}')
        x = self.fc(x)
        # print(f'fc: shape:{x.shape}, type: {x.dtype}')
        # x = nn.Linear(num_flat_features(x), self.num_classes)(x)

        return x

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
