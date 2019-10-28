'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
from collections import OrderedDict
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import getNetImageSizeAndNumFeats

logger = logging.getLogger(__name__)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_size=32, get_perceptual_feats=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        M = 1
        if image_size % 32 == 0:
            # 32 -> 512 * 1 * 1, 64 -> 512 * 2 * 2, ..., 256 -> 512 * 7 * 7 
            M = image_size / 2**5
        else:
            assert 0, 'image size %d not supported' %(image_size)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*M*M, num_classes)

        self._initialize_weights()
        self.get_perceptual_feats = get_perceptual_feats
        self.Out = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        Out = []
        out = F.relu(self.bn1(self.conv1(x)))
        # first layer was not hooked, therefore we have to add its result manually
        Out.append(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.get_perceptual_feats:
            for k, v in self.Out.iteritems():
                Out.append(v)
            Out.append(out)
            return out, Out
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _get_hook(self, layer_num, layer):
        Out = self.Out
        def myhook(module, _input, _out):
            Out[layer_num] = _out            
        layer.register_forward_hook(myhook)

def ResNet18(get_perceptual_feats=False, num_classes=10, image_size=32): # for imagenet (1000 classes)
    net =  ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, image_size=image_size, get_perceptual_feats = get_perceptual_feats)
    if get_perceptual_feats:
        imgSizeL, numFeatsL = getNetImageSizeAndNumFeats(net, image_size=image_size)
        net.ImageSizePerLayer = np.array(imgSizeL)
        net.numberOfFeaturesPerLayer = np.array(numFeatsL)
        
        # registers a hook for each RELU layer
        layer_num = 0
        for ftrLayers in [net.layer1, net.layer2, net.layer3, net.layer4]: 
            for resBlock in ftrLayers:
                for modl in resBlock.modules():
                    if str(modl)[0:4] == 'ReLU':
                        logger.info("# registering hook module {} ".format(str(modl)))
                        net._get_hook(layer_num, modl)
                        layer_num += 1
                
        ImgSizeL, numFeatsL = getNetImageSizeAndNumFeats(net, image_size=image_size)
        net.ImageSizePerLayer = np.array(ImgSizeL)
        net.numberOfFeaturesPerLayer = np.array(numFeatsL)
    return net
