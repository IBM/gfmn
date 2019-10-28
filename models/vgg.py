from collections import OrderedDict
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from util import getNetImageSizeAndNumFeats

logger = logging.getLogger(__name__)

cfg = {
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, vgg_name, get_perceptual_feats=False, num_classes=10, image_size=32, classifier_depth=3):
        super(VGG, self).__init__()        
        vgg_modules = cfg[vgg_name]

        num_M = 0
        for m in vgg_modules:
            if m == 'M':
                num_M += 1
            
        self.features = self._make_layers(cfg[vgg_name])

        if image_size % (2**num_M) == 0:
            # 32 -> 512 * 1 * 1, 64 -> 512 * 2 * 2, ..., 256 -> 512 * 7 * 7 
            M = image_size / 2**num_M
            logger.info("_make_classifier: {},{},{},{}".format(M, 512 * M * M, num_classes, classifier_depth))
            self.classifier = self._make_classifier(512 * M * M, num_classes, classifier_depth)
        else:
            assert 0, 'image size %d not supported' %(image_size)
            
        self._initialize_weights()
        self.name = vgg_name
        self.Out = OrderedDict()
        self.get_perceptual_feats = get_perceptual_feats
            

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if self.get_perceptual_feats:
            Out = []
            for k, v in self.Out.iteritems():
                Out.append(v)
            Out.append(out)
            return out, Out
        else:
            return out


    def _make_classifier(self, input_size, num_classes, depth=3):        
        assert depth > 0 and depth <= 3, 'depth must be in [1,3] range not %d' %(depth)
        
        if depth == 1:
            classify = nn.Sequential(
                nn.Linear(input_size, num_classes),
            )
        elif depth == 2:
            classify = nn.Sequential(
                nn.Linear(input_size, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        elif depth == 3:
            classify = nn.Sequential(
                nn.Linear(input_size, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            assert 0, 'classifier depth %d is not supported' % depth

        return classify

        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu') # for torchvision version > v0.2.0
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)

    def _get_hook(self, layer_num):
        
        Out = self.Out
        def myhook(module, _input, _out):
            Out[layer_num] = _out            
        self.features[layer_num].register_forward_hook(myhook)

def buildVGGNet(net_type = 'VGG19', get_perceptual_feats=False, num_classes = 10, image_size = 32, classifier_depth=1):
    logger.info("buildVGGNet: {} {} {} {} {}".format(net_type,get_perceptual_feats,num_classes,image_size,classifier_depth))
    net = VGG(net_type, get_perceptual_feats, num_classes, image_size = image_size, classifier_depth=classifier_depth)
    logger.info("# net : {} {}".format(len(net.features), net))
    if get_perceptual_feats:
        for idx in range(len(net.features)):
            if str(net.features[idx])[0:4] == 'ReLU':
                logger.info("# registering hook module ({}), {}".format(idx, str(net.features[idx])))
                net._get_hook(idx)
        ImgSizeL, numFeatsL = getNetImageSizeAndNumFeats(net, image_size=image_size)
        net.ImageSizePerLayer = np.array(ImgSizeL)
        net.numberOfFeaturesPerLayer = np.array(numFeatsL)
    return net 

def VGG19(get_perceptual_feats=False, num_classes = 10, image_size = 32, classifier_depth=1):
    logger.info("VGG19: {},{},{},{}".format(get_perceptual_feats,num_classes,image_size,classifier_depth))
    return buildVGGNet('VGG19', get_perceptual_feats, num_classes, image_size = image_size, classifier_depth=classifier_depth)
