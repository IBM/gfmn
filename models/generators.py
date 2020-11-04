from collections import OrderedDict
from math import log
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DeconvDecoder(nn.Module):

    def __init__(self, ngpu, encSize, ngf, numberOfChannels = 3, removeBatchNorm = False, 
                 useRelu = True, num_convs=4, numExtraLayers = 0, sizeOfFirstDeconvKernel = 4):
        super(DeconvDecoder, self).__init__()
        self.ngpu = ngpu
        
        useBatchNorm = not removeBatchNorm
        
        nnLayers = OrderedDict()
        
        numOutChannelFirstConv = ngf * 8
        if num_convs < 4:
            numOutChannelFirstConv = ngf * 4
            
        # first deconv goes from the encoding size
        nnLayers["deconv_%d"%(len(nnLayers))]   = nn.ConvTranspose2d(encSize, numOutChannelFirstConv, sizeOfFirstDeconvKernel, 1, 0, bias=False)
        if useBatchNorm:
            nnLayers["btn_%d"%(len(nnLayers))]  =  nn.BatchNorm2d(numOutChannelFirstConv)
        if useRelu:
            nnLayers["relu_%d"%(len(nnLayers))] = nn.ReLU(True)

        for i in range(num_convs - 4):
            self.createDeconvBlock(nnLayers, ngf * 8, ngf * 8, useBatchNorm, useRelu)
        
        if num_convs >= 4:     
            self.createDeconvBlock(nnLayers, ngf * 8, ngf * 4, useBatchNorm, useRelu)
        
        self.createDeconvBlock(nnLayers, ngf * 4, ngf * 2, useBatchNorm, useRelu)
        self.createDeconvBlock(nnLayers, ngf * 2, ngf , useBatchNorm, useRelu)
        
        for i in  xrange(numExtraLayers):
            self.createConvBlock(nnLayers, ngf, ngf, useBatchNorm, useRelu)
        
        self.createDeconvBlock(nnLayers, ngf , numberOfChannels, False, False)
        nnLayers["tanh_%d"%(len(nnLayers))] = nn.Tanh()
        
        self.net = nn.Sequential(nnLayers)

    def createDeconvBlock(self, layersDict, input_nc, output_nc, useBatchNorm = True, useRelu = True):
        layersDict["deconv_%d"%(len(layersDict))] = nn.ConvTranspose2d(input_nc, output_nc, 4, 2, 1, bias=False)
        if useBatchNorm:
            layersDict["btnDeconv_%d"%(len(layersDict))] =  nn.BatchNorm2d(output_nc)
        if useRelu:
            layersDict["reluDeconv_%d"%(len(layersDict))] = nn.ReLU(True)

    def createConvBlock(self, layersDict, input_nc, output_nc, useBatchNorm = True, useRelu = True):
        layersDict["conv_%d"%(len(layersDict))] = nn.Conv2d(input_nc, output_nc, 3, 1, 1, bias=False)
        if useBatchNorm:
            layersDict["btnConv_%d"%(len(layersDict))] =  nn.BatchNorm2d(output_nc)
        if useRelu:
            layersDict["reluConv_%d"%(len(layersDict))] = nn.ReLU(True)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else:
            output = self.net(input)
            
        return output


class ConvEncoderSkipConnectionBlock(nn.Module):
    def __init__(self, dataForSkipConnections, submodule, blockId, input_nc, output_nc, useBatchNorm, nonLinearity='',
                 filterSize=4, stride=2, padding=1, dropoutRate=0, useFCLayer=False):
        super(ConvEncoderSkipConnectionBlock, self).__init__()

        self._useFCLayer = useFCLayer
        self._dataForSkipConnections = dataForSkipConnections

        nnLayers = OrderedDict()

        if submodule != None:
            nnLayers["submodel"] = submodule

        if self._useFCLayer:
            nnLayers["fc_%d" % blockId] = nn.Linear(input_nc, output_nc, bias=True)
        else:
            nnLayers["conv_%d" % blockId] = nn.Conv2d(input_nc, output_nc, filterSize, stride, padding, bias=False)
        if useBatchNorm:
            nnLayers["btnConv_%d" % blockId] = nn.BatchNorm2d(output_nc)
        if nonLinearity == 'tanh':
            nnLayers["reluConv_%d" % blockId] = nn.Tanh()
        elif nonLinearity == 'relu':
            nnLayers["reluConv_%d" % blockId] = nn.ReLU(True)
        elif nonLinearity == 'lrelu':
            nnLayers["lreluConv_%d" % blockId] = nn.LeakyReLU(0.2, inplace=True)

        if dropoutRate > 0:
            nnLayers["dropout_%d" % blockId] = nn.Dropout(dropoutRate)

        self.net = nn.Sequential(nnLayers)

    def forward(self, input):
        if self._useFCLayer:
            output = self.net(input.view(input.size()[0], -1))
            output = output.view(output.size()[0], output.size()[1], 1, 1)
        else:
            output = self.net(input)
        self._dataForSkipConnections.append(output)
        return output

class ConvEncoderSkipConnections(nn.Module):
    
    def __init__(self, ngpu, encSize, ndf, numberOfChannels = 3, 
                 removeBatchNorm = False, nonLinearity = 'relu', num_convs=4, 
                 dropoutRate=0, nonLinearityOfLastLayer = 'tanh', useFCForLastLayer = False, numExtraLayers = 0):
        super(ConvEncoderSkipConnections, self).__init__()
        self.ngpu = ngpu
        
        self.encSize   = encSize
        self._dataForSkipConnections = []
        
        self.ImageSizePerLayer = [1, 4, 8, 16, 32, 64][:num_convs+1]
        self.numberOfFeaturesPerLayer = None
        # image size is >= 64
        if num_convs > 3:
            self.numberOfFeaturesPerLayer = [self.encSize, 8192, 16384, 32768, 65536, 131072][:num_convs+1]
        # image size is 32
        else:
            self.numberOfFeaturesPerLayer = [self.encSize, 4096, 8192, 16384]
        
        useBatchNorm = not removeBatchNorm
        
        blockId = 0
        
        net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, None, blockId, 
                                             numberOfChannels, ndf, useBatchNorm, nonLinearity = nonLinearity, dropoutRate=dropoutRate)
        blockId += 1

        for i in xrange(numExtraLayers):
            net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, net, blockId, 
                                                 ndf, ndf , useBatchNorm, nonLinearity = nonLinearity, dropoutRate=dropoutRate,
                                                 filterSize = 3, stride = 1, padding = 1)
            blockId += 1
            # the number of features output in the extra layer is the same as the one in the first layer
            self.numberOfFeaturesPerLayer.append(self.numberOfFeaturesPerLayer[-1])
            self.ImageSizePerLayer.append(self.ImageSizePerLayer[-1])
        
        self.numberOfFeaturesPerLayer = np.asarray(self.numberOfFeaturesPerLayer)
        self.ImageSizePerLayer = np.asarray(self.ImageSizePerLayer)
        
        
        net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, net, blockId, 
                                             ndf, ndf * 2, useBatchNorm, nonLinearity = nonLinearity, dropoutRate=dropoutRate)
        blockId += 1
                
        
        net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, net, blockId, 
                                             ndf*2, ndf * 4, useBatchNorm, nonLinearity = nonLinearity, dropoutRate=dropoutRate)
        blockId += 1
        
        if num_convs >= 4:
            
            net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, net, 3, 
                                                 ndf*4, ndf * 8, useBatchNorm, nonLinearity = nonLinearity, dropoutRate=dropoutRate)
            blockId += 1
            numChannelsForLastLayer = ndf * 8
        else:
            numChannelsForLastLayer = ndf * 4
            
        for i in range(num_convs - 4):
            
            net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, net, blockId, 
                                                 ndf*8, ndf * 8, useBatchNorm, nonLinearity = nonLinearity, dropoutRate=dropoutRate)
            blockId += 1
            
        # last conv output is the encoding size
        # nnLayers["conv_%d"%(len(nnLayers))]      = nn.Conv2d(ndf * 8, encSize, 4, 1, 0, bias=False)
        net = ConvEncoderSkipConnectionBlock(self._dataForSkipConnections, net, blockId, 
                                             numChannelsForLastLayer, encSize, useBatchNorm=False, nonLinearity = nonLinearityOfLastLayer, 
                                             filterSize = 4, stride = 1, padding = 0, useFCLayer = useFCForLastLayer)
        
        self.net = net

    def forward(self, input):
        # removes data of skip connections from previous execution 
        del self._dataForSkipConnections[:]
        
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else:
            output = self.net(input)
        
        return [output, self._dataForSkipConnections[:]]

    def getNumberOfFeaturesPerLayer(self):
        return self.numberOfFeaturesPerLayer

    def getImageSizePerLayer(self):
        return self.ImageSizePerLayer

class ResnetG(nn.Module):
    def __init__(self, nz, nc, ndf, imageSize = 32, adaptFilterSize = False, useConvAtSkipConn = False):
        super(ResnetG, self).__init__()
        self.nz = nz
        self.ndf = ndf

        if adaptFilterSize == True and useConvAtSkipConn == False:
            useConvAtSkipConn = True
            logger.warn("WARNING: In ResnetG, setting useConvAtSkipConn to True because adaptFilterSize is True.")

        numUpsampleBlocks = int(log(imageSize, 2)) - 2 
        
        numLayers = numUpsampleBlocks + 1
        filterSizePerLayer = [ndf] * numLayers
        if adaptFilterSize:
            for i in xrange(numLayers - 1, -1, -1):
                if i == numLayers - 1:
                    filterSizePerLayer[i] = ndf
                else:
                    filterSizePerLayer[i] = filterSizePerLayer[i+1]*2
            
        firstL = nn.ConvTranspose2d(nz, filterSizePerLayer[0], 4, 1, 0, bias=False)
        nn.init.xavier_uniform(firstL.weight.data, 1.)
        lastL  = nn.Conv2d(filterSizePerLayer[-1], nc, 3, stride=1, padding=1)
        nn.init.xavier_uniform(lastL.weight.data, 1.)

        nnLayers = OrderedDict()
        # first deconv goes from the z size
        nnLayers["firstConv"]   = firstL
        
        layerNumber = 1
        for i in xrange(numUpsampleBlocks):
            nnLayers["resblock_%d"%i] = ResidualBlockG(filterSizePerLayer[layerNumber-1], filterSizePerLayer[layerNumber], stride=2, useConvAtSkipConn = useConvAtSkipConn)
            layerNumber += 1
        nnLayers["batchNorm"] = nn.BatchNorm2d(filterSizePerLayer[-1])
        nnLayers["relu"]      = nn.ReLU()
        nnLayers["lastConv"]  = lastL
        nnLayers["tanh"]      = nn.Tanh()

        self.net = nn.Sequential(nnLayers)

    def forward(self, input):
        return self.net(input)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, size=None):
        super(Upsample, self).__init__()
        self.upsample = F.upsample_nearest
        self.size = size
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.upsample(x, size=self.size, scale_factor = self.scale_factor)
        return x

class ResidualBlockG(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, useConvAtSkipConn = False):
        super(ResidualBlockG, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        
        if useConvAtSkipConn:
            self.conv_bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.conv_bypass.weight.data, 1.)
        
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            if useConvAtSkipConn:
                self.bypass = nn.Sequential(self.conv_bypass, Upsample(scale_factor=2))
            else:
                self.bypass = Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)