from __future__ import print_function
import argparse
import os
import random
import logging.handlers
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from models.resnet import ResNet18
from models.vgg import VGG19
from models.generators import DeconvDecoder, ConvEncoderSkipConnections, ResnetG
from dataset import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | mnist | stl10', default='cifar10')
parser.add_argument('--dataroot', required=True, help='path to dataset', default="./data/")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--centerCropSize', type=int, default=0, help='Size to use when performing center cropping.')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=float, default=2e6, help='number of generator updates')
parser.add_argument('--firstBatchId', type=int, default=0, help='sequential number to be used in the first batch')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer: default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netEnc', nargs='*', default=[''], help="path to netEnc (to continue training)")
parser.add_argument('--netEncType', nargs='*', default=['vgg19'],
                    help="type of encoder/feature extractor to use: 'encoder | vgg19 [default]| resnet18'")
parser.add_argument('--modelDir', default='', help="path to previously trained model (to continue training)")
parser.add_argument('--netGType', default='dcgan', help="type of generator/decoder to use: 'dcgan[default] | resnet'")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed', default=31)

parser.add_argument('--setEvalAtTest', action='store_true', help='Sets model to eval mode during test time')
parser.add_argument('--numBatchsToValid', type=int, default=500,
                    help='Number of batches after which the validation is applied')
parser.add_argument('--numLayersToFtrMatching', type=int, default=16,
                    help='Number of layers of the encoder/feature extractor used to perform feature matching')
parser.add_argument('--ftrMatchingWithTopLayers', action='store_true',
                    help="Uses the top <--numLayersToFtrMatching> layers to perform feature matching.")
parser.add_argument('--mAvrgAlpha', type=float, default=1.0,
                    help='Term used to balance the trade-off in the regular moving average.')

parser.add_argument('--lrMovAvrg', type=float, default=1e-5, help='Learning rate for moving average, default=0.0002')

parser.add_argument('--useRegularMovAvrg', action='store_true',
                    help="Use regular moving average instead of Adam-based moving average")

parser.add_argument('--setEncToEval', action='store_true', help="Sets encoder/feature extractor to eval state.")

parser.add_argument('--numOfEncExtraLayers', type=int, default=2, help="Number of extra layers in the encoder.")
parser.add_argument('--numOfDecExtraLayers', type=int, default=2, help="Number of extra layers in the generator.")

parser.add_argument('--useAutoEncoder', action='store_true',
                    help="Use features from autoencoder instead of a classifier.")

parser.add_argument('--vggClassifierDepth', type=int, default=1,
                    help="Number of fully connected layers in the VGG classifier.")

parser.add_argument('--notAdaptFilterSize', action='store_true',
                    help="Does not use a different number of filters for each conv. layer [Resnet generator only].")

parser.add_argument('--useConvAtGSkipConn', action='store_true',
                    help="For Resnet generator, applies a conv. layer to the input before upsampling in the skip connection.")

parser.add_argument('--saveFeatureExtractor', action='store_true',
                    help="Saves the feature extractor model to directory specified by --outf.")

parser.add_argument('--numBatchsToSaveModelToNewFile', type=float, default=4e5,
                    help="Number of batches after which the model is saved into a NEW file.")
parser.add_argument('--numBatchsToSaveModel', type=float, default=3000,
                    help="Number of epochs after which the model is saved (to the same file).")
parser.add_argument('--numClassesInFtrExt', type=int, default=1000,
                    help="Number of classes in the feature extractor classifier.")


# Logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s(%(name)s): %(message)s')
consH = logging.StreamHandler()
consH.setFormatter(formatter)
consH.setLevel(logging.DEBUG)
logger.addHandler(consH)
request_file_handler = True
log = logger

opt = parser.parse_args()
log.info("Opt: {}".format(opt))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
log.info("Random Seed: {}".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

if opt.centerCropSize == 0:
    opt.centerCropSize = opt.imageSize

imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
imageNetNormMin = -imageNetNormMean / imageNetNormStd
imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
imageNetNormRange = imageNetNormMax - imageNetNormMin

dataloader = load_dataset(opt)

ngpu = int(opt.ngpu)
nz   = int(opt.nz)
ngf  = int(opt.ngf)
ndf  = int(opt.ndf)
mAvrgAlpha = opt.mAvrgAlpha

sizeOfFirstDeconvKernel = 4
if opt.imageSize == 48:
    sizeOfFirstDeconvKernel = 6

nc = 3
num_convs = 4
if opt.imageSize in [256, 224]:
    num_convs = 6
elif opt.imageSize == 128:
    num_convs = 5
elif opt.imageSize in [32, 48]:
    num_convs = 3


# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ConvEncoder') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# LOAD ALL FEATURE EXTRACTORS/ENCODERS
netEnc = []
curNetEncId = -1
for netEncType in opt.netEncType:
    curNetEncId += 1

    if netEncType == 'resnet18':  # pretrained resnet18
        netEnc.append(ResNet18(get_perceptual_feats=True, num_classes=opt.numClassesInFtrExt, image_size=opt.imageSize))
        state = torch.load(opt.netEnc[curNetEncId])
        net = state['net']
        netEnc[-1].load_state_dict(net.state_dict())

    elif netEncType == 'vgg19':  # CIFAR-10 pretrained vgg19
        netEnc.append(VGG19(get_perceptual_feats=True, num_classes=opt.numClassesInFtrExt, image_size=opt.imageSize,
                            classifier_depth=opt.vggClassifierDepth))
        log.info("Reading Feat Exatractor #{} from {}".format(curNetEncId,opt.netEnc[curNetEncId]))
        state = torch.load(opt.netEnc[curNetEncId])
        net = state['net']
        netEnc[-1].load_state_dict(net.state_dict())

    else:
        netEnc.append(ConvEncoderSkipConnections(ngpu, nz, ndf, numberOfChannels=nc,
                                                 removeBatchNorm=False,
                                                 nonLinearity='relu', num_convs=num_convs,
                                                 dropoutRate=0.0,
                                                 nonLinearityOfLastLayer='tanh',
                                                 useFCForLastLayer=False,
                                                 numExtraLayers=opt.numOfEncExtraLayers))

        netEnc[-1].apply(weights_init)
        if opt.netEnc[curNetEncId] != '':
            netEnc[-1].load_state_dict(torch.load(opt.netEnc[curNetEncId]))

    log.info('# Encoder :{}'.format(netEncType))

# CREATES THE GENERATOR
log.info('# Generator:')
if opt.netGType == "dcgan":
    netG = DeconvDecoder(ngpu, nz, ngf, numberOfChannels=nc, removeBatchNorm=False,
                         useRelu=True, num_convs=num_convs, numExtraLayers=opt.numOfDecExtraLayers,
                         sizeOfFirstDeconvKernel=sizeOfFirstDeconvKernel)
    netG.apply(weights_init)
elif opt.netGType == "resnet":
    netG = ResnetG(nz, nc, ngf, opt.imageSize, adaptFilterSize=not opt.notAdaptFilterSize,
                   useConvAtSkipConn=opt.useConvAtGSkipConn)

if opt.modelDir != '':
    netG.load_state_dict(torch.load('%s/netG.pth' % (opt.modelDir)))
log.info(netG)

numFeaturesInEnc = 0
numFeaturesForEachSelectedLayer = []
# computes the total number of features
for curNetEnc in netEnc:
    # number of features output for each layer, from the last to the first layer
    numFeaturesForEachEncLayer = curNetEnc.numberOfFeaturesPerLayer
    log.info("# numFeaturesForEachEncLayer (from top to bottom): {}".format(numFeaturesForEachEncLayer))

    numLayersToFtrMatching = min(opt.numLayersToFtrMatching, len(numFeaturesForEachEncLayer))
    log.info("@ opt.ftrMatchingWithTopLayers: {}".format(opt.ftrMatchingWithTopLayers))
    log.info("@ actual numLayersToFtrMatching: {}".format(numLayersToFtrMatching))

    if opt.ftrMatchingWithTopLayers:
        numFeaturesInEnc += sum(numFeaturesForEachEncLayer[:numLayersToFtrMatching])
        numFeaturesForEachSelectedLayer = numFeaturesForEachEncLayer[:numLayersToFtrMatching]
    else:
        numFeaturesInEnc += sum(numFeaturesForEachEncLayer[-numLayersToFtrMatching:])
        numFeaturesForEachSelectedLayer = numFeaturesForEachEncLayer[-numLayersToFtrMatching:]

if not opt.ftrMatchingWithTopLayers:
    numFeaturesForEachSelectedLayer = numFeaturesForEachSelectedLayer[::-1]  # orders from last to first layer
log.info("# of features to be used: {}".format(numFeaturesInEnc))

# creates Networks for moving Mean and Variance.
netMean = nn.Linear(numFeaturesInEnc, 1, bias=False)
if opt.modelDir != '' and os.path.exists('%s/netMean.pth' % (opt.modelDir)):
    netMean.load_state_dict(torch.load('%s/netMean.pth' % (opt.modelDir)))
log.info(netMean)
netVar = nn.Linear(numFeaturesInEnc, 1, bias=False)
if opt.modelDir != '' and os.path.exists('%s/netVar.pth' % (opt.modelDir)):
    netVar.load_state_dict(torch.load('%s/netVar.pth' % (opt.modelDir)))
log.info(netVar)

criterionL1Loss = nn.L1Loss()
criterionL2Loss = nn.MSELoss()
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(min(64, opt.batchSize), nz, 1, 1).normal_(0, 1) # for visual inspection

# Variables used to renormalize the data to ImageNet scale.
imageNetNormMinV   = torch.FloatTensor(imageNetNormMin)
imageNetNormRangeV = torch.FloatTensor(imageNetNormRange)

if opt.cuda:
    for curNetEnc in netEnc:
        curNetEnc.cuda()
    netG.cuda()
    netMean.cuda()
    netVar.cuda()
    criterionL1Loss.cuda()
    criterionL2Loss.cuda()
    input = input.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    imageNetNormMinV = imageNetNormMinV.cuda()
    imageNetNormRangeV = imageNetNormRangeV.cuda()

imageNetNormMinV.resize_(1, 3, 1, 1)
imageNetNormRangeV.resize_(1, 3, 1, 1)
imageNetNormMinV = Variable(imageNetNormMinV)
imageNetNormRangeV = Variable(imageNetNormRangeV)

fixed_noise = Variable(fixed_noise)

# setups optimizers
parametersG = set()
parametersG |= set(netG.parameters())
optimizerG = optim.Adam(parametersG, lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerMean = optim.Adam(netMean.parameters(), lr=opt.lrMovAvrg, betas=(opt.beta1, 0.999))
optimizerVar = optim.Adam(netVar.parameters(), lr=opt.lrMovAvrg, betas=(opt.beta1, 0.999))

def extractFeatures(batchOfData, detachOutput=False):
    """
    Applies feature extractor. Concatenate feature vectors from all selected layers.
    """
    # gets features from each layer of netEnc
    ftrs = []
    for curNetEnc in netEnc:
        ftrsPerLayer = curNetEnc(batchOfData)[1]

        for lId in xrange(1, numLayersToFtrMatching + 1):
            cLid = lId - 1  # gets features in forward order
            if opt.ftrMatchingWithTopLayers:
                cLid = -lId  # gets features in backward order (last layers first)

            ftrsOfLayer = ftrsPerLayer[cLid].view(ftrsPerLayer[cLid].size()[0], -1)
            if detachOutput:
                ftrs.append(ftrsOfLayer.detach())
            else:
                ftrs.append(ftrsOfLayer)
    ftrs = torch.cat(ftrs, dim=1)
    return ftrs

movingAvrgMeanFakeData = []
movingAvrgVarFakeData = []

globalFtrMeanValues = []
globalFtrVarValues  = []
featureSqrdValues   = []

numExamplesProcessed = 0.0
log.info("Computing mean features from TRUE data")
for i, data in enumerate(dataloader, 1):
    # gets real images
    real_cpu, _ = data
    if opt.cuda:
        real_cpu = real_cpu.cuda()

    input.resize_as_(real_cpu).copy_(real_cpu)
    realData = Variable(input)
    numExamplesProcessed += realData.size()[0]

    # extracts features for TRUE data
    allFtrsTrue = extractFeatures(realData, detachOutput=True)
    if len(globalFtrMeanValues) < 1:
        globalFtrMeanValues = torch.sum(allFtrsTrue, dim=0).detach()
        featureSqrdValues = torch.sum(allFtrsTrue ** 2, dim=0).detach()
    else:
        globalFtrMeanValues += torch.sum(allFtrsTrue, dim=0).detach()
        featureSqrdValues += torch.sum(allFtrsTrue ** 2, dim=0).detach()

# variance = (SumSq - (Sum x Sum) / n) / (n - 1)
globalFtrVarValues = (featureSqrdValues - (globalFtrMeanValues ** 2) / numExamplesProcessed) / (
            numExamplesProcessed - 1)
log.info("Normalizing sum of features with denominator: {}".format(numExamplesProcessed))
globalFtrMeanValues = globalFtrMeanValues / numExamplesProcessed

def saveModel(suffix=""):
    # saving current best model
    torch.save(netG.state_dict(), '%s/netG%s.pth' % (opt.outf, suffix))
    if not opt.useRegularMovAvrg:
        torch.save(netMean.state_dict(), '%s/netMean%s.pth' % (opt.outf, suffix))
        torch.save(netVar.state_dict(), '%s/netVar%s.pth' % (opt.outf, suffix))


numBatchsToSaveModelToNewFile = int(
    opt.numBatchsToSaveModelToNewFile)  # save (copy)  models every <numBatchsToSaveModelToNewFile> of batches
numBatchsToSaveModel = int(opt.numBatchsToSaveModel)
niter = int(opt.niter)

# number of batches after which we increase the sequential used in the output images file name
numBatchsToIncOutImgSeq = 5000
if opt.imageSize == 32:
    numBatchsToIncOutImgSeq = 10000

avrgLossNetGMean = 0.0
avrgLossNetGVar  = 0.0
avrgLossNetMean  = 0.0
avrgLossNetVar   = 0.0
batch_size = opt.batchSize
#######################################################
# GFMN Training Loop.
#######################################################
for iterId in range(opt.firstBatchId, niter):

    for curNetEnc in netEnc:
        curNetEnc.zero_grad()
    netG.zero_grad()
    netMean.zero_grad()
    netVar.zero_grad()

    if opt.setEncToEval:
        for curNetEnc in netEnc:
            curNetEnc.eval()

    # creates noise
    noise.resize_(batch_size, nz, 1, 1).normal_(0, 1.0)
    noisev = Variable(noise)

    fakeData = netG(noisev)

    if not opt.useAutoEncoder:
        # normalizes the generated images using imagenet min-max ranges
        # newValue = (((fakeData - OldMin) * NewRange) / OldRange) + NewMin
        fakeData = (((fakeData + 1) * imageNetNormRangeV) / 2) + imageNetNormMinV

    # extract features from FAKE data
    ftrsFakeData = extractFeatures(fakeData, detachOutput=False)

    # uses regular moving average
    if opt.useRegularMovAvrg:
        # updates model using regular moving average of mean and variance
        ftrsMeanFakeData = torch.mean(ftrsFakeData, 0)
        ftrsVarFakeData  = torch.var(ftrsFakeData, 0)

        if len(movingAvrgMeanFakeData) < 1:
            movingAvrgMeanFakeData = ftrsMeanFakeData.clone().detach()
            movingAvrgVarFakeData = ftrsVarFakeData.clone().detach()

        # updates moving average of variance
        movingAvrgVarFakeData = (1.0 - mAvrgAlpha) * movingAvrgVarFakeData.detach() + mAvrgAlpha * ftrsVarFakeData

        # updates moving average of mean
        movingAvrgMeanFakeData = (1.0 - mAvrgAlpha) * movingAvrgMeanFakeData.detach() + mAvrgAlpha * ftrsMeanFakeData

        lossNetG = criterionL2Loss(movingAvrgMeanFakeData, globalFtrMeanValues.detach()) + \
               criterionL2Loss(movingAvrgVarFakeData, globalFtrVarValues.detach())

        avrgLossNetGMean += lossNetG.data[0]
        lossNetG.backward()
        optimizerG.step()

    # uses Adam moving average
    else:
        # updates moving average of mean differences
        ftrsMeanFakeData = torch.mean(ftrsFakeData, 0)
        diffFtrMeanTrueFake = globalFtrMeanValues.detach() - ftrsMeanFakeData.detach()
        lossNetMean = criterionL2Loss(netMean.weight, diffFtrMeanTrueFake.detach().view(1, -1))
        lossNetMean.backward()
        avrgLossNetMean += lossNetMean.data[0]
        optimizerMean.step()

        # updates moving average of variance differences
        ftrsVarFakeData = torch.var(ftrsFakeData, 0)
        diffFtrVarTrueFake = globalFtrVarValues.detach() - ftrsVarFakeData.detach()
        lossNetVar = criterionL2Loss(netVar.weight, diffFtrVarTrueFake.detach().view(1, -1))
        lossNetVar.backward()
        avrgLossNetVar += lossNetVar.data[0]
        optimizerVar.step()

        # updates generator
        meanDiffXTrueMean = netMean(globalFtrMeanValues.view(1, -1)).detach()
        meanDiffXFakeMean = netMean(ftrsMeanFakeData.view(1, -1))
        varDiffXTrueVar   = netVar(globalFtrVarValues.view(1, -1)).detach()
        varDiffXFakeVar   = netVar(ftrsVarFakeData.view(1, -1))

        lossNetGMean = (meanDiffXTrueMean - meanDiffXFakeMean)
        avrgLossNetGMean += lossNetGMean.data[0]

        lossNetGVar = (varDiffXTrueVar - varDiffXFakeVar)
        avrgLossNetGVar += lossNetGVar.data[0]

        lossNetG = lossNetGMean + lossNetGVar
        lossNetG.backward()
        optimizerG.step()

    if (iterId + 1) % opt.numBatchsToValid == 0:

        if opt.setEvalAtTest:
            for curNetEnc in netEnc:
                curNetEnc.eval()
            netG.eval()

        log.info('[{}/{}] Loss_Gz: {.6f} Loss_GzVar: {.6f} Loss_vMean: {.6f} Loss_vVar: {.6f}'.format
                 (iterId + 1, niter,
                 avrgLossNetGMean / opt.numBatchsToValid, avrgLossNetGVar / opt.numBatchsToValid,
                 avrgLossNetMean / opt.numBatchsToValid, avrgLossNetVar / opt.numBatchsToValid))

        fileSuffix = (iterId + 1) / numBatchsToIncOutImgSeq
        fixed_noiseIn = fixed_noise
        fake = netG(fixed_noiseIn).detach()
        vutils.save_image(fake.data[:64],
                          '%s/fake_samples_iterId_%04d.png' % (opt.outf, fileSuffix),
                          normalize=True)

        os.sys.stdout.flush()

        avrgLossNetGMean = 0.0
        avrgLossNetMean  = 0.0
        avrgLossNetGVar  = 0.0
        avrgLossNetVar   = 0.0

        if opt.setEvalAtTest:
            for curNetEnc in netEnc:
                curNetEnc.train()
            netG.train()

    # saving models
    if (iterId + 1) % numBatchsToSaveModel == 0:
        # checkpointing
        if iterId + 1 == numBatchsToSaveModel and opt.saveFeatureExtractor:
            encId = 0
            for curNetEnc in netEnc:
                torch.save(curNetEnc.state_dict(), '%s/netEnc_%s.pth' % (opt.outf, opt.netEncType[encId]))
                encId += 1

        saveModel(suffix="")

    # saving model with a different suffix
    if (iterId + 1) % numBatchsToSaveModelToNewFile == 0:
        saveModel(suffix=".%02d" % (iterId / numBatchsToSaveModelToNewFile))
