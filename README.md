# **Learning Implicit Generative Models by Matching Perceptual Features**, _ICCV 2019_

> Perceptual features (PFs) have been used with great success in tasks such as transfer learning, style transfer, and super-resolution. However, the efficacy of PFs as key source of information for learning generative models is not well studied. We investigate here the use of PFs in the context of learning implicit generative models through moment matching (MM). More specifically, we propose a new effective MM approach that learns implicit generative models by performing mean and covariance matching of features extracted from pretrained ConvNets. Our proposed approach improves upon existing MM methods by: (1) breaking away from the problematic min/max game of adversarial learning; (2) avoiding online learning of kernel functions; and (3) being efficient with respect to both number of used moments and required minibatch size. Our experimental results demonstrate that, due to the expressiveness of PFs from pretrained deep ConvNets, our method achieves state-of-the-art results for challenging benchmarks.

## Related Links
+ Paper : [Link](http://openaccess.thecvf.com/content_ICCV_2019/papers/dos_Santos_Learning_Implicit_Generative_Models_by_Matching_Perceptual_Features_ICCV_2019_paper.pdf)
+ Blog Post : [Link](https://www.ibm.com/blogs/research/2019/10/learning-implicit-generative-models/)

## Requirements
* python 2.7
* pytorch 0.3.0

Please install requirements by `pip install -r requirements.txt`

## Experiments
### Training a generator for CIFAR10
+ Feature Extractor : `VGG19` and `Resnet18` 
+ Generator Architecture : `Resnet`
```bash
python gfmn.py --netGType resnet  --netEncType vgg19 resnet18  --dataset cifar10 \
--netEnc [path-to-pretrained-vgg19-model]  [path-to-pretrained-resnet18-model]
```
`[path-to-pretrained-X-model]` : Path to pretrained VGG19/Resnet18 classifiers. Refer [downloads](#downloads) section.

### Training a generator for STL10
+ Feature Extractor : `VGG19` and `Resnet18` 
+ Generator Architecture : `Resnet`
```bash
python gfmn.py --netGType resnet --netEncType vgg19 resnet18  --dataset stl10 \
--netEnc [path-to-pretrained-vgg19-model]  [path-to-pretrained-resnet18-model]
```
`[path-to-pretrained-X-model]` : Path to pretrained VGG19/Resnet18 classifiers. Refer [downloads](#downloads) section.

## Downloads
You can download pre-trained VGG19/Resnet18 classifiers from this [link](https://drive.google.com/drive/folders/1NJqVTEzfH0BQvvAZiqo-Nf2whbx3woIH?usp=sharing). These are the feature extractors we used in the above scripts to replicate the results, with `--netEnc` option. 

## bibtex
>  
    @InProceedings{Santos_2019_ICCV,
    author = {Santos, Cicero Nogueira dos and Mroueh, Youssef and Padhi, Inkit and Dognin, Pierre},
    title = {Learning Implicit Generative Models by Matching Perceptual Features},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}}

## Contributors
Cicero([@cicerons](https://github.com/cicerons)) / Youssef / Inkit([@ink-pad](https://github.com/ink-pad)) / Pierre
