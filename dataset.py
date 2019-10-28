import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

def load_dataset(opt):
    if opt.dataset in ['imagenet', 'celeba']:
        transformations = []
        if opt.centerCropSize > opt.imageSize:
            transformations.extend([transforms.CenterCrop(opt.centerCropSize),
                                    transforms.Scale(opt.imageSize)])
        else:
            transformations.extend([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.centerCropSize)])

        if not opt.useAutoEncoder:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose(transformations))
    elif opt.dataset == 'lsun':
        transformations = []
        if opt.centerCropSize > opt.imageSize:
            transformations.extend([transforms.CenterCrop(opt.centerCropSize),
                                    transforms.Scale(opt.imageSize)])
        else:
            transformations.extend([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.centerCropSize)])

        if not opt.useAutoEncoder:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        dataset = dset.LSUN(opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose(transformations))

    elif opt.dataset == 'cifar10':
        transformations = [transforms.Scale(opt.imageSize), transforms.ToTensor()]
        if not opt.useAutoEncoder:
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))
        else:
            transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose(transformations))
    elif opt.dataset == 'stl10':
        transformations = [transforms.Scale(opt.imageSize), transforms.ToTensor()]
        if not opt.useAutoEncoder:
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))
        else:
            transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        dataset = dset.STL10(root=opt.dataroot, split='unlabeled', download=True,
                             transform=transforms.Compose(transformations))

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    return dataloader