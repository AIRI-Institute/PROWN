import torch
import os

from torch.utils.data import Dataset
from torchvision import transforms, datasets
from typing import *

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
# on aisecure gpu1
# os.environ[IMAGENET_LOC_ENV] = "/data/datasets/imagenet/ILSVRC2012"
# on asedl
# os.environ[IMAGENET_LOC_ENV] = "/srv/local/data/ImageNet/ILSVRC2012_full"
# on asedl but mount to aisecure gpu1
# os.environ[IMAGENET_LOC_ENV] = "/home/linyi2/data_mnt/imagenet/ILSVRC2012"
# on asedl but sync mount
# os.environ[IMAGENET_LOC_ENV] = "/home/linyi2/data_mnt/local_imagenet/data/ImageNet/ILSVRC2012_full"
# on aws server
# os.environ[IMAGENET_LOC_ENV] = '/data/imagenet/ILSVRC2012'
# on gpu3 server
os.environ[IMAGENET_LOC_ENV] = '/workspace/raid/data/datasets/imagenet/'

# list of all datasets
DATASETS = ["imagenet", "cifar10", "cifar100", "mnist", "svhn"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar100":
        return _cifar100(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "fashionmnist":
        return _fashion_mnist(split)
    elif dataset == "svhn":
        return _svhn(split)
    else:
        raise Exception("Unknown dataset")


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    elif dataset == "mnist":
        return 10
    elif dataset == "svhn":
        return 10
    else:
        raise Exception("Unknown dataset")


def get_dataset_shape(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return (3, 224, 224)
    elif dataset == "cifar10":
        return (3, 32, 32)
    elif dataset == "cifar100":
        return (3, 32, 32)
    elif dataset == "mnist":
        return (1, 28, 28)
    elif dataset == "svhn":
        return (3, 32, 32)
    else:
        raise Exception("Unknown dataset")

# def get_normalize_layer(dataset: str, device=None) -> torch.nn.Module:
#     """Return the dataset's normalization layer"""
#     if dataset == "imagenet":
#         return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV, device)
#     elif dataset == "cifar10":
#         return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV, device)
#     elif dataset == "cifar100":
#         return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV, device)
#     elif dataset == "mnist":
#         return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV, device)
#     else:
#         raise Exception("Unknown dataset")


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STDDEV = [0.2675, 0.2565, 0.2761]

_MNIST_MEAN = [0.5]
_MNIST_STDDEV = [0.5]

_SVHN_MEAN = [0.4377, 0.4438, 0.4728]
_SVHN_STDDEV = [0.1201, 0.1231, 0.1052]

_DEFAULT_MEAN = [0.5, 0.5, 0.5]
_DEFAULT_STDDEV = [0.5, 0.5, 0.5]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
        ]))

    
def _cifar100(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STDDEV)
        ]))
    elif split == "test":
        return datasets.CIFAR100("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STDDEV)
        ]))
    

def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STDDEV)
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STDDEV)
        ])
    return datasets.ImageFolder(subdir, transform)


def _mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STDDEV)
        ]))
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STDDEV)
        ]))


def _fashion_mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.FashionMNIST("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.FashionMNIST("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _svhn(split: str) -> Dataset:
    if split == "train":
        return datasets.SVHN('./dataset_cache', split=split, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_SVHN_MEAN, std=_SVHN_STDDEV)
        ]))
    elif split == "test":
        return datasets.SVHN("./dataset_cache", split=split, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=_SVHN_MEAN, std=_SVHN_STDDEV)
        ]))
        

def get_bounds(dataset):
    if dataset == "imagenet":
        dataset_mean, dataset_std = _IMAGENET_MEAN, _IMAGENET_STDDEV
    elif dataset == "cifar10":
        dataset_mean, dataset_std = _CIFAR10_MEAN, _CIFAR10_STDDEV
    elif dataset == "cifar100":
        dataset_mean, dataset_std = _CIFAR100_MEAN, _CIFAR100_STDDEV
    elif dataset == "mnist":
        dataset_mean, dataset_std = _MNIST_MEAN, _MNIST_STDDEV
    else:
        raise Exception("Unknown dataset")
    
    dataset_mean, dataset_std = torch.tensor(dataset_mean), torch.tensor(dataset_std)
    
    zeros = torch.zeros_like(dataset_mean)
    zeros_to = (zeros - dataset_mean) / dataset_std
    ones = torch.ones_like(dataset_mean)
    ones_to = (ones - dataset_mean) / dataset_std
    l, r = min(zeros_to).item(), max(ones_to).item()
    
    return (l, r)


# class NormalizeLayer(torch.nn.Module):
#     """Standardize the channels of a batch of images by subtracting the dataset mean
#       and dividing by the dataset standard deviation.

#       In order to certify radii in original coordinates rather than standardized coordinates, we
#       add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
#       layer of the classifier rather than as a part of preprocessing as is typical.
#       """

#     def __init__(self, means: List[float], sds: List[float], device):
#         """
#         :param means: the channel means
#         :param sds: the channel standard deviations
#         """
#         super(NormalizeLayer, self).__init__()
# #         self.means = torch.tensor(means).cuda()
# #         self.sds = torch.tensor(sds).cuda()

#         self.means = torch.tensor(means).to(device)
#         self.sds = torch.tensor(sds).to(device)
#         self.device = device


#     def forward(self, input: torch.tensor):
#         (batch_size, num_channels, height, width) = input.shape
#         means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
#         sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
#         return (input - means) / sds
