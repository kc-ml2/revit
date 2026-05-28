"""
Datasets package for loading and preparing data loaders.
"""

from .cifar10 import get_cifar10_loaders
from .rot_mnist import get_rot_mnist_loaders
from .pcam import get_pcam_loaders

__all__ = [
    'get_cifar10_loaders',
    'get_rot_mnist_loaders',
    'get_pcam_loaders',
]

