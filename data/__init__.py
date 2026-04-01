"""
Data loaders for DAMCHA experiments.

Supports:
- CV: CIFAR-10, CIFAR-100, MNIST
- NLP: WMT, CommonGen, CNN/DailyMail
"""

from .datasets import (
    get_image_reconstruction_datasets,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_mnist_dataloader,
    get_test_dataloader,
    get_dataset_info,
)

from .datasets_nlp import (
    get_nlp_dataloader,
    get_nlp_dataset_info,
)

__all__ = [
    'get_image_reconstruction_datasets',
    'get_cifar10_dataloader',
    'get_cifar100_dataloader',
    'get_mnist_dataloader',
    'get_test_dataloader',
    'get_dataset_info',
    'get_nlp_dataloader',
    'get_nlp_dataset_info',
]
