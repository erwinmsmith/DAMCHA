"""
Dataset loaders for Zero-Shot Memory Learning experiments.
"""

import os
from typing import Tuple, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


def _resolve_cifar_dir(data_dir: str, dataset_subdir: str) -> str:
    """Resolve effective data directory, preferring rawdata subdirectory if it exists."""
    candidate = os.path.join(data_dir, dataset_subdir)
    if os.path.exists(candidate):
        return candidate
    return data_dir


class ImageReconstructionDataset(Dataset):
    """
    Dataset wrapper for image reconstruction tasks.
    Returns (image, image) pairs for autoencoder-style training.
    """

    def __init__(self, dataset):
        """
        Initialize reconstruction dataset.

        Args:
            dataset: Base dataset (e.g., CIFAR10)
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, image  # Return (input, target) as same image


def get_cifar10_transforms(img_size: int = 32) -> transforms.Compose:
    """
    Get data transforms for CIFAR-10 dataset.

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    if img_size == 32:
        # Standard CIFAR-10 transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # Resize for different image sizes
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    return transform


def get_cifar10_dataloader(data_dir: str,
                          batch_size: int = 64,
                          img_size: int = 32,
                          num_workers: int = 4,
                          train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders for image reconstruction.

    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size for data loaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        train_split: Fraction of training data to use for training (rest for validation)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    transform = get_cifar10_transforms(img_size)

    # Resolve effective data directory (use cifar-10 subdirectory if available)
    effective_dir = _resolve_cifar_dir(data_dir, 'cifar-10')

    # Download and load CIFAR-10 training data
    full_train_dataset = CIFAR10(
        root=effective_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Split training data into train and validation
    total_size = len(full_train_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Wrap with reconstruction dataset
    train_recon_dataset = ImageReconstructionDataset(train_dataset)
    val_recon_dataset = ImageReconstructionDataset(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_recon_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_recon_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def get_cifar100_dataloader(data_dir: str,
                           batch_size: int = 64,
                           img_size: int = 32,
                           num_workers: int = 4,
                           train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-100 data loaders for image reconstruction.

    Args:
        data_dir: Directory to store/load CIFAR-100 data
        batch_size: Batch size for data loaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        train_split: Fraction of training data to use for training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    transform = get_cifar10_transforms(img_size)  # Same transforms work for CIFAR-100

    # Resolve effective data directory (use cifar-100 subdirectory if available)
    effective_dir = _resolve_cifar_dir(data_dir, 'cifar-100')

    # Download and load CIFAR-100 training data
    full_train_dataset = CIFAR100(
        root=effective_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Split training data into train and validation
    total_size = len(full_train_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Wrap with reconstruction dataset
    train_recon_dataset = ImageReconstructionDataset(train_dataset)
    val_recon_dataset = ImageReconstructionDataset(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_recon_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_recon_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def get_mnist_transforms(img_size: int = 32) -> transforms.Compose:
    """
    Get data transforms for MNIST dataset.
    MNIST is grayscale (1 channel), we convert to 3 channels for consistency.

    Args:
        img_size: Target image size

    Returns:
        Composed transforms
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1 channel to 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform


def get_mnist_dataloader(data_dir: str,
                         batch_size: int = 64,
                         img_size: int = 32,
                         num_workers: int = 4,
                         train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST data loaders for image reconstruction.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for data loaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        train_split: Fraction of training data to use for training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    transform = get_mnist_transforms(img_size)

    # Download and load MNIST training data
    full_train_dataset = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Split training data into train and validation
    total_size = len(full_train_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Wrap with reconstruction dataset
    train_recon_dataset = ImageReconstructionDataset(train_dataset)
    val_recon_dataset = ImageReconstructionDataset(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_recon_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_recon_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def get_image_reconstruction_datasets(dataset_name: str,
                                    data_dir: str,
                                    batch_size: int = 64,
                                    img_size: int = 32,
                                    num_workers: int = 4,
                                    train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """
    Get image reconstruction datasets by name.

    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'mnist')
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        train_split: Fraction of training data for training

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_dataloader(
            data_dir, batch_size, img_size, num_workers, train_split
        )
    elif dataset_name.lower() == 'cifar100':
        return get_cifar100_dataloader(
            data_dir, batch_size, img_size, num_workers, train_split
        )
    elif dataset_name.lower() == 'mnist':
        return get_mnist_dataloader(
            data_dir, batch_size, img_size, num_workers, train_split
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: ['cifar10', 'cifar100', 'mnist']")


def get_test_dataloader(dataset_name: str,
                       data_dir: str,
                       batch_size: int = 64,
                       img_size: int = 32,
                       num_workers: int = 4) -> DataLoader:
    """
    Get test data loader for evaluation.

    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'mnist')
        data_dir: Directory containing the data
        batch_size: Batch size for data loader
        img_size: Target image size
        num_workers: Number of worker processes for data loading

    Returns:
        Test data loader
    """
    # Load test dataset based on dataset name
    if dataset_name.lower() == 'cifar10':
        transform = get_cifar10_transforms(img_size)
        test_dataset = CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        transform = get_cifar10_transforms(img_size)
        test_dataset = CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'mnist':
        transform = get_mnist_transforms(img_size)
        test_dataset = MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: ['cifar10', 'cifar100', 'mnist']")

    # Wrap with reconstruction dataset
    test_recon_dataset = ImageReconstructionDataset(test_dataset)

    # Create test data loader
    test_loader = DataLoader(
        test_recon_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return test_loader


def create_custom_dataloader(images: torch.Tensor,
                             batch_size: int = 64,
                             shuffle: bool = True,
                             num_workers: int = 4) -> DataLoader:
    """
    Create data loader from custom tensor data.

    Args:
        images: Tensor of images (N, C, H, W)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        Data loader for custom data
    """
    dataset = ImageReconstructionDataset(images)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with dataset information
    """
    info = {
        'cifar10': {
            'num_classes': 10,
            'img_size': 32,
            'num_channels': 3,
            'train_size': 50000,
            'test_size': 10000,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        },
        'cifar100': {
            'num_classes': 100,
            'img_size': 32,
            'num_channels': 3,
            'train_size': 50000,
            'test_size': 10000,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        },
        'mnist': {
            'num_classes': 10,
            'img_size': 28,  # Original size, will be resized to 32
            'num_channels': 1,  # Original channels, converted to 3
            'train_size': 60000,
            'test_size': 10000,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }
    }

    return info.get(dataset_name.lower(), {})