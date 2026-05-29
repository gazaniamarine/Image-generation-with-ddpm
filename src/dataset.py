"""
Dataset module for loading and preprocessing CIFAR-10.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path


class CIFAR10Dataset(Dataset):
    """
    Wrapper around CIFAR-10 dataset with normalization to [-1, 1].
    """
    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        image_size: int = 32,
        normalize_to: str = "[-1, 1]"
    ):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            root: Root directory for dataset
            train: Load training set if True, test set otherwise
            download: Download dataset if not found
            image_size: Size to resize images to (default: 32)
            normalize_to: Normalize to [-1, 1] or [0, 1]
        """
        self.root = root
        self.train = train
        self.image_size = image_size
        self.normalize_to = normalize_to
        
        # Create dataset directory
        Path(root).mkdir(parents=True, exist_ok=True)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        
        # Load CIFAR-10
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get normalized image tensor.
        
        Args:
            idx: Index of image to retrieve
            
        Returns:
            Normalized image tensor with values in [-1, 1]
        """
        image, _ = self.dataset[idx]
        
        # Normalize to [-1, 1]
        if self.normalize_to == "[-1, 1]":
            image = 2 * image - 1  # Convert from [0, 1] to [-1, 1]
        
        return image


def get_data_loaders(
    batch_size: int = 128,
    image_size: int = 32,
    num_workers: int = 4,
    normalize_to: str = "[-1, 1]",
    data_root: str = "data",
) -> tuple:
    """
    Get train and test DataLoaders for CIFAR-10.
    
    Args:
        batch_size: Batch size for DataLoader
        image_size: Size to resize images to
        num_workers: Number of workers for DataLoader
        normalize_to: Normalize to [-1, 1] or [0, 1]
        data_root: Root directory for dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = CIFAR10Dataset(
        root=data_root,
        train=True,
        download=True,
        image_size=image_size,
        normalize_to=normalize_to,
    )
    
    test_dataset = CIFAR10Dataset(
        root=data_root,
        train=False,
        download=True,
        image_size=image_size,
        normalize_to=normalize_to,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader
