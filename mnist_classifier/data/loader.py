"""MNIST data loading utilities."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os
from pathlib import Path


class MNISTDataLoader:
    """Handles MNIST dataset loading and preprocessing."""
    
    def __init__(self, data_root: str = "data", batch_size: int = 64, 
                 val_split: float = 0.1, random_seed: int = 42):
        """
        Initialize MNIST data loader.
        
        Args:
            data_root: Root directory for data storage
            batch_size: Batch size for data loaders
            val_split: Fraction of training data to use for validation
            random_seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Define transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Data augmentation transforms for training
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load MNIST train and test datasets."""
        print("Loading MNIST dataset...")
        
        # Load training data
        self.train_dataset_full = torchvision.datasets.MNIST(
            root=str(self.data_root), 
            train=True, 
            download=True, 
            transform=self.base_transform
        )
        
        # Load test data
        self.test_dataset = torchvision.datasets.MNIST(
            root=str(self.data_root), 
            train=False, 
            download=True, 
            transform=self.base_transform
        )
        
        # Split training data into train and validation
        train_size = int((1 - self.val_split) * len(self.train_dataset_full))
        val_size = len(self.train_dataset_full) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset_full, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        print(f"Dataset loaded: {len(self.train_dataset)} train, "
              f"{len(self.val_dataset)} validation, {len(self.test_dataset)} test samples")
    
    def get_pytorch_loaders(self, use_augmentation: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch data loaders for train, validation, and test sets.
        
        Args:
            use_augmentation: Whether to use data augmentation for training
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create augmented training dataset if requested
        if use_augmentation:
            train_dataset_aug = torchvision.datasets.MNIST(
                root=str(self.data_root), 
                train=True, 
                download=False, 
                transform=self.augment_transform
            )
            # Apply same split as before
            train_size = int((1 - self.val_split) * len(train_dataset_aug))
            val_size = len(train_dataset_aug) - train_size
            train_dataset, _ = random_split(
                train_dataset_aug, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.random_seed)
            )
        else:
            train_dataset = self.train_dataset
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_numpy_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                    np.ndarray, np.ndarray, np.ndarray]:
        """
        Get MNIST data as numpy arrays for sklearn/XGBoost models.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("Converting MNIST data to numpy arrays...")
        
        # Convert training data
        X_train = []
        y_train = []
        for idx in self.train_dataset.indices:
            data, target = self.train_dataset_full[idx]
            X_train.append(data.numpy().flatten())
            y_train.append(target)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Convert validation data
        X_val = []
        y_val = []
        for idx in self.val_dataset.indices:
            data, target = self.train_dataset_full[idx]
            X_val.append(data.numpy().flatten())
            y_val.append(target)
        
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Convert test data
        X_test = []
        y_test = []
        for data, target in self.test_dataset:
            X_test.append(data.numpy().flatten())
            y_test.append(target)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Numpy arrays created: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_sample_images(self, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample images for testing and visualization.
        
        Args:
            num_samples: Number of sample images to return
            
        Returns:
            Tuple of (images, labels)
        """
        indices = np.random.choice(len(self.test_dataset), num_samples, replace=False)
        images = []
        labels = []
        
        for idx in indices:
            img, label = self.test_dataset[idx]
            images.append(img)
            labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset."""
        train_labels = [self.train_dataset_full[i][1] for i in self.train_dataset.indices]
        val_labels = [self.train_dataset_full[i][1] for i in self.val_dataset.indices]
        test_labels = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        
        def count_classes(labels):
            unique, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique, counts))
        
        return {
            'train': count_classes(train_labels),
            'validation': count_classes(val_labels),
            'test': count_classes(test_labels)
        }


def create_data_loader(data_root: str = "data", batch_size: int = 64, 
                      val_split: float = 0.1, random_seed: int = 42) -> MNISTDataLoader:
    """
    Create and return a configured MNIST data loader.
    
    Args:
        data_root: Root directory for data storage
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured MNISTDataLoader instance
    """
    return MNISTDataLoader(data_root, batch_size, val_split, random_seed)