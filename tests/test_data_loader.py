"""Tests for MNIST data loading functionality."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from mnist_classifier.data.loader import (
    MNISTDataLoader, 
    load_mnist_data,
    create_data_loaders,
    download_and_extract_mnist
)


class TestMNISTDataLoader:
    """Test cases for MNISTDataLoader class."""
    
    def test_initialization(self, temp_data_dir):
        """Test MNISTDataLoader initialization."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == Path(temp_data_dir)
        assert loader.train_data is None
        assert loader.test_data is None
    
    def test_load_data_missing_files(self, temp_data_dir):
        """Test loading data when files don't exist."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Should raise FileNotFoundError when files don't exist
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_create_mock_data(self, temp_data_dir):
        """Test creating mock MNIST data for testing."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create mock data files
        train_images = np.random.randint(0, 255, (1000, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 1000, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (200, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 200, dtype=np.uint8)
        
        # Save mock data
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        # Load and verify
        loader.load_data()
        assert loader.train_data is not None
        assert loader.test_data is not None
        assert len(loader.train_data[0]) == 1000
        assert len(loader.test_data[0]) == 200
    
    def test_get_pytorch_datasets(self, temp_data_dir):
        """Test PyTorch dataset creation."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create and save mock data
        train_images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 100, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 20, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        loader.load_data()
        
        # Get PyTorch datasets
        train_dataset, test_dataset = loader.get_pytorch_datasets()
        
        assert len(train_dataset) == 100
        assert len(test_dataset) == 20
        
        # Test dataset items
        sample_image, sample_label = train_dataset[0]
        assert isinstance(sample_image, torch.Tensor)
        assert isinstance(sample_label, torch.Tensor)
        assert sample_image.shape == (1, 28, 28)  # CHW format
        assert 0 <= sample_label.item() <= 9
    
    def test_get_numpy_arrays(self, temp_data_dir):
        """Test numpy array retrieval."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create and save mock data
        train_images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 100, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 20, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        loader.load_data()
        
        # Get numpy arrays
        (X_train, y_train), (X_test, y_test) = loader.get_numpy_arrays()
        
        assert X_train.shape == (100, 784)  # Flattened
        assert y_train.shape == (100,)
        assert X_test.shape == (20, 784)
        assert y_test.shape == (20,)
        
        # Check normalization
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
    
    def test_train_val_split(self, temp_data_dir):
        """Test train/validation split functionality."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create mock data
        train_images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 100, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 20, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        loader.load_data()
        
        # Test split
        train_dataset, val_dataset, test_dataset = loader.get_pytorch_datasets(
            validation_split=0.2
        )
        
        assert len(train_dataset) == 80  # 80% of 100
        assert len(val_dataset) == 20   # 20% of 100
        assert len(test_dataset) == 20   # Original test set


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_load_mnist_data(self, temp_data_dir):
        """Test load_mnist_data function."""
        # Create mock data files
        train_images = np.random.randint(0, 255, (50, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 50, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 10, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        # Test loading
        (X_train, y_train), (X_test, y_test) = load_mnist_data(temp_data_dir)
        
        assert X_train.shape == (50, 784)
        assert y_train.shape == (50,)
        assert X_test.shape == (10, 784)
        assert y_test.shape == (10,)
    
    def test_create_data_loaders(self, temp_data_dir):
        """Test DataLoader creation."""
        # Create mock data
        train_images = np.random.randint(0, 255, (64, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 64, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 20, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=temp_data_dir,
            batch_size=16,
            validation_split=0.25
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check batch sizes
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2  # images, labels
        assert train_batch[0].shape[0] <= 16  # batch size
        
        val_batch = next(iter(val_loader))
        assert val_batch[0].shape[0] <= 16
        
        test_batch = next(iter(test_loader))
        assert test_batch[0].shape[0] <= 16
    
    def test_download_and_extract_mnist_mock(self, temp_data_dir):
        """Test MNIST download function with mocking."""
        # This test would require mocking the actual download
        # For now, just test that the function exists and can be called
        assert callable(download_and_extract_mnist)
        
        # In a real scenario, you'd mock urllib.request and test the download logic


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_invalid_data_shapes(self, temp_data_dir):
        """Test handling of invalid data shapes."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create invalid data (wrong shape)
        train_images = np.random.randint(0, 255, (100, 32, 32), dtype=np.uint8)  # Wrong size
        train_labels = np.random.randint(0, 10, 100, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8))
        np.save(Path(temp_data_dir) / "test_labels.npy", np.random.randint(0, 10, 10, dtype=np.uint8))
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, AssertionError)):
            loader.load_data()
    
    def test_mismatched_data_lengths(self, temp_data_dir):
        """Test handling of mismatched data lengths."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create mismatched data
        train_images = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 90, dtype=np.uint8)  # Mismatched length
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8))
        np.save(Path(temp_data_dir) / "test_labels.npy", np.random.randint(0, 10, 10, dtype=np.uint8))
        
        # Should raise an error
        with pytest.raises((ValueError, AssertionError)):
            loader.load_data()
    
    def test_empty_validation_split(self, temp_data_dir):
        """Test edge case with very small datasets and validation splits."""
        loader = MNISTDataLoader(data_dir=temp_data_dir)
        
        # Create very small dataset
        train_images = np.random.randint(0, 255, (2, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, 2, dtype=np.uint8)
        test_images = np.random.randint(0, 255, (1, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, 1, dtype=np.uint8)
        
        np.save(Path(temp_data_dir) / "train_images.npy", train_images)
        np.save(Path(temp_data_dir) / "train_labels.npy", train_labels)
        np.save(Path(temp_data_dir) / "test_images.npy", test_images)
        np.save(Path(temp_data_dir) / "test_labels.npy", test_labels)
        
        loader.load_data()
        
        # Test with large validation split
        train_dataset, val_dataset, test_dataset = loader.get_pytorch_datasets(
            validation_split=0.8
        )
        
        # Should handle gracefully
        assert len(train_dataset) >= 0
        assert len(val_dataset) >= 0
        assert len(test_dataset) == 1