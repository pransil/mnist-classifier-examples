"""Data preprocessing utilities for MNIST classifier."""

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Union
import joblib
from pathlib import Path


class MNISTPreprocessor:
    """Handles data preprocessing for different model types."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler to use ("standard", "minmax", or "none")
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.pca = None
        self.is_fitted = False
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type != "none":
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, X: np.ndarray) -> 'MNISTPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training data to fit on
            
        Returns:
            Self for method chaining
        """
        if self.scaler is not None:
            print(f"Fitting {self.scaler_type} scaler on training data...")
            self.scaler.fit(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        X_transformed = X.copy()
        
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)
        
        if self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data (if possible).
        
        Args:
            X: Data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transforming")
        
        X_inverse = X.copy()
        
        if self.pca is not None:
            X_inverse = self.pca.inverse_transform(X_inverse)
        
        if self.scaler is not None:
            X_inverse = self.scaler.inverse_transform(X_inverse)
        
        return X_inverse
    
    def add_pca(self, n_components: Union[int, float] = 0.95, fit_data: Optional[np.ndarray] = None):
        """
        Add PCA dimensionality reduction to the preprocessing pipeline.
        
        Args:
            n_components: Number of components or variance ratio to keep
            fit_data: Data to fit PCA on (if None, must call fit separately)
        """
        print(f"Adding PCA with {n_components} components...")
        self.pca = PCA(n_components=n_components)
        
        if fit_data is not None:
            # Apply existing preprocessing first
            if self.scaler is not None:
                if not self.is_fitted:
                    self.scaler.fit(fit_data)
                processed_data = self.scaler.transform(fit_data)
            else:
                processed_data = fit_data
            
            self.pca.fit(processed_data)
            print(f"PCA fitted: {self.pca.n_components_} components explain "
                  f"{self.pca.explained_variance_ratio_.sum():.3f} of variance")
    
    def save(self, filepath: str):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        preprocessor_data = {
            'scaler_type': self.scaler_type,
            'scaler': self.scaler,
            'pca': self.pca,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MNISTPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        preprocessor_data = joblib.load(filepath)
        
        preprocessor = cls(preprocessor_data['scaler_type'])
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.pca = preprocessor_data['pca']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


class PyTorchTransforms:
    """PyTorch-specific data transforms for neural networks."""
    
    @staticmethod
    def get_basic_transforms() -> transforms.Compose:
        """Get basic transforms for MNIST data."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    @staticmethod
    def get_augmentation_transforms() -> transforms.Compose:
        """Get data augmentation transforms for training."""
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    @staticmethod
    def get_test_transforms() -> transforms.Compose:
        """Get transforms for test/inference data."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def create_preprocessor_for_model(model_type: str, **kwargs) -> MNISTPreprocessor:
    """
    Create a preprocessor configured for a specific model type.
    
    Args:
        model_type: Type of model ("mlp", "cnn", "xgboost")
        **kwargs: Additional arguments for preprocessor
        
    Returns:
        Configured preprocessor instance
    """
    if model_type.lower() == "mlp":
        return MNISTPreprocessor(scaler_type="standard", **kwargs)
    elif model_type.lower() == "cnn":
        return MNISTPreprocessor(scaler_type="none", **kwargs)  # CNNs handle normalization
    elif model_type.lower() == "xgboost":
        return MNISTPreprocessor(scaler_type="minmax", **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def normalize_pixel_values(images: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range.
    
    Args:
        images: Array of images with pixel values
        
    Returns:
        Normalized images
    """
    return images.astype(np.float32) / 255.0


def denormalize_pixel_values(images: np.ndarray) -> np.ndarray:
    """
    Denormalize pixel values back to [0, 255] range.
    
    Args:
        images: Array of normalized images
        
    Returns:
        Denormalized images
    """
    return (images * 255.0).astype(np.uint8)