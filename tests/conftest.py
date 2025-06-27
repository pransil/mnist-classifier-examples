"""Pytest configuration and fixtures for MNIST Classifier tests."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn

# Test data fixtures

@pytest.fixture
def sample_mnist_image():
    """Generate a sample 28x28 MNIST-like image."""
    return np.random.rand(28, 28).astype(np.float32)

@pytest.fixture
def sample_mnist_batch():
    """Generate a batch of sample MNIST-like images."""
    return np.random.rand(32, 28, 28).astype(np.float32)

@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return np.random.randint(0, 10, size=32)

@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    return np.random.randint(0, 10, size=32)

@pytest.fixture
def sample_probabilities():
    """Generate sample prediction probabilities."""
    probs = np.random.rand(32, 10)
    # Normalize to sum to 1
    return probs / probs.sum(axis=1, keepdims=True)

# Model fixtures

@pytest.fixture
def simple_mlp_model():
    """Create a simple MLP model for testing."""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layers = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.flatten(x)
            return self.layers(x)
    
    model = SimpleMLP()
    model.eval()
    return model

@pytest.fixture
def simple_cnn_model():
    """Create a simple CNN model for testing."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x
    
    model = SimpleCNN()
    model.eval()
    return model

# Data fixtures

@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# Configuration fixtures

@pytest.fixture
def sample_hyperparams():
    """Sample hyperparameters for testing."""
    return {
        'lr': 0.001,
        'weight_decay': 1e-4,
        'dropout_rate': 0.3,
        'batch_size': 32,
        'epochs': 5
    }

@pytest.fixture
def sample_model_results():
    """Sample model evaluation results."""
    return {
        'model_1': {
            'accuracy': 0.95,
            'precision_macro': 0.94,
            'recall_macro': 0.93,
            'f1_macro': 0.94,
            'training_time': 120.5,
            'inference_time': 0.01
        },
        'model_2': {
            'accuracy': 0.97,
            'precision_macro': 0.96,
            'recall_macro': 0.96,
            'f1_macro': 0.96,
            'training_time': 200.3,
            'inference_time': 0.02
        }
    }

# API fixtures

@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient
    from mnist_classifier.api.main import create_app
    
    # Create app with temporary models directory
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(models_dir=tmpdir)
        client = TestClient(app)
        yield client

@pytest.fixture
def sample_api_request():
    """Sample API request data."""
    image_data = np.random.rand(28, 28).tolist()
    return {
        "image_data": image_data,
        "model_name": "test_model",
        "return_probabilities": True
    }

@pytest.fixture
def sample_batch_request():
    """Sample batch API request data."""
    images = [np.random.rand(28, 28).tolist() for _ in range(3)]
    return {
        "images": images,
        "model_name": "test_model",
        "return_probabilities": False
    }

# Utility fixtures

@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    class MockMLflow:
        def __init__(self):
            self.runs = {}
            self.current_run = None
        
        def start_run(self, run_name=None):
            self.current_run = run_name or "test_run"
            return self
        
        def end_run(self):
            self.current_run = None
        
        def log_metric(self, key, value, step=None):
            pass
        
        def log_param(self, key, value):
            pass
        
        def log_artifact(self, path):
            pass
    
    return MockMLflow()

# Setup and teardown

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set environment variables for testing
    os.environ['TESTING'] = '1'
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

# Parametrized fixtures

@pytest.fixture(params=['mlp', 'cnn', 'xgboost'])
def model_type(request):
    """Parametrized fixture for different model types."""
    return request.param

@pytest.fixture(params=[1, 5, 10, 32])
def batch_size(request):
    """Parametrized fixture for different batch sizes."""
    return request.param

@pytest.fixture(params=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
def metric_name(request):
    """Parametrized fixture for different metrics."""
    return request.param