"""Tests for MNIST classifier models."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb

from mnist_classifier.models.mlp import (
    MLPClassifier, 
    get_mlp_variant,
    create_mlp_model
)
from mnist_classifier.models.cnn import (
    CNNClassifier,
    get_cnn_variant, 
    create_cnn_model
)
from mnist_classifier.models.xgboost_model import (
    XGBoostClassifier,
    get_xgboost_variant,
    create_xgboost_model
)


class TestMLPClassifier:
    """Test cases for MLP model."""
    
    def test_mlp_initialization(self):
        """Test MLP model initialization."""
        model = MLPClassifier(
            input_size=784,
            hidden_sizes=[128, 64],
            num_classes=10,
            dropout_rate=0.3
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'layers')
        assert hasattr(model, 'dropout')
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = MLPClassifier(
            input_size=784,
            hidden_sizes=[128, 64],
            num_classes=10
        )
        
        # Test with batch of flattened MNIST images
        batch_size = 16
        input_tensor = torch.randn(batch_size, 784)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("variant", ["small", "medium", "large", "xlarge", "deep"])
    def test_mlp_variants(self, variant):
        """Test different MLP variants."""
        config = get_mlp_variant(variant)
        model = MLPClassifier(**config)
        
        # Test forward pass
        input_tensor = torch.randn(8, 784)
        output = model(input_tensor)
        
        assert output.shape == (8, 10)
        assert isinstance(model, nn.Module)
    
    def test_mlp_parameter_count(self):
        """Test parameter counting for MLP."""
        model = MLPClassifier(
            input_size=784,
            hidden_sizes=[100],
            num_classes=10
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Expected: (784*100 + 100) + (100*10 + 10) = 78400 + 100 + 1000 + 10 = 79510
        expected_params = 784 * 100 + 100 + 100 * 10 + 10
        assert param_count == expected_params
    
    def test_create_mlp_model_function(self):
        """Test MLP model creation function."""
        model = create_mlp_model(variant="medium")
        
        assert isinstance(model, MLPClassifier)
        
        # Test forward pass
        input_tensor = torch.randn(4, 784)
        output = model(input_tensor)
        assert output.shape == (4, 10)


class TestCNNClassifier:
    """Test cases for CNN model."""
    
    def test_cnn_initialization(self):
        """Test CNN model initialization."""
        model = CNNClassifier(
            num_classes=10,
            channels=[16, 32],
            kernel_sizes=[3, 3],
            hidden_size=128,
            dropout_rate=0.25
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'conv_layers')
        assert hasattr(model, 'classifier')
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        model = CNNClassifier(
            num_classes=10,
            channels=[16, 32],
            kernel_sizes=[3, 3],
            hidden_size=128
        )
        
        # Test with batch of MNIST images (batch_size, channels, height, width)
        batch_size = 8
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("variant", ["simple", "medium", "deep", "modern", "large", "efficient"])
    def test_cnn_variants(self, variant):
        """Test different CNN variants."""
        config = get_cnn_variant(variant)
        model = CNNClassifier(**config)
        
        # Test forward pass
        input_tensor = torch.randn(4, 1, 28, 28)
        output = model(input_tensor)
        
        assert output.shape == (4, 10)
        assert isinstance(model, nn.Module)
    
    def test_cnn_with_batch_norm(self):
        """Test CNN with batch normalization."""
        model = CNNClassifier(
            num_classes=10,
            channels=[16, 32],
            kernel_sizes=[3, 3],
            use_batch_norm=True
        )
        
        # Check that batch norm layers exist
        found_batch_norm = False
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                found_batch_norm = True
                break
        
        assert found_batch_norm
        
        # Test forward pass
        input_tensor = torch.randn(4, 1, 28, 28)
        output = model(input_tensor)
        assert output.shape == (4, 10)
    
    def test_create_cnn_model_function(self):
        """Test CNN model creation function."""
        model = create_cnn_model(variant="simple")
        
        assert isinstance(model, CNNClassifier)
        
        # Test forward pass
        input_tensor = torch.randn(2, 1, 28, 28)
        output = model(input_tensor)
        assert output.shape == (2, 10)


class TestXGBoostClassifier:
    """Test cases for XGBoost model."""
    
    def test_xgboost_initialization(self):
        """Test XGBoost model initialization."""
        model = XGBoostClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
        
        assert hasattr(model, 'params')
        assert hasattr(model, 'model')
        assert model.model is None  # Not trained yet
    
    def test_xgboost_training(self, sample_mnist_batch, sample_labels):
        """Test XGBoost training."""
        model = XGBoostClassifier(
            n_estimators=5,  # Small for testing
            max_depth=3,
            learning_rate=0.3
        )
        
        # Prepare data (flatten images)
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        
        # Train model
        model.train(X_train, y_train)
        
        assert model.model is not None
        assert hasattr(model.model, 'predict')
    
    def test_xgboost_prediction(self, sample_mnist_batch, sample_labels):
        """Test XGBoost prediction."""
        model = XGBoostClassifier(
            n_estimators=5,
            max_depth=3,
            learning_rate=0.3
        )
        
        # Prepare and train
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        model.train(X_train, y_train)
        
        # Test prediction
        X_test = X_train[:5]  # Use subset for testing
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == 5
        assert all(0 <= p <= 9 for p in predictions)
        assert probabilities.shape == (5, 10)
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
    
    @pytest.mark.parametrize("variant", ["fast", "balanced", "deep", "conservative", "aggressive"])
    def test_xgboost_variants(self, variant):
        """Test different XGBoost variants."""
        config = get_xgboost_variant(variant)
        model = XGBoostClassifier(**config)
        
        assert isinstance(model, XGBoostClassifier)
        assert model.params['n_estimators'] > 0
        assert model.params['max_depth'] > 0
        assert model.params['learning_rate'] > 0
    
    def test_xgboost_early_stopping(self, sample_mnist_batch, sample_labels):
        """Test XGBoost with early stopping."""
        model = XGBoostClassifier(
            n_estimators=100,  # Large number
            max_depth=3,
            learning_rate=0.1,
            early_stopping_rounds=5
        )
        
        # Prepare data
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        
        # Create validation set (using part of training data for testing)
        split_idx = len(X_train) // 2
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # Train with validation
        model.train(X_train, y_train, X_val, y_val)
        
        assert model.model is not None
    
    def test_create_xgboost_model_function(self):
        """Test XGBoost model creation function."""
        model = create_xgboost_model(variant="fast")
        
        assert isinstance(model, XGBoostClassifier)
        assert model.params['n_estimators'] > 0


class TestModelComparisons:
    """Test model comparisons and edge cases."""
    
    def test_model_output_consistency(self):
        """Test that all models produce consistent output shapes."""
        batch_size = 4
        
        # Create models
        mlp = create_mlp_model("small")
        cnn = create_cnn_model("simple")
        
        # Test MLP
        mlp_input = torch.randn(batch_size, 784)
        mlp_output = mlp(mlp_input)
        assert mlp_output.shape == (batch_size, 10)
        
        # Test CNN
        cnn_input = torch.randn(batch_size, 1, 28, 28)
        cnn_output = cnn(cnn_input)
        assert cnn_output.shape == (batch_size, 10)
    
    def test_model_gradients(self):
        """Test that models can compute gradients."""
        mlp = create_mlp_model("small")
        mlp.train()
        
        input_tensor = torch.randn(2, 784, requires_grad=True)
        output = mlp(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in mlp.parameters():
            assert param.grad is not None
    
    def test_model_serialization(self, temp_models_dir):
        """Test model saving and loading."""
        model = create_mlp_model("small")
        
        # Save model
        save_path = f"{temp_models_dir}/test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = create_mlp_model("small")
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Test that outputs are the same
        input_tensor = torch.randn(2, 784)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = loaded_model(input_tensor)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_invalid_configurations(self):
        """Test handling of invalid model configurations."""
        # Test MLP with empty hidden layers
        with pytest.raises((ValueError, TypeError)):
            MLPClassifier(input_size=784, hidden_sizes=[], num_classes=10)
        
        # Test CNN with mismatched channels and kernels
        with pytest.raises((ValueError, IndexError)):
            CNNClassifier(
                channels=[16, 32, 64],
                kernel_sizes=[3, 3],  # Missing one kernel size
                num_classes=10
            )
        
        # Test XGBoost with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            XGBoostClassifier(n_estimators=-1)  # Negative estimators


class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_inference_speed(self):
        """Test basic inference speed (not actual benchmarking)."""
        import time
        
        model = create_mlp_model("small")
        model.eval()
        
        input_tensor = torch.randn(32, 784)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Should be reasonably fast (less than 10ms for small model)
        assert avg_time < 0.01
        assert output.shape == (32, 10)
    
    def test_memory_usage(self):
        """Test basic memory usage characteristics."""
        model = create_cnn_model("simple")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable for MNIST (less than 1M parameters)
        assert param_count < 1_000_000
        
        # Test memory during forward pass
        input_tensor = torch.randn(64, 1, 28, 28)
        output = model(input_tensor)
        
        assert output.shape == (64, 10)
        
        # Cleanup
        del input_tensor, output