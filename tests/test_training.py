"""Tests for training functionality."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from mnist_classifier.training.trainer import (
    PyTorchTrainer,
    XGBoostTrainer,
    create_pytorch_trainer,
    create_xgboost_trainer
)
from mnist_classifier.training.hyperparams import (
    HyperparameterGrid,
    HyperparameterSampler,
    HyperparameterManager,
    get_best_hyperparams_by_model
)


class TestPyTorchTrainer:
    """Test cases for PyTorch trainer."""
    
    def test_trainer_initialization(self, simple_mlp_model):
        """Test trainer initialization."""
        trainer = PyTorchTrainer(
            model=simple_mlp_model,
            device='cpu',
            save_dir='temp'
        )
        
        assert trainer.model == simple_mlp_model
        assert trainer.device == torch.device('cpu')
        assert trainer.save_dir == Path('temp')
        assert trainer.optimizer is None
        assert trainer.scheduler is None
    
    def test_setup_training(self, simple_mlp_model, sample_hyperparams):
        """Test training setup."""
        trainer = PyTorchTrainer(
            model=simple_mlp_model,
            device='cpu'
        )
        
        # Setup training
        trainer.setup_training(
            lr=sample_hyperparams['lr'],
            weight_decay=sample_hyperparams['weight_decay'],
            optimizer_type='adam',
            scheduler_type='cosine',
            epochs=sample_hyperparams['epochs']
        )
        
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    
    def test_train_epoch(self, simple_mlp_model):
        """Test single epoch training."""
        trainer = PyTorchTrainer(
            model=simple_mlp_model,
            device='cpu'
        )
        
        trainer.setup_training(lr=0.01, epochs=1)
        
        # Create mock data loader
        batch_size = 8
        data = [(torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,))) for _ in range(3)]
        
        # Train one epoch
        metrics = trainer.train_epoch(data, epoch=1)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_validate_epoch(self, simple_mlp_model):
        """Test validation epoch."""
        trainer = PyTorchTrainer(
            model=simple_mlp_model,
            device='cpu'
        )
        
        trainer.setup_training(lr=0.01, epochs=1)
        
        # Create mock validation data
        batch_size = 8
        val_data = [(torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,))) for _ in range(2)]
        
        # Validate
        metrics = trainer.validate_epoch(val_data, epoch=1)
        
        assert 'val_loss' in metrics
        assert 'val_accuracy' in metrics
        assert isinstance(metrics['val_loss'], float)
        assert isinstance(metrics['val_accuracy'], float)
    
    @patch('mnist_classifier.training.trainer.mlflow')
    def test_full_training_loop(self, mock_mlflow, simple_mlp_model):
        """Test complete training loop."""
        trainer = PyTorchTrainer(
            model=simple_mlp_model,
            device='cpu'
        )
        
        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Create mock data
        batch_size = 8
        train_data = [(torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,))) for _ in range(3)]
        val_data = [(torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,))) for _ in range(2)]
        
        # Train
        history = trainer.train(
            train_loader=train_data,
            val_loader=val_data,
            epochs=2,
            lr=0.01,
            model_name='test_model'
        )
        
        assert 'train_loss' in history
        assert 'train_accuracy' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['train_loss']) == 2  # 2 epochs
    
    def test_save_and_load_model(self, simple_mlp_model, temp_models_dir):
        """Test model saving and loading."""
        trainer = PyTorchTrainer(
            model=simple_mlp_model,
            device='cpu',
            save_dir=temp_models_dir
        )
        
        # Save model
        save_path = trainer.save_model('test_model', accuracy=0.95)
        
        assert save_path.exists()
        assert save_path.suffix == '.pth'
        
        # Load and verify
        checkpoint = torch.load(save_path)
        assert 'model_state_dict' in checkpoint
        assert 'accuracy' in checkpoint
        assert checkpoint['accuracy'] == 0.95


class TestXGBoostTrainer:
    """Test cases for XGBoost trainer."""
    
    def test_xgboost_trainer_initialization(self):
        """Test XGBoost trainer initialization."""
        trainer = XGBoostTrainer(save_dir='temp')
        
        assert trainer.save_dir == Path('temp')
        assert trainer.model is None
        assert trainer.feature_importance_ is None
    
    def test_xgboost_training(self, sample_mnist_batch, sample_labels):
        """Test XGBoost training."""
        trainer = XGBoostTrainer()
        
        # Prepare data
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        X_val = X_train[:10]
        y_val = y_train[:10]
        
        # Train
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={
                'n_estimators': 5,
                'max_depth': 3,
                'learning_rate': 0.3
            },
            model_name='test_xgb'
        )
        
        assert trainer.model is not None
        assert 'train_accuracy' in history
        assert 'val_accuracy' in history
        assert trainer.feature_importance_ is not None
    
    def test_xgboost_prediction(self, sample_mnist_batch, sample_labels):
        """Test XGBoost prediction."""
        trainer = XGBoostTrainer()
        
        # Train first
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            params={
                'n_estimators': 5,
                'max_depth': 3,
                'learning_rate': 0.3
            },
            model_name='test_xgb'
        )
        
        # Test prediction
        X_test = X_train[:5]
        predictions = trainer.predict(X_test)
        probabilities = trainer.predict_proba(X_test)
        
        assert len(predictions) == 5
        assert probabilities.shape == (5, 10)
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
    
    @patch('mnist_classifier.training.trainer.mlflow')
    def test_xgboost_mlflow_logging(self, mock_mlflow, sample_mnist_batch, sample_labels):
        """Test MLflow logging in XGBoost trainer."""
        trainer = XGBoostTrainer()
        
        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Train
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            params={'n_estimators': 5, 'max_depth': 3},
            model_name='test_xgb'
        )
        
        # Verify MLflow was called
        assert mock_mlflow.start_run.called
    
    def test_xgboost_save_model(self, sample_mnist_batch, sample_labels, temp_models_dir):
        """Test XGBoost model saving."""
        trainer = XGBoostTrainer(save_dir=temp_models_dir)
        
        # Train and save
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            params={'n_estimators': 5, 'max_depth': 3},
            model_name='test_xgb'
        )
        
        save_path = trainer.save_model('test_xgb', accuracy=0.9)
        
        assert save_path.exists()
        assert save_path.suffix == '.pkl'


class TestHyperparameterGrid:
    """Test hyperparameter grid functionality."""
    
    def test_mlp_hyperparams(self):
        """Test MLP hyperparameter generation."""
        hyperparams = HyperparameterGrid.get_mlp_hyperparams()
        
        assert len(hyperparams) > 0
        assert all('variant' in hp for hp in hyperparams)
        assert all('lr' in hp for hp in hyperparams)
        assert all('weight_decay' in hp for hp in hyperparams)
    
    def test_mlp_focused_hyperparams(self):
        """Test focused MLP hyperparameters."""
        hyperparams = HyperparameterGrid.get_mlp_focused_hyperparams()
        
        assert len(hyperparams) > 0
        assert len(hyperparams) < 50  # Should be focused, not exhaustive
        
        # Check structure
        for hp in hyperparams:
            assert 'variant' in hp
            assert hp['variant'] in ['small', 'medium', 'large']
            assert 'lr' in hp
            assert 'epochs' in hp
    
    def test_cnn_hyperparams(self):
        """Test CNN hyperparameter generation."""
        hyperparams = HyperparameterGrid.get_cnn_hyperparams()
        
        assert len(hyperparams) > 0
        assert all('variant' in hp for hp in hyperparams)
        assert all('lr' in hp for hp in hyperparams)
        assert all('dropout_rate' in hp for hp in hyperparams)
    
    def test_xgboost_hyperparams(self):
        """Test XGBoost hyperparameter generation."""
        hyperparams = HyperparameterGrid.get_xgboost_hyperparams()
        
        assert len(hyperparams) > 0
        assert all('n_estimators' in hp for hp in hyperparams)
        assert all('max_depth' in hp for hp in hyperparams)
        assert all('learning_rate' in hp for hp in hyperparams)


class TestHyperparameterSampler:
    """Test hyperparameter sampling functionality."""
    
    def test_sample_mlp_hyperparams(self):
        """Test MLP hyperparameter sampling."""
        hyperparams = HyperparameterSampler.sample_mlp_hyperparams(
            n_samples=5, 
            random_state=42
        )
        
        assert len(hyperparams) == 5
        
        for hp in hyperparams:
            assert 'variant' in hp
            assert 'lr' in hp
            assert 0.0001 <= hp['lr'] <= 0.01  # Should be clipped
            assert 'weight_decay' in hp
            assert 'dropout_rate' in hp
    
    def test_sample_cnn_hyperparams(self):
        """Test CNN hyperparameter sampling."""
        hyperparams = HyperparameterSampler.sample_cnn_hyperparams(
            n_samples=3,
            random_state=42
        )
        
        assert len(hyperparams) == 3
        
        for hp in hyperparams:
            assert 'variant' in hp
            assert hp['variant'] in ['simple', 'medium', 'deep', 'modern']
            assert 'lr' in hp
            assert 'epochs' in hp
    
    def test_sample_xgboost_hyperparams(self):
        """Test XGBoost hyperparameter sampling."""
        hyperparams = HyperparameterSampler.sample_xgboost_hyperparams(
            n_samples=4,
            random_state=42
        )
        
        assert len(hyperparams) == 4
        
        for hp in hyperparams:
            assert 'n_estimators' in hp
            assert 'max_depth' in hp
            assert 'learning_rate' in hp
            assert 0.01 <= hp['learning_rate'] <= 0.5  # Should be clipped


class TestHyperparameterManager:
    """Test hyperparameter manager functionality."""
    
    @pytest.mark.parametrize("search_type", ["grid", "focused", "random"])
    def test_manager_initialization(self, search_type):
        """Test manager initialization with different search types."""
        manager = HyperparameterManager(search_type=search_type)
        
        assert manager.search_type == search_type
        assert hasattr(manager, 'grid')
        assert hasattr(manager, 'sampler')
    
    @pytest.mark.parametrize("model_type", ["mlp", "cnn", "xgboost"])
    def test_get_hyperparameters(self, model_type):
        """Test getting hyperparameters for different model types."""
        manager = HyperparameterManager(search_type='focused')
        
        hyperparams = manager.get_hyperparameters(model_type)
        
        assert len(hyperparams) > 0
        assert all(isinstance(hp, dict) for hp in hyperparams)
    
    def test_random_search(self):
        """Test random search functionality."""
        manager = HyperparameterManager(search_type='random')
        
        hyperparams = manager.get_hyperparameters(
            'mlp', 
            n_samples=5, 
            random_state=42
        )
        
        assert len(hyperparams) == 5
    
    def test_experiment_summary(self):
        """Test experiment summary generation."""
        manager = HyperparameterManager(search_type='focused')
        
        summary = manager.get_experiment_summary('mlp')
        
        assert 'model_type' in summary
        assert 'search_type' in summary
        assert 'total_configurations' in summary
        assert summary['model_type'] == 'mlp'
        assert summary['search_type'] == 'focused'
    
    def test_invalid_search_type(self):
        """Test handling of invalid search type."""
        with pytest.raises(ValueError):
            manager = HyperparameterManager(search_type='invalid')
            manager.get_hyperparameters('mlp')
    
    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        manager = HyperparameterManager(search_type='focused')
        
        with pytest.raises(ValueError):
            manager.get_hyperparameters('invalid_model')


class TestTrainerFactories:
    """Test trainer factory functions."""
    
    def test_create_pytorch_trainer(self, simple_mlp_model):
        """Test PyTorch trainer factory."""
        trainer = create_pytorch_trainer(
            model=simple_mlp_model,
            device='cpu',
            save_dir='temp'
        )
        
        assert isinstance(trainer, PyTorchTrainer)
        assert trainer.model == simple_mlp_model
    
    def test_create_xgboost_trainer(self):
        """Test XGBoost trainer factory."""
        trainer = create_xgboost_trainer(save_dir='temp')
        
        assert isinstance(trainer, XGBoostTrainer)
        assert trainer.save_dir == Path('temp')


class TestBestHyperparams:
    """Test best hyperparameters functionality."""
    
    def test_get_best_hyperparams_by_model(self):
        """Test getting best hyperparameters."""
        best_params = get_best_hyperparams_by_model()
        
        assert 'mlp' in best_params
        assert 'cnn' in best_params
        assert 'xgboost' in best_params
        
        # Check MLP params
        mlp_params = best_params['mlp']
        assert 'variant' in mlp_params
        assert 'lr' in mlp_params
        assert 'epochs' in mlp_params
        
        # Check CNN params
        cnn_params = best_params['cnn']
        assert 'variant' in cnn_params
        assert 'lr' in cnn_params
        
        # Check XGBoost params
        xgb_params = best_params['xgboost']
        assert 'n_estimators' in xgb_params
        assert 'max_depth' in xgb_params


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_end_to_end_pytorch_training(self, simple_mlp_model):
        """Test end-to-end PyTorch training workflow."""
        # Create trainer
        trainer = PyTorchTrainer(model=simple_mlp_model, device='cpu')
        
        # Get hyperparameters
        manager = HyperparameterManager(search_type='focused')
        hyperparams = manager.get_hyperparameters('mlp')
        best_hp = hyperparams[0]  # Use first config
        
        # Create mock data
        batch_size = 8
        train_data = [(torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,))) for _ in range(2)]
        val_data = [(torch.randn(batch_size, 784), torch.randint(0, 10, (batch_size,))) for _ in range(1)]
        
        # Train with hyperparameters
        history = trainer.train(
            train_loader=train_data,
            val_loader=val_data,
            epochs=2,
            lr=best_hp['lr'],
            weight_decay=best_hp['weight_decay'],
            model_name='integration_test'
        )
        
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
    
    @patch('mnist_classifier.training.trainer.mlflow')
    def test_end_to_end_xgboost_training(self, mock_mlflow, sample_mnist_batch, sample_labels):
        """Test end-to-end XGBoost training workflow."""
        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Create trainer
        trainer = XGBoostTrainer()
        
        # Get hyperparameters
        manager = HyperparameterManager(search_type='focused')
        hyperparams = manager.get_hyperparameters('xgboost')
        best_hp = hyperparams[0]
        
        # Prepare data
        X_train = sample_mnist_batch.reshape(len(sample_mnist_batch), -1)
        y_train = sample_labels
        
        # Train
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            params=best_hp,
            model_name='integration_test_xgb'
        )
        
        assert 'train_accuracy' in history
        assert trainer.model is not None