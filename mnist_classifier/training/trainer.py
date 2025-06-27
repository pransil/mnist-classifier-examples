"""Training loops and utilities for different model types."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import warnings

from ..utils.mlflow_utils import MLflowTracker, log_training_metrics, log_final_metrics
from ..models.mlp import MLP
from ..models.cnn import CNN
from ..models.xgboost_model import XGBoostClassifier
from ..data.preprocessor import MNISTPreprocessor


class PyTorchTrainer:
    """Trainer for PyTorch models (MLP and CNN)."""
    
    def __init__(self, model: Union[MLP, CNN], device: str = 'auto',
                 mlflow_tracker: Optional[MLflowTracker] = None):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cpu', 'cuda', 'auto')
            mlflow_tracker: MLflow tracker for logging
        """
        self.model = model
        self.mlflow_tracker = mlflow_tracker
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        print(f"Trainer initialized with device: {self.device}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, lr: float = 0.001, weight_decay: float = 1e-4,
              optimizer_type: str = 'adam', scheduler_type: Optional[str] = 'cosine',
              patience: int = 10, save_best: bool = True, 
              plot_every: int = 5, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the PyTorch model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            scheduler_type: Learning rate scheduler ('cosine', 'step', 'plateau', None)
            patience: Early stopping patience
            save_best: Whether to save best model state
            plot_every: Plot training curves every N epochs
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Setup scheduler
        scheduler = None
        if scheduler_type:
            if scheduler_type.lower() == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            elif scheduler_type.lower() == 'step':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
            elif scheduler_type.lower() == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience//2)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        early_stopping_counter = 0
        
        # Log hyperparameters
        if self.mlflow_tracker:
            hyperparams = {
                'model_type': self.model.__class__.__name__,
                'epochs': epochs,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'optimizer': optimizer_type,
                'scheduler': scheduler_type,
                'batch_size': train_loader.batch_size,
                'device': str(self.device)
            }
            
            # Add model-specific parameters
            if hasattr(self.model, 'get_model_info'):
                model_info = self.model.get_model_info()
                hyperparams.update({
                    'total_parameters': model_info['total_parameters'],
                    'trainable_parameters': model_info['trainable_parameters']
                })
            
            self.mlflow_tracker.log_params(hyperparams)
        
        if verbose:
            print(f"Training {self.model.__class__.__name__} for {epochs} epochs...")
            print(f"Optimizer: {optimizer_type}, LR: {lr}, Scheduler: {scheduler_type}")
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                if scheduler_type.lower() == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Store history
            current_lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log to MLflow
            if self.mlflow_tracker:
                log_training_metrics(self.mlflow_tracker, epoch, train_loss, train_acc, val_loss, val_acc)
                self.mlflow_tracker.log_metric('learning_rate', current_lr, step=epoch)
            
            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                early_stopping_counter = 0
                
                if save_best:
                    self.best_model_state = self.model.state_dict().copy()
            else:
                early_stopping_counter += 1
            
            epoch_time = time.time() - start_time
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Plot training curves
            if (epoch + 1) % plot_every == 0:
                self._plot_training_curves(epoch + 1)
            
            # Early stopping
            if early_stopping_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            self.epoch = epoch + 1
        
        # Load best model if saved
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"Loaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        # Final plot
        self._plot_training_curves(self.epoch, save_plot=True)
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                    criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _plot_training_curves(self, epoch: int, save_plot: bool = False):
        """Plot training curves."""
        if not self.history['train_loss']:
            return
        
        from datetime import datetime
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Add main title showing model being tested
        model_name = self.model.__class__.__name__
        fig.suptitle(f'{model_name} Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs_range, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs_range, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs_range, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs_range, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs_range, self.history['learning_rates'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Loss vs Accuracy scatter
        ax4.scatter(self.history['train_loss'], self.history['train_acc'], alpha=0.6, label='Train')
        ax4.scatter(self.history['val_loss'], self.history['val_acc'], alpha=0.6, label='Val')
        ax4.set_title('Loss vs Accuracy')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for main title
        
        # Always save plot with timestamp
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
        
        if save_plot:
            # Final plot with epoch number
            plot_path = plots_dir / f"{model_name}_training_curves_epoch_{epoch}_{timestamp}.png"
        else:
            # Intermediate plot
            plot_path = plots_dir / f"{model_name}_training_progress_{timestamp}.png"
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_path}")
        
        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifact(str(plot_path), "plots")
        
        # Show plot without blocking execution
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure plot displays
        plt.close()


class XGBoostTrainer:
    """Trainer for XGBoost models."""
    
    def __init__(self, model: XGBoostClassifier, preprocessor: Optional[MNISTPreprocessor] = None,
                 mlflow_tracker: Optional[MLflowTracker] = None):
        """
        Initialize XGBoost trainer.
        
        Args:
            model: XGBoost classifier
            preprocessor: Data preprocessor
            mlflow_tracker: MLflow tracker for logging
        """
        self.model = model
        self.preprocessor = preprocessor
        self.mlflow_tracker = mlflow_tracker
        self.training_time = 0.0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print training progress
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        
        # Preprocess data if preprocessor is provided
        if self.preprocessor:
            if verbose:
                print("Preprocessing training data...")
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_val_processed = self.preprocessor.transform(X_val)
        else:
            X_train_processed = X_train
            X_val_processed = X_val
        
        # Log hyperparameters
        if self.mlflow_tracker:
            model_info = self.model.get_model_info()
            hyperparams = model_info['parameters'].copy()
            hyperparams.update({
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'num_features': X_train.shape[1],
                'preprocessor': self.preprocessor.__class__.__name__ if self.preprocessor else None
            })
            self.mlflow_tracker.log_params(hyperparams)
        
        # Train model
        self.model.fit(X_train_processed, y_train, X_val_processed, y_val, verbose=verbose)
        
        self.training_time = time.time() - start_time
        
        # Get training history
        history = self.model.get_training_history()
        
        # Plot training curves
        self._plot_training_curves(history)
        
        # Log final metrics
        if self.mlflow_tracker:
            train_metrics = self.model.evaluate(X_train_processed, y_train)
            val_metrics = self.model.evaluate(X_val_processed, y_val)
            
            final_metrics = {
                'train_accuracy': train_metrics['accuracy'],
                'train_log_loss': train_metrics['log_loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_log_loss': val_metrics['log_loss'],
                'training_time': self.training_time
            }
            
            log_final_metrics(self.mlflow_tracker, final_metrics)
        
        if verbose:
            print(f"XGBoost training completed in {self.training_time:.2f} seconds")
        
        return {
            'history': history,
            'training_time': self.training_time,
            'model_info': self.model.get_model_info()
        }
    
    def _plot_training_curves(self, history: Dict[str, List[float]]):
        """Plot XGBoost training curves."""
        if not history or 'train' not in history:
            return
        
        from datetime import datetime
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Add main title showing model being tested
        fig.suptitle('XGBoost Model Training Progress', fontsize=16, fontweight='bold')
        
        # Extract metrics
        train_metric = None
        val_metric = None
        
        for key, values in history.items():
            if 'train' in key:
                train_metric = values
                metric_name = key.split('_')[-1] if '_' in key else key
            elif 'val' in key:
                val_metric = values
        
        if train_metric:
            epochs = range(len(train_metric))
            
            # Training curves
            axes[0].plot(epochs, train_metric, 'b-', label=f'Training {metric_name}')
            if val_metric:
                axes[0].plot(epochs, val_metric, 'r-', label=f'Validation {metric_name}')
            
            axes[0].set_title(f'Training Progress')
            axes[0].set_xlabel('Boosting Round')
            axes[0].set_ylabel(metric_name.capitalize())
            axes[0].legend()
            axes[0].grid(True)
            
            # Feature importance (if available)
            feature_importance = self.model.get_feature_importance()
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
                
                features, importance = zip(*top_features)
                axes[1].barh(range(len(features)), importance)
                axes[1].set_yticks(range(len(features)))
                axes[1].set_yticklabels([f'f{f}' for f in features])
                axes[1].set_title('Top 20 Feature Importance')
                axes[1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for main title
        
        # Always save plot with timestamp
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
        plot_path = plots_dir / f"XGBoost_training_curves_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_path}")
        
        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifact(str(plot_path), "plots")
        
        # Show plot without blocking execution
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure plot displays
        plt.close()


def create_trainer(model_type: str, model: Any, **kwargs) -> Union[PyTorchTrainer, XGBoostTrainer]:
    """
    Factory function to create appropriate trainer.
    
    Args:
        model_type: Type of model ('pytorch', 'xgboost')
        model: Model instance
        **kwargs: Additional arguments for trainer
        
    Returns:
        Configured trainer instance
    """
    if model_type.lower() == 'pytorch':
        return PyTorchTrainer(model, **kwargs)
    elif model_type.lower() == 'xgboost':
        return XGBoostTrainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")