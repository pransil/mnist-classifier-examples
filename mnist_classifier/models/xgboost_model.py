"""XGBoost classifier model implementation."""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple, List
import joblib
from pathlib import Path
import time


class XGBoostClassifier:
    """XGBoost classifier wrapper for MNIST digit classification."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.3,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 eval_metric: str = 'mlogloss',
                 early_stopping_rounds: Optional[int] = 10,
                 verbosity: int = 1):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed
            n_jobs: Number of parallel threads
            eval_metric: Evaluation metric
            early_stopping_rounds: Early stopping patience
            verbosity: Verbosity level
        """
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 10,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'eval_metric': eval_metric,
            'verbosity': verbosity
        }
        
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.training_history = {}
        self.is_fitted = False
        self.feature_importance = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'XGBoostClassifier':
        """
        Fit the XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        if verbose:
            print(f"Training XGBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        start_time = time.time()
        
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Set up evaluation sets
        eval_set = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_set.append((dval, 'val'))
        
        # Prepare training parameters (remove n_estimators from params, use num_boost_round)
        train_params = self.params.copy()
        num_boost_round = train_params.pop('n_estimators', 100)
        
        # Store evaluation results
        evals_result = {}
        
        # Train the model
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=eval_set,
            evals_result=evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=verbose
        )
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        # Store training history
        self.training_history = evals_result
        
        # Calculate feature importance
        self.feature_importance = self.model.get_score(importance_type='weight')
        
        if verbose:
            print(f"Training completed in {self.training_time:.2f} seconds")
            print(f"Best iteration: {self.model.best_iteration}")
            print(f"Best score: {self.model.best_score:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        dtest = xgb.DMatrix(X)
        probabilities = self.model.predict(dtest)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        
        # Calculate log loss manually
        epsilon = 1e-15  # Small value to avoid log(0)
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        log_loss = -np.mean(np.log(probabilities[np.arange(len(y)), y]))
        
        return {
            'accuracy': accuracy,
            'log_loss': log_loss,
            'error_rate': 1 - accuracy
        }
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Feature importance dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.get_score(importance_type=importance_type)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.training_history
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Save XGBoost model
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(model_path))
        
        # Save additional metadata
        metadata = {
            'params': self.params,
            'training_history': self.training_history,
            'training_time': self.training_time,
            'feature_importance': self.feature_importance
        }
        
        metadata_path = model_path.with_suffix('.metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'XGBoostClassifier':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        model_path = Path(filepath)
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = model_path.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.params = metadata.get('params', self.params)
            self.training_history = metadata.get('training_history', {})
            self.training_time = metadata.get('training_time', 0)
            self.feature_importance = metadata.get('feature_importance', {})
        
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_type': 'XGBoost',
            'parameters': self.params.copy(),
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info.update({
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score,
                'training_time': self.training_time,
                'num_features': len(self.feature_importance) if self.feature_importance else 0
            })
        
        return info


class XGBoostVariants:
    """Factory class for creating different XGBoost configurations."""
    
    @staticmethod
    def create_fast_xgb(random_state: int = 42) -> XGBoostClassifier:
        """Create a fast XGBoost for quick experiments."""
        return XGBoostClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            early_stopping_rounds=5
        )
    
    @staticmethod
    def create_balanced_xgb(random_state: int = 42) -> XGBoostClassifier:
        """Create a balanced XGBoost with good performance/speed trade-off."""
        return XGBoostClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            early_stopping_rounds=10
        )
    
    @staticmethod
    def create_deep_xgb(random_state: int = 42) -> XGBoostClassifier:
        """Create a deep XGBoost with more trees and regularization."""
        return XGBoostClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=random_state,
            early_stopping_rounds=15
        )
    
    @staticmethod
    def create_conservative_xgb(random_state: int = 42) -> XGBoostClassifier:
        """Create a conservative XGBoost with heavy regularization."""
        return XGBoostClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=random_state,
            early_stopping_rounds=20
        )
    
    @staticmethod
    def create_aggressive_xgb(random_state: int = 42) -> XGBoostClassifier:
        """Create an aggressive XGBoost with high learning rate."""
        return XGBoostClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.01,
            reg_lambda=0.1,
            random_state=random_state,
            early_stopping_rounds=25
        )


def create_xgboost_model(variant: str = 'balanced', **kwargs) -> XGBoostClassifier:
    """
    Factory function to create XGBoost models.
    
    Args:
        variant: Model variant ('fast', 'balanced', 'deep', 'conservative', 'aggressive')
        **kwargs: Additional arguments for model creation
        
    Returns:
        Configured XGBoost classifier
    """
    variant = variant.lower()
    
    if variant == 'fast':
        return XGBoostVariants.create_fast_xgb(**kwargs)
    elif variant == 'balanced':
        return XGBoostVariants.create_balanced_xgb(**kwargs)
    elif variant == 'deep':
        return XGBoostVariants.create_deep_xgb(**kwargs)
    elif variant == 'conservative':
        return XGBoostVariants.create_conservative_xgb(**kwargs)
    elif variant == 'aggressive':
        return XGBoostVariants.create_aggressive_xgb(**kwargs)
    else:
        raise ValueError(f"Unknown XGBoost variant: {variant}")


def model_summary_xgb(model: XGBoostClassifier) -> str:
    """
    Generate a summary of the XGBoost model.
    
    Args:
        model: XGBoost classifier
        
    Returns:
        Model summary string
    """
    info = model.get_model_info()
    params = info['parameters']
    
    summary = f"""
XGBoost Model Summary
====================
Objective: {params['objective']}
Number of Classes: {params['num_class']}
Number of Estimators: {params['n_estimators']}
Max Depth: {params['max_depth']}
Learning Rate: {params['learning_rate']}
Subsample: {params['subsample']}
Column Sample by Tree: {params['colsample_bytree']}
L1 Regularization (alpha): {params['reg_alpha']}
L2 Regularization (lambda): {params['reg_lambda']}
Evaluation Metric: {params['eval_metric']}

Model Status: {'Fitted' if info['is_fitted'] else 'Not Fitted'}
"""
    
    if info['is_fitted']:
        summary += f"""
Training Results:
----------------
Best Iteration: {info['best_iteration']}
Best Score: {info['best_score']:.4f}
Training Time: {info['training_time']:.2f} seconds
Number of Features: {info['num_features']}
"""
    
    return summary.strip()