"""Hyperparameter configurations for different model types."""

import itertools
from typing import Dict, List, Any, Union, Tuple
import numpy as np


class HyperparameterGrid:
    """Manages hyperparameter grids for different model types."""
    
    @staticmethod
    def get_mlp_hyperparams() -> List[Dict[str, Any]]:
        """
        Get hyperparameter combinations for MLP models.
        
        Returns:
            List of hyperparameter dictionaries
        """
        param_grid = {
            'variant': ['small', 'medium', 'large'],
            'lr': [0.001, 0.003, 0.01],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'dropout_rate': [0.2, 0.3, 0.4],
            'optimizer_type': ['adam', 'adamw'],
            'scheduler_type': ['cosine', 'step', None],
            'epochs': [20, 30],
            'batch_size': [64, 128]
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        param_list = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_list.append(param_dict)
        
        return param_list
    
    @staticmethod
    def get_mlp_focused_hyperparams() -> List[Dict[str, Any]]:
        """
        Get focused hyperparameter combinations for MLP (fewer but well-chosen).
        
        Returns:
            List of hyperparameter dictionaries
        """
        configurations = [
            # Small MLP variations
            {
                'variant': 'small',
                'lr': 0.001,
                'weight_decay': 1e-4,
                'dropout_rate': 0.2,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 25,
                'batch_size': 64
            },
            {
                'variant': 'small',
                'lr': 0.003,
                'weight_decay': 1e-5,
                'dropout_rate': 0.3,
                'optimizer_type': 'adamw',
                'scheduler_type': 'step',
                'epochs': 30,
                'batch_size': 128
            },
            
            # Medium MLP variations
            {
                'variant': 'medium',
                'lr': 0.001,
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 25,
                'batch_size': 64
            },
            {
                'variant': 'medium',
                'lr': 0.003,
                'weight_decay': 1e-3,
                'dropout_rate': 0.4,
                'optimizer_type': 'adamw',
                'scheduler_type': None,
                'epochs': 30,
                'batch_size': 128
            },
            
            # Large MLP variations
            {
                'variant': 'large',
                'lr': 0.001,
                'weight_decay': 1e-3,
                'dropout_rate': 0.4,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 20,
                'batch_size': 64
            },
            {
                'variant': 'large',
                'lr': 0.0003,
                'weight_decay': 1e-4,
                'dropout_rate': 0.5,
                'optimizer_type': 'adamw',
                'scheduler_type': 'step',
                'epochs': 25,
                'batch_size': 128
            }
        ]
        
        return configurations
    
    @staticmethod
    def get_cnn_hyperparams() -> List[Dict[str, Any]]:
        """
        Get hyperparameter combinations for CNN models.
        
        Returns:
            List of hyperparameter dictionaries
        """
        configurations = [
            # Simple CNN variations
            {
                'variant': 'simple',
                'lr': 0.001,
                'weight_decay': 1e-4,
                'dropout_rate': 0.25,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 15,
                'batch_size': 64
            },
            {
                'variant': 'simple',
                'lr': 0.003,
                'weight_decay': 1e-5,
                'dropout_rate': 0.3,
                'optimizer_type': 'adamw',
                'scheduler_type': 'step',
                'epochs': 20,
                'batch_size': 128
            },
            
            # Medium CNN variations
            {
                'variant': 'medium',
                'lr': 0.001,
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 15,
                'batch_size': 64
            },
            {
                'variant': 'medium',
                'lr': 0.003,
                'weight_decay': 1e-3,
                'dropout_rate': 0.4,
                'optimizer_type': 'adamw',
                'scheduler_type': None,
                'epochs': 20,
                'batch_size': 128
            },
            
            # Deep CNN variations
            {
                'variant': 'deep',
                'lr': 0.0003,
                'weight_decay': 1e-3,
                'dropout_rate': 0.4,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 15,
                'batch_size': 32
            },
            {
                'variant': 'deep',
                'lr': 0.001,
                'weight_decay': 1e-4,
                'dropout_rate': 0.5,
                'optimizer_type': 'adamw',
                'scheduler_type': 'step',
                'epochs': 20,
                'batch_size': 64
            },
            
            # Modern CNN variations
            {
                'variant': 'modern',
                'lr': 0.001,
                'weight_decay': 1e-4,
                'dropout_rate': 0.3,
                'optimizer_type': 'adam',
                'scheduler_type': 'cosine',
                'epochs': 15,
                'batch_size': 64
            },
            {
                'variant': 'modern',
                'lr': 0.003,
                'weight_decay': 1e-5,
                'dropout_rate': 0.25,
                'optimizer_type': 'adamw',
                'scheduler_type': None,
                'epochs': 18,
                'batch_size': 128
            }
        ]
        
        return configurations
    
    @staticmethod
    def get_xgboost_hyperparams() -> List[Dict[str, Any]]:
        """
        Get hyperparameter combinations for XGBoost models.
        
        Returns:
            List of hyperparameter dictionaries
        """
        configurations = [
            # Fast XGBoost variations
            {
                'variant': 'fast',
                'n_estimators': 50,
                'max_depth': 4,
                'learning_rate': 0.3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0
            },
            {
                'variant': 'fast',
                'n_estimators': 75,
                'max_depth': 5,
                'learning_rate': 0.2,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            
            # Balanced XGBoost variations
            {
                'variant': 'balanced',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            {
                'variant': 'balanced',
                'n_estimators': 150,
                'max_depth': 7,
                'learning_rate': 0.08,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.2,
                'reg_lambda': 1.5
            },
            
            # Deep XGBoost variations
            {
                'variant': 'deep',
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.1,
                'reg_lambda': 1.5
            },
            {
                'variant': 'deep',
                'n_estimators': 300,
                'max_depth': 10,
                'learning_rate': 0.03,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'reg_alpha': 0.3,
                'reg_lambda': 2.0
            },
            
            # Conservative XGBoost variations
            {
                'variant': 'conservative',
                'n_estimators': 150,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'reg_alpha': 1.0,
                'reg_lambda': 2.0
            },
            {
                'variant': 'conservative',
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.03,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'reg_alpha': 1.5,
                'reg_lambda': 3.0
            },
            
            # Aggressive XGBoost variations
            {
                'variant': 'aggressive',
                'n_estimators': 300,
                'max_depth': 10,
                'learning_rate': 0.2,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.01,
                'reg_lambda': 0.1
            },
            {
                'variant': 'aggressive',
                'n_estimators': 500,
                'max_depth': 12,
                'learning_rate': 0.15,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.05,
                'reg_lambda': 0.5
            }
        ]
        
        return configurations


class HyperparameterSampler:
    """Samples hyperparameters for random search."""
    
    @staticmethod
    def sample_mlp_hyperparams(n_samples: int = 10, random_state: int = 42) -> List[Dict[str, Any]]:
        """
        Sample MLP hyperparameters randomly.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            List of sampled hyperparameter dictionaries
        """
        np.random.seed(random_state)
        
        samples = []
        for _ in range(n_samples):
            sample = {
                'variant': np.random.choice(['small', 'medium', 'large']),
                'lr': np.random.lognormal(np.log(0.001), 0.5),  # Log-normal around 0.001
                'weight_decay': np.power(10, np.random.uniform(-6, -2)),
                'dropout_rate': np.random.uniform(0.1, 0.5),
                'optimizer_type': np.random.choice(['adam', 'adamw']),
                'scheduler_type': np.random.choice(['cosine', 'step', None]),
                'epochs': np.random.choice([15, 20, 25, 30]),
                'batch_size': np.random.choice([32, 64, 128])
            }
            
            # Clip learning rate to reasonable range
            sample['lr'] = np.clip(sample['lr'], 0.0001, 0.01)
            
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def sample_cnn_hyperparams(n_samples: int = 10, random_state: int = 42) -> List[Dict[str, Any]]:
        """
        Sample CNN hyperparameters randomly.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            List of sampled hyperparameter dictionaries
        """
        np.random.seed(random_state)
        
        samples = []
        for _ in range(n_samples):
            sample = {
                'variant': np.random.choice(['simple', 'medium', 'deep', 'modern']),
                'lr': np.random.lognormal(np.log(0.001), 0.3),
                'weight_decay': np.power(10, np.random.uniform(-6, -2)),
                'dropout_rate': np.random.uniform(0.1, 0.5),
                'optimizer_type': np.random.choice(['adam', 'adamw']),
                'scheduler_type': np.random.choice(['cosine', 'step', None]),
                'epochs': np.random.choice([10, 15, 20, 25]),
                'batch_size': np.random.choice([32, 64, 128])
            }
            
            # Clip learning rate to reasonable range
            sample['lr'] = np.clip(sample['lr'], 0.0001, 0.01)
            
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def sample_xgboost_hyperparams(n_samples: int = 10, random_state: int = 42) -> List[Dict[str, Any]]:
        """
        Sample XGBoost hyperparameters randomly.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            List of sampled hyperparameter dictionaries
        """
        np.random.seed(random_state)
        
        samples = []
        for _ in range(n_samples):
            sample = {
                'n_estimators': int(np.random.choice([50, 100, 150, 200, 300])),
                'max_depth': int(np.random.choice([3, 4, 5, 6, 7, 8, 10])),
                'learning_rate': np.random.lognormal(np.log(0.1), 0.5),
                'subsample': np.random.uniform(0.5, 1.0),
                'colsample_bytree': np.random.uniform(0.5, 1.0),
                'reg_alpha': np.power(10, np.random.uniform(-2, 1)),
                'reg_lambda': np.power(10, np.random.uniform(-1, 1))
            }
            
            # Clip learning rate to reasonable range
            sample['learning_rate'] = np.clip(sample['learning_rate'], 0.01, 0.5)
            
            samples.append(sample)
        
        return samples


class HyperparameterManager:
    """Manages hyperparameter configurations for experiments."""
    
    def __init__(self, search_type: str = 'grid'):
        """
        Initialize hyperparameter manager.
        
        Args:
            search_type: Type of search ('grid', 'random', 'focused')
        """
        self.search_type = search_type
        self.grid = HyperparameterGrid()
        self.sampler = HyperparameterSampler()
    
    def get_hyperparameters(self, model_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Get hyperparameter configurations for a model type.
        
        Args:
            model_type: Type of model ('mlp', 'cnn', 'xgboost')
            **kwargs: Additional arguments for sampling
            
        Returns:
            List of hyperparameter configurations
        """
        model_type = model_type.lower()
        
        if self.search_type == 'grid':
            if model_type == 'mlp':
                return self.grid.get_mlp_hyperparams()
            elif model_type == 'cnn':
                return self.grid.get_cnn_hyperparams()
            elif model_type == 'xgboost':
                return self.grid.get_xgboost_hyperparams()
        
        elif self.search_type == 'focused':
            if model_type == 'mlp':
                return self.grid.get_mlp_focused_hyperparams()
            elif model_type == 'cnn':
                return self.grid.get_cnn_hyperparams()  # Use regular grid for CNN
            elif model_type == 'xgboost':
                return self.grid.get_xgboost_hyperparams()  # Use regular grid for XGBoost
        
        elif self.search_type == 'random':
            n_samples = kwargs.get('n_samples', 10)
            random_state = kwargs.get('random_state', 42)
            
            if model_type == 'mlp':
                return self.sampler.sample_mlp_hyperparams(n_samples, random_state)
            elif model_type == 'cnn':
                return self.sampler.sample_cnn_hyperparams(n_samples, random_state)
            elif model_type == 'xgboost':
                return self.sampler.sample_xgboost_hyperparams(n_samples, random_state)
        
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        raise ValueError(f"Unknown model type: {model_type}")
    
    def get_experiment_summary(self, model_type: str) -> Dict[str, Any]:
        """
        Get a summary of the hyperparameter experiment.
        
        Args:
            model_type: Type of model
            
        Returns:
            Summary dictionary
        """
        hyperparams = self.get_hyperparameters(model_type)
        
        summary = {
            'model_type': model_type,
            'search_type': self.search_type,
            'total_configurations': len(hyperparams),
            'parameter_ranges': {}
        }
        
        # Calculate parameter ranges
        if hyperparams:
            for key in hyperparams[0].keys():
                values = [hp[key] for hp in hyperparams if hp[key] is not None]
                if values:
                    if isinstance(values[0], (int, float)):
                        summary['parameter_ranges'][key] = {
                            'min': min(values),
                            'max': max(values),
                            'unique_values': len(set(values))
                        }
                    else:
                        summary['parameter_ranges'][key] = {
                            'unique_values': list(set(values))
                        }
        
        return summary


def create_hyperparameter_manager(search_type: str = 'focused') -> HyperparameterManager:
    """
    Factory function to create hyperparameter manager.
    
    Args:
        search_type: Type of search ('grid', 'random', 'focused')
        
    Returns:
        Configured hyperparameter manager
    """
    return HyperparameterManager(search_type)


def get_best_hyperparams_by_model() -> Dict[str, Dict[str, Any]]:
    """
    Get the best known hyperparameters for each model type.
    
    Returns:
        Dictionary of best hyperparameters
    """
    return {
        'mlp': {
            'variant': 'medium',
            'lr': 0.001,
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'optimizer_type': 'adam',
            'scheduler_type': 'cosine',
            'epochs': 25,
            'batch_size': 64
        },
        'cnn': {
            'variant': 'medium',
            'lr': 0.001,
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'optimizer_type': 'adam',
            'scheduler_type': 'cosine',
            'epochs': 15,
            'batch_size': 64
        },
        'xgboost': {
            'variant': 'balanced',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
    }