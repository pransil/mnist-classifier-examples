"""Model predictor for MNIST Classifier API."""

import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from ..models.mlp import MLPClassifier, get_mlp_variant
from ..models.cnn import CNNClassifier, get_cnn_variant
from ..models.xgboost_model import XGBoostClassifier


logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and predictions for the API."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model predictor.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.best_model_name = None
        self.default_model_name = "mlp"
        
        # Try to load models on initialization
        self._load_available_models()
    
    def _load_available_models(self) -> None:
        """Load all available models from the models directory."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return
        
        # Load PyTorch models
        if TORCH_AVAILABLE:
            self._load_pytorch_models()
        
        # Load XGBoost models
        if XGBOOST_AVAILABLE:
            self._load_xgboost_models()
        
        # Determine best model
        self._determine_best_model()
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def _load_pytorch_models(self) -> None:
        """Load saved PyTorch models."""
        for model_file in self.models_dir.glob("*.pth"):
            try:
                model_name = model_file.stem
                model_type = model_name.split('_')[0]  # Extract model type from filename
                
                if model_type == 'mlp':
                    model = self._load_mlp_model(model_file, model_name)
                elif model_type == 'cnn':
                    model = self._load_cnn_model(model_file, model_name)
                else:
                    continue
                
                if model is not None:
                    self.models[model_name] = model
                    self.model_info[model_name] = {
                        'type': model_type,
                        'file_path': str(model_file),
                        'is_available': True
                    }
                    
            except Exception as e:
                logger.error(f"Failed to load PyTorch model {model_file}: {e}")
    
    def _load_mlp_model(self, model_file: Path, model_name: str) -> Optional[nn.Module]:
        """Load an MLP model."""
        try:
            # Try to determine variant from filename
            parts = model_name.split('_')
            variant = parts[1] if len(parts) > 1 else 'medium'
            
            # Create model architecture
            model_config = get_mlp_variant(variant)
            model = MLPClassifier(**model_config)
            
            # Load state dict
            checkpoint = torch.load(model_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Extract additional info if available
                if 'accuracy' in checkpoint:
                    self.model_info[model_name] = self.model_info.get(model_name, {})
                    self.model_info[model_name]['accuracy'] = checkpoint['accuracy']
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load MLP model {model_name}: {e}")
            return None
    
    def _load_cnn_model(self, model_file: Path, model_name: str) -> Optional[nn.Module]:
        """Load a CNN model."""
        try:
            # Try to determine variant from filename
            parts = model_name.split('_')
            variant = parts[1] if len(parts) > 1 else 'medium'
            
            # Create model architecture
            model_config = get_cnn_variant(variant)
            model = CNNClassifier(**model_config)
            
            # Load state dict
            checkpoint = torch.load(model_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Extract additional info if available
                if 'accuracy' in checkpoint:
                    self.model_info[model_name] = self.model_info.get(model_name, {})
                    self.model_info[model_name]['accuracy'] = checkpoint['accuracy']
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load CNN model {model_name}: {e}")
            return None
    
    def _load_xgboost_models(self) -> None:
        """Load saved XGBoost models."""
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                if not model_name.startswith('xgboost'):
                    continue
                
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    model = model_data.get('model')
                    accuracy = model_data.get('accuracy')
                else:
                    model = model_data
                    accuracy = None
                
                if model is not None:
                    self.models[model_name] = model
                    self.model_info[model_name] = {
                        'type': 'xgboost',
                        'file_path': str(model_file),
                        'is_available': True,
                        'accuracy': accuracy
                    }
                    
            except Exception as e:
                logger.error(f"Failed to load XGBoost model {model_file}: {e}")
    
    def _determine_best_model(self) -> None:
        """Determine the best model based on accuracy."""
        best_accuracy = 0.0
        best_model = None
        
        for model_name, info in self.model_info.items():
            accuracy = info.get('accuracy', 0.0)
            if accuracy and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        if best_model:
            self.best_model_name = best_model
        else:
            # Fall back to first available model
            if self.models:
                self.best_model_name = list(self.models.keys())[0]
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get information about all available models."""
        models_info = []
        
        for model_name, model in self.models.items():
            info = self.model_info.get(model_name, {})
            
            # Get model parameters count
            parameters = None
            if hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
                try:
                    parameters = sum(p.numel() for p in model.parameters())
                except:
                    pass
            
            models_info.append({
                'name': model_name,
                'type': info.get('type', 'unknown'),
                'accuracy': info.get('accuracy'),
                'parameters': parameters,
                'training_time': info.get('training_time'),
                'is_available': True
            })
        
        return models_info
    
    def predict(self, image_data: np.ndarray, model_name: Optional[str] = None,
                return_probabilities: bool = False) -> Tuple[int, float, Optional[Dict[str, float]], str, float]:
        """
        Make a prediction on image data.
        
        Args:
            image_data: 28x28 numpy array with pixel values
            model_name: Name of model to use, or None for best model
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Tuple of (predicted_digit, confidence, probabilities, model_used, processing_time_ms)
        """
        start_time = time.time()
        
        # Select model
        if model_name == "best" or model_name is None:
            model_name = self.best_model_name or self.default_model_name
        
        if model_name not in self.models:
            available_models = list(self.models.keys())
            if available_models:
                model_name = available_models[0]
                logger.warning(f"Requested model not found, using {model_name}")
            else:
                raise ValueError("No models available for prediction")
        
        model = self.models[model_name]
        model_type = self.model_info[model_name]['type']
        
        # Prepare input data
        if model_type in ['mlp', 'cnn']:
            input_data = self._prepare_pytorch_input(image_data, model_type)
            predicted_digit, confidence, probabilities = self._predict_pytorch(
                model, input_data, return_probabilities
            )
        elif model_type == 'xgboost':
            input_data = self._prepare_xgboost_input(image_data)
            predicted_digit, confidence, probabilities = self._predict_xgboost(
                model, input_data, return_probabilities
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return predicted_digit, confidence, probabilities, model_name, processing_time_ms
    
    def _prepare_pytorch_input(self, image_data: np.ndarray, model_type: str) -> torch.Tensor:
        """Prepare input data for PyTorch models."""
        # Ensure image is 28x28
        if image_data.shape != (28, 28):
            raise ValueError(f"Image must be 28x28, got {image_data.shape}")
        
        # Convert to tensor and add batch dimension
        if model_type == 'mlp':
            # Flatten for MLP
            input_tensor = torch.FloatTensor(image_data.flatten()).unsqueeze(0)
        elif model_type == 'cnn':
            # Add channel dimension for CNN
            input_tensor = torch.FloatTensor(image_data).unsqueeze(0).unsqueeze(0)
        
        return input_tensor
    
    def _prepare_xgboost_input(self, image_data: np.ndarray) -> np.ndarray:
        """Prepare input data for XGBoost models."""
        if image_data.shape != (28, 28):
            raise ValueError(f"Image must be 28x28, got {image_data.shape}")
        
        # Flatten and add batch dimension
        return image_data.flatten().reshape(1, -1)
    
    def _predict_pytorch(self, model: nn.Module, input_data: torch.Tensor,
                        return_probabilities: bool) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """Make prediction with PyTorch model."""
        with torch.no_grad():
            outputs = model(input_data)
            probabilities_tensor = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities_tensor, dim=1).item()
            confidence = probabilities_tensor[0, predicted_class].item()
        
        probabilities = None
        if return_probabilities:
            probabilities = {
                str(i): float(probabilities_tensor[0, i].item())
                for i in range(10)
            }
        
        return predicted_class, confidence, probabilities
    
    def _predict_xgboost(self, model, input_data: np.ndarray,
                        return_probabilities: bool) -> Tuple[int, float, Optional[Dict[str, float]]]:
        """Make prediction with XGBoost model."""
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(input_data)
        
        # Get probabilities
        probabilities_array = model.predict(dmatrix)
        
        # Handle different XGBoost output formats
        if probabilities_array.ndim == 1:
            # Single prediction
            probabilities_array = probabilities_array.reshape(1, -1)
        
        predicted_class = np.argmax(probabilities_array[0])
        confidence = probabilities_array[0, predicted_class]
        
        probabilities = None
        if return_probabilities:
            probabilities = {
                str(i): float(probabilities_array[0, i])
                for i in range(10)
            }
        
        return int(predicted_class), float(confidence), probabilities
    
    def batch_predict(self, images: List[np.ndarray], model_name: Optional[str] = None,
                     return_probabilities: bool = False) -> List[Tuple[int, float, Optional[Dict[str, float]], str, float]]:
        """
        Make batch predictions on multiple images.
        
        Args:
            images: List of 28x28 numpy arrays
            model_name: Name of model to use
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction tuples
        """
        results = []
        
        for image in images:
            result = self.predict(image, model_name, return_probabilities)
            results.append(result)
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        if model_name not in self.model_info:
            return None
        
        info = self.model_info[model_name].copy()
        
        # Add runtime information
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
                try:
                    info['parameters'] = sum(p.numel() for p in model.parameters())
                except:
                    pass
        
        return info
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available for predictions."""
        return model_name in self.models
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.best_model_name or self.default_model_name
    
    def get_best_model(self) -> str:
        """Get the best performing model name."""
        return self.best_model_name or self.default_model_name