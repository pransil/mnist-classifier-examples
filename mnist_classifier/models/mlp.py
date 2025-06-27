"""Multi-Layer Perceptron (MLP) model implementation using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import numpy as np


class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST digit classification."""
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [512, 256], 
                 num_classes: int = 10, dropout_rate: float = 0.2, 
                 activation: str = 'relu'):
        """
        Initialize MLP model.
        
        Args:
            input_size: Size of input features (28*28=784 for MNIST)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes (10 for MNIST)
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        # Select activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.activation_name.lower() == 'relu':
                    # He initialization for ReLU
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # Xavier initialization for other activations
                    nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width) or (batch_size, features)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten input if needed (for image inputs)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through hidden layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = self.activation(x)
            x = dropout(x)
        
        # Output layer (no activation, raw logits for CrossEntropyLoss)
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, x):
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """
        Get class predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'MLP',
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_layers': len(self.hidden_sizes) + 1  # +1 for output layer
        }


class MLPVariants:
    """Factory class for creating different MLP architectures."""
    
    @staticmethod
    def create_small_mlp(dropout_rate: float = 0.2) -> MLP:
        """Create a small MLP with 2 hidden layers."""
        return MLP(
            hidden_sizes=[256, 128],
            dropout_rate=dropout_rate,
            activation='relu'
        )
    
    @staticmethod
    def create_medium_mlp(dropout_rate: float = 0.3) -> MLP:
        """Create a medium MLP with 3 hidden layers."""
        return MLP(
            hidden_sizes=[512, 256, 128],
            dropout_rate=dropout_rate,
            activation='relu'
        )
    
    @staticmethod
    def create_large_mlp(dropout_rate: float = 0.4) -> MLP:
        """Create a large MLP with 4 hidden layers."""
        return MLP(
            hidden_sizes=[1024, 512, 256, 128],
            dropout_rate=dropout_rate,
            activation='relu'
        )
    
    @staticmethod
    def create_deep_mlp(dropout_rate: float = 0.3) -> MLP:
        """Create a deep MLP with 5 hidden layers."""
        return MLP(
            hidden_sizes=[512, 512, 256, 256, 128],
            dropout_rate=dropout_rate,
            activation='relu'
        )
    
    @staticmethod
    def create_wide_mlp(dropout_rate: float = 0.3) -> MLP:
        """Create a wide MLP with fewer but larger layers."""
        return MLP(
            hidden_sizes=[1024, 512],
            dropout_rate=dropout_rate,
            activation='relu'
        )
    
    @staticmethod
    def create_gelu_mlp(dropout_rate: float = 0.2) -> MLP:
        """Create an MLP with GELU activation."""
        return MLP(
            hidden_sizes=[512, 256, 128],
            dropout_rate=dropout_rate,
            activation='gelu'
        )
    
    @staticmethod
    def create_custom_mlp(hidden_sizes: List[int], dropout_rate: float = 0.2, 
                         activation: str = 'relu') -> MLP:
        """Create a custom MLP with specified architecture."""
        return MLP(
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            activation=activation
        )


def create_mlp_model(variant: str = 'medium', **kwargs) -> MLP:
    """
    Factory function to create MLP models.
    
    Args:
        variant: Model variant ('small', 'medium', 'large', 'deep', 'wide', 'gelu', 'custom')
        **kwargs: Additional arguments for model creation
        
    Returns:
        Configured MLP model
    """
    variant = variant.lower()
    
    if variant == 'small':
        return MLPVariants.create_small_mlp(**kwargs)
    elif variant == 'medium':
        return MLPVariants.create_medium_mlp(**kwargs)
    elif variant == 'large':
        return MLPVariants.create_large_mlp(**kwargs)
    elif variant == 'deep':
        return MLPVariants.create_deep_mlp(**kwargs)
    elif variant == 'wide':
        return MLPVariants.create_wide_mlp(**kwargs)
    elif variant == 'gelu':
        return MLPVariants.create_gelu_mlp(**kwargs)
    elif variant == 'custom':
        return MLPVariants.create_custom_mlp(**kwargs)
    else:
        raise ValueError(f"Unknown MLP variant: {variant}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def model_summary(model: MLP, input_shape: tuple = (1, 28, 28)) -> str:
    """
    Generate a summary of the MLP model.
    
    Args:
        model: MLP model
        input_shape: Shape of input tensor (excluding batch dimension)
        
    Returns:
        Model summary string
    """
    model_info = model.get_model_info()
    param_count = count_parameters(model)
    
    summary = f"""
MLP Model Summary
================
Architecture: {model_info['hidden_sizes']} → {model_info['num_classes']}
Input Size: {model_info['input_size']}
Activation: {model_info['activation']}
Dropout Rate: {model_info['dropout_rate']}
Number of Layers: {model_info['num_layers']}

Parameters:
-----------
Total Parameters: {param_count['total']:,}
Trainable Parameters: {param_count['trainable']:,}
Non-trainable Parameters: {param_count['non_trainable']:,}

Model Size: {param_count['total'] * 4 / 1024 / 1024:.2f} MB (float32)
"""
    
    return summary.strip()