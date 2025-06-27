"""Convolutional Neural Network (CNN) model implementation using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import math


class CNN(nn.Module):
    """Convolutional Neural Network for MNIST digit classification."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10,
                 conv_layers: List[Dict] = None, fc_layers: List[int] = [128],
                 dropout_rate: float = 0.25, use_batch_norm: bool = True):
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of input channels (1 for grayscale MNIST)
            num_classes: Number of output classes (10 for MNIST)
            conv_layers: List of conv layer configs [{'out_channels': int, 'kernel_size': int, 'stride': int, 'padding': int}]
            fc_layers: List of fully connected layer sizes
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(CNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Default conv layer configuration if none provided
        if conv_layers is None:
            conv_layers = [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ]
        
        self.conv_layers_config = conv_layers
        self.fc_layers_config = fc_layers
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = input_channels
        current_size = 28  # MNIST image size
        
        for i, layer_config in enumerate(conv_layers):
            out_channels = layer_config['out_channels']
            kernel_size = layer_config['kernel_size']
            stride = layer_config.get('stride', 1)
            padding = layer_config.get('padding', 0)
            
            # Convolutional layer
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv)
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm2d(out_channels))
            else:
                self.batch_norms.append(nn.Identity())
            
            # Max pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pools.append(pool)
            
            # Calculate output size after conv and pooling
            current_size = ((current_size + 2 * padding - kernel_size) // stride + 1) // 2
            in_channels = out_channels
        
        # Calculate flattened size
        self.flattened_size = in_channels * current_size * current_size
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        prev_size = self.flattened_size
        
        for fc_size in fc_layers:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_dropouts.append(nn.Dropout(dropout_rate))
            prev_size = fc_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # Final dropout
        self.final_dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Ensure input has correct shape
        if x.dim() == 3:  # Add channel dimension if missing
            x = x.unsqueeze(1)
        
        # Convolutional layers
        for conv, bn, pool in zip(self.conv_layers, self.batch_norms, self.pools):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc, dropout in zip(self.fc_layers, self.fc_dropouts):
            x = fc(x)
            x = F.relu(x)
            x = dropout(x)
        
        # Output layer
        x = self.final_dropout(x)
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, x):
        """Get prediction probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CNN',
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'conv_layers': self.conv_layers_config,
            'fc_layers': self.fc_layers_config,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'flattened_size': self.flattened_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_conv_layers': len(self.conv_layers_config),
            'num_fc_layers': len(self.fc_layers_config)
        }


class CNNVariants:
    """Factory class for creating different CNN architectures."""
    
    @staticmethod
    def create_simple_cnn(dropout_rate: float = 0.25) -> CNN:
        """Create a simple CNN with 2 conv layers."""
        conv_layers = [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        return CNN(
            conv_layers=conv_layers,
            fc_layers=[128],
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )
    
    @staticmethod
    def create_medium_cnn(dropout_rate: float = 0.3) -> CNN:
        """Create a medium CNN with 3 conv layers."""
        conv_layers = [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        return CNN(
            conv_layers=conv_layers,
            fc_layers=[256, 128],
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )
    
    @staticmethod
    def create_deep_cnn(dropout_rate: float = 0.4) -> CNN:
        """Create a deep CNN with 4 conv layers."""
        conv_layers = [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        return CNN(
            conv_layers=conv_layers,
            fc_layers=[512, 256],
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )
    
    @staticmethod
    def create_wide_cnn(dropout_rate: float = 0.3) -> CNN:
        """Create a wide CNN with more channels per layer."""
        conv_layers = [
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        return CNN(
            conv_layers=conv_layers,
            fc_layers=[512],
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )
    
    @staticmethod
    def create_leNet_style(dropout_rate: float = 0.2) -> CNN:
        """Create a LeNet-style CNN."""
        conv_layers = [
            {'out_channels': 6, 'kernel_size': 5, 'stride': 1, 'padding': 0},
            {'out_channels': 16, 'kernel_size': 5, 'stride': 1, 'padding': 0}
        ]
        return CNN(
            conv_layers=conv_layers,
            fc_layers=[120, 84],
            dropout_rate=dropout_rate,
            use_batch_norm=False  # Original LeNet didn't use batch norm
        )
    
    @staticmethod
    def create_modern_cnn(dropout_rate: float = 0.3) -> CNN:
        """Create a modern CNN with smaller kernels and more layers."""
        conv_layers = [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        return CNN(
            conv_layers=conv_layers,
            fc_layers=[256],
            dropout_rate=dropout_rate,
            use_batch_norm=True
        )


def create_cnn_model(variant: str = 'medium', **kwargs) -> CNN:
    """
    Factory function to create CNN models.
    
    Args:
        variant: Model variant ('simple', 'medium', 'deep', 'wide', 'lenet', 'modern')
        **kwargs: Additional arguments for model creation
        
    Returns:
        Configured CNN model
    """
    variant = variant.lower()
    
    if variant == 'simple':
        return CNNVariants.create_simple_cnn(**kwargs)
    elif variant == 'medium':
        return CNNVariants.create_medium_cnn(**kwargs)
    elif variant == 'deep':
        return CNNVariants.create_deep_cnn(**kwargs)
    elif variant == 'wide':
        return CNNVariants.create_wide_cnn(**kwargs)
    elif variant == 'lenet':
        return CNNVariants.create_leNet_style(**kwargs)
    elif variant == 'modern':
        return CNNVariants.create_modern_cnn(**kwargs)
    else:
        raise ValueError(f"Unknown CNN variant: {variant}")


def calculate_conv_output_size(input_size: int, kernel_size: int, 
                              stride: int = 1, padding: int = 0) -> int:
    """Calculate output size after convolution."""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def calculate_cnn_output_shape(input_shape: Tuple[int, int, int], 
                              conv_layers: List[Dict]) -> Tuple[int, int, int]:
    """
    Calculate the output shape after all conv layers.
    
    Args:
        input_shape: (channels, height, width)
        conv_layers: List of conv layer configurations
        
    Returns:
        Output shape (channels, height, width)
    """
    channels, height, width = input_shape
    
    for layer_config in conv_layers:
        channels = layer_config['out_channels']
        kernel_size = layer_config['kernel_size']
        stride = layer_config.get('stride', 1)
        padding = layer_config.get('padding', 0)
        
        # After convolution
        height = calculate_conv_output_size(height, kernel_size, stride, padding)
        width = calculate_conv_output_size(width, kernel_size, stride, padding)
        
        # After pooling (assuming 2x2 max pooling)
        height = height // 2
        width = width // 2
    
    return channels, height, width


def model_summary_cnn(model: CNN, input_shape: Tuple[int, int, int] = (1, 28, 28)) -> str:
    """
    Generate a summary of the CNN model.
    
    Args:
        model: CNN model
        input_shape: Shape of input tensor (channels, height, width)
        
    Returns:
        Model summary string
    """
    model_info = model.get_model_info()
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate output shapes
    output_shape = calculate_cnn_output_shape(input_shape, model_info['conv_layers'])
    
    summary = f"""
CNN Model Summary
=================
Architecture: {model_info['num_conv_layers']} Conv + {model_info['num_fc_layers']} FC layers
Input Shape: {input_shape}
Output Shape after Conv: {output_shape}
Flattened Size: {model_info['flattened_size']}

Convolutional Layers:
"""
    
    for i, layer in enumerate(model_info['conv_layers']):
        summary += f"  Conv{i+1}: {layer['out_channels']} filters, {layer['kernel_size']}x{layer['kernel_size']} kernel\n"
    
    summary += f"\nFully Connected Layers: {model_info['fc_layers']}\n"
    summary += f"Dropout Rate: {model_info['dropout_rate']}\n"
    summary += f"Batch Normalization: {model_info['use_batch_norm']}\n"
    
    summary += f"""
Parameters:
-----------
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)
"""
    
    return summary.strip()