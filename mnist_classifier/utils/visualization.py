"""Visualization utilities for MNIST classifier."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_mnist_samples(images: np.ndarray, labels: np.ndarray, 
                      predictions: Optional[np.ndarray] = None,
                      num_samples: int = 10, figsize: Tuple[int, int] = (15, 6),
                      save_path: Optional[str] = None) -> None:
    """
    Plot MNIST image samples with labels and predictions.
    
    Args:
        images: Array of images (N, H, W) or (N, 1, H, W)
        labels: True labels
        predictions: Predicted labels (optional)
        num_samples: Number of samples to plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Ensure images are in correct format
    if images.ndim == 4:
        images = images.squeeze(1)
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i], cmap='gray')
        
        title = f'True: {labels[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'
            if labels[i] != predictions[i]:
                axes[i].set_title(title, color='red')
            else:
                axes[i].set_title(title, color='green')
        else:
            axes[i].set_title(title)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = True, figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        figsize: Figure size
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_comparison(histories: Dict[str, Dict[str, List[float]]],
                           metric: str = 'accuracy', figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None) -> None:
    """
    Compare training histories of multiple models.
    
    Args:
        histories: Dictionary of {model_name: history_dict}
        metric: Metric to plot ('accuracy', 'loss')
        figsize: Figure size
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for model_name, history in histories.items():
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history and val_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            
            ax1.plot(epochs, history[train_key], label=f'{model_name} (Train)', linestyle='-')
            ax2.plot(epochs, history[val_key], label=f'{model_name} (Val)', linestyle='--')
    
    ax1.set_title(f'Training {metric.capitalize()}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metric.capitalize())
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title(f'Validation {metric.capitalize()}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric.capitalize())
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                         metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Plot model comparison across multiple metrics.
    
    Args:
        results: Dictionary of {model_name: {metric: value}}
        metrics: List of metrics to compare
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Prepare data for plotting
    models = list(results.keys())
    metric_values = {metric: [results[model].get(metric, 0) for model in models] 
                    for metric in metrics}
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            bars = axes[i].bar(models, metric_values[metric])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
    
    # Remove empty subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance_dict: Dict[str, float], 
                          top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        importance_dict: Dictionary of {feature: importance}
        top_n: Number of top features to plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importance = zip(*sorted_features)
    
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), [f'Feature {f}' for f in features])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, importance)):
        plt.text(value + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(train_sizes: np.ndarray, train_scores: np.ndarray, 
                        val_scores: np.ndarray, metric_name: str = 'Accuracy',
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curves showing performance vs training set size.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        metric_name: Name of the metric being plotted
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel(metric_name)
    plt.title(f'Learning Curves - {metric_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_heatmap(results_df: pd.DataFrame, 
                               param1: str, param2: str, metric: str = 'accuracy',
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
    """
    Plot heatmap of hyperparameter search results.
    
    Args:
        results_df: DataFrame with hyperparameter results
        param1: First parameter name
        param2: Second parameter name
        metric: Metric to plot
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Pivot the data for heatmap
    heatmap_data = results_df.pivot_table(values=metric, index=param2, columns=param1)
    
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'{metric.capitalize()} by {param1} and {param2}')
    plt.xlabel(param1.replace('_', ' ').title())
    plt.ylabel(param2.replace('_', ' ').title())
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_interactive_results_dashboard(results: Dict[str, Any],
                                       save_path: Optional[str] = None) -> None:
    """
    Create an interactive dashboard with Plotly.
    
    Args:
        results: Dictionary containing model results and metrics
        save_path: Path to save the HTML dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Model Accuracy Comparison', 'Training Time Comparison',
                       'Model Size Comparison', 'Confusion Matrix'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )
    
    models = list(results.keys())
    
    # Accuracy comparison
    accuracies = [results[model].get('test_accuracy', 0) for model in models]
    fig.add_trace(
        go.Bar(x=models, y=accuracies, name='Test Accuracy', 
               text=[f'{acc:.3f}' for acc in accuracies], textposition='auto'),
        row=1, col=1
    )
    
    # Training time comparison
    train_times = [results[model].get('training_time', 0) for model in models]
    fig.add_trace(
        go.Bar(x=models, y=train_times, name='Training Time (s)',
               text=[f'{time:.1f}s' for time in train_times], textposition='auto'),
        row=1, col=2
    )
    
    # Model size comparison (if available)
    model_sizes = [results[model].get('model_size_mb', 0) for model in models]
    fig.add_trace(
        go.Bar(x=models, y=model_sizes, name='Model Size (MB)',
               text=[f'{size:.1f}MB' for size in model_sizes], textposition='auto'),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="MNIST Classifier Model Comparison Dashboard",
        showlegend=False,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()


def save_all_plots(output_dir: str = "plots") -> None:
    """Create output directory for plots."""
    plots_dir = Path(output_dir)
    plots_dir.mkdir(exist_ok=True)
    return plots_dir