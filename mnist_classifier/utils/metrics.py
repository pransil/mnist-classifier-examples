"""Evaluation metrics for MNIST classifier."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import warnings


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, num_classes: int = 10, class_names: Optional[List[str]] = None):
        """
        Initialize model evaluator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes (defaults to digits 0-9)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: Optional[np.ndarray] = None,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {
            'model_name': model_name,
            'num_samples': len(y_true),
            'num_classes': self.num_classes
        }
        
        # Basic classification metrics
        results.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Per-class metrics
        results.update(self._calculate_per_class_metrics(y_true, y_pred))
        
        # Confusion matrix
        results['confusion_matrix'] = self._calculate_confusion_matrix(y_true, y_pred)
        
        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            results.update(self._calculate_probability_metrics(y_true, y_pred_proba))
        
        # Error analysis
        results.update(self._analyze_errors(y_true, y_pred))
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate per-class metrics."""
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            per_class[f'precision_class_{class_name}'] = precision[i] if i < len(precision) else 0.0
            per_class[f'recall_class_{class_name}'] = recall[i] if i < len(recall) else 0.0
            per_class[f'f1_class_{class_name}'] = f1[i] if i < len(f1) else 0.0
        
        return per_class
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate probability-based metrics."""
        metrics = {}
        
        try:
            # Binarize labels for multi-class ROC AUC
            y_true_binary = label_binarize(y_true, classes=range(self.num_classes))
            
            # ROC AUC (macro and weighted)
            if y_true_binary.shape[1] > 1:  # Multi-class
                metrics['roc_auc_macro'] = roc_auc_score(y_true_binary, y_pred_proba, average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_binary, y_pred_proba, average='weighted')
            else:  # Binary case
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            # Average precision score
            if y_true_binary.shape[1] > 1:
                metrics['avg_precision_macro'] = average_precision_score(y_true_binary, y_pred_proba, average='macro')
                metrics['avg_precision_weighted'] = average_precision_score(y_true_binary, y_pred_proba, average='weighted')
            else:
                metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
            
            # Log loss (cross-entropy)
            epsilon = 1e-15
            y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
            log_loss = -np.mean(np.log(y_pred_proba_clipped[np.arange(len(y_true)), y_true]))
            metrics['log_loss'] = log_loss
            
            # Prediction confidence statistics
            max_probs = np.max(y_pred_proba, axis=1)
            metrics['mean_confidence'] = np.mean(max_probs)
            metrics['std_confidence'] = np.std(max_probs)
            metrics['min_confidence'] = np.min(max_probs)
            metrics['max_confidence'] = np.max(max_probs)
            
        except Exception as e:
            warnings.warn(f"Error calculating probability metrics: {e}")
        
        return metrics
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = {}
        
        # Error rate
        errors['error_rate'] = 1 - accuracy_score(y_true, y_pred)
        
        # Most confused classes
        cm = confusion_matrix(y_true, y_pred)
        
        # Off-diagonal elements (errors)
        error_matrix = cm.copy()
        np.fill_diagonal(error_matrix, 0)
        
        # Find most confused pairs
        max_error_idx = np.unravel_index(np.argmax(error_matrix), error_matrix.shape)
        errors['most_confused_pair'] = {
            'true_class': self.class_names[max_error_idx[0]],
            'predicted_class': self.class_names[max_error_idx[1]],
            'count': int(error_matrix[max_error_idx])
        }
        
        # Class-wise error rates
        class_totals = np.sum(cm, axis=1)
        class_errors = np.sum(error_matrix, axis=1)
        class_error_rates = np.divide(class_errors, class_totals, 
                                    out=np.zeros_like(class_errors, dtype=float), 
                                    where=class_totals!=0)
        
        errors['class_error_rates'] = {
            self.class_names[i]: float(rate) for i, rate in enumerate(class_error_rates)
        }
        
        # Worst performing class
        worst_class_idx = np.argmax(class_error_rates)
        errors['worst_class'] = {
            'class': self.class_names[worst_class_idx],
            'error_rate': float(class_error_rates[worst_class_idx])
        }
        
        return errors
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate sklearn classification report."""
        return classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True, figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
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
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_class_metrics(self, results: Dict[str, Any], 
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> None:
        """Plot per-class metrics."""
        classes = self.class_names
        
        # Extract per-class metrics
        precision = [results.get(f'precision_class_{cls}', 0) for cls in classes]
        recall = [results.get(f'recall_class_{cls}', 0) for cls in classes]
        f1 = [results.get(f'f1_class_{cls}', 0) for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class BenchmarkEvaluator:
    """Evaluates and compares multiple models."""
    
    def __init__(self, num_classes: int = 10):
        """Initialize benchmark evaluator."""
        self.num_classes = num_classes
        self.evaluator = ModelEvaluator(num_classes)
        self.results = {}
    
    def add_model_results(self, model_name: str, y_true: np.ndarray, 
                         y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                         training_time: Optional[float] = None,
                         inference_time: Optional[float] = None) -> None:
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            training_time: Training time in seconds
            inference_time: Inference time in seconds
        """
        results = self.evaluator.evaluate_model(y_true, y_pred, y_pred_proba, model_name)
        
        # Add timing information
        if training_time is not None:
            results['training_time'] = training_time
        if inference_time is not None:
            results['inference_time'] = inference_time
            results['inference_speed'] = len(y_true) / inference_time if inference_time > 0 else 0
        
        self.results[model_name] = results
    
    def get_comparison_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comparison table of all models.
        
        Args:
            metrics: List of metrics to include
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                      'training_time', 'inference_time']
        
        data = []
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            for metric in metrics:
                row[metric] = results.get(metric, None)
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.round(4)
        return df
    
    def get_best_models(self, metric: str = 'accuracy', top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top performing models by a metric.
        
        Args:
            metric: Metric to rank by
            top_k: Number of top models to return
            
        Returns:
            List of (model_name, metric_value) tuples
        """
        if not self.results:
            return []
        
        model_scores = []
        for model_name, results in self.results.items():
            if metric in results:
                model_scores.append((model_name, results[metric]))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return model_scores[:top_k]
    
    def plot_model_comparison(self, metrics: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> None:
        """Plot model comparison across metrics."""
        if not self.results:
            print("No results to plot")
            return
        
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        models = list(self.results.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
            if i < len(axes):
                values = [self.results[model].get(metric, 0) for model in models]
                
                bars = axes[i].bar(models, values, alpha=0.7)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
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
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of all model results."""
        if not self.results:
            return "No model results available."
        
        report = "MNIST Classifier Model Evaluation Summary\n"
        report += "=" * 50 + "\n\n"
        
        # Best models by key metrics
        key_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        for metric in key_metrics:
            best_models = self.get_best_models(metric, top_k=3)
            if best_models:
                report += f"Best models by {metric}:\n"
                for i, (model, score) in enumerate(best_models, 1):
                    report += f"  {i}. {model}: {score:.4f}\n"
                report += "\n"
        
        # Detailed results for each model
        for model_name, results in self.results.items():
            report += f"Model: {model_name}\n"
            report += "-" * 30 + "\n"
            report += f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n"
            report += f"Precision (macro): {results.get('precision_macro', 'N/A'):.4f}\n"
            report += f"Recall (macro): {results.get('recall_macro', 'N/A'):.4f}\n"
            report += f"F1-score (macro): {results.get('f1_macro', 'N/A'):.4f}\n"
            
            if 'training_time' in results:
                report += f"Training time: {results['training_time']:.2f}s\n"
            if 'inference_time' in results:
                report += f"Inference time: {results['inference_time']:.4f}s\n"
            
            report += "\n"
        
        return report


def create_evaluator(num_classes: int = 10) -> ModelEvaluator:
    """Factory function to create model evaluator."""
    return ModelEvaluator(num_classes)


def create_benchmark_evaluator(num_classes: int = 10) -> BenchmarkEvaluator:
    """Factory function to create benchmark evaluator."""
    return BenchmarkEvaluator(num_classes)