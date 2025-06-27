"""Tests for evaluation metrics functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from mnist_classifier.utils.metrics import (
    ModelEvaluator,
    BenchmarkEvaluator,
    create_evaluator,
    create_benchmark_evaluator
)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(num_classes=10)
        
        assert evaluator.num_classes == 10
        assert len(evaluator.class_names) == 10
        assert evaluator.class_names == [str(i) for i in range(10)]
    
    def test_evaluator_custom_class_names(self):
        """Test evaluator with custom class names."""
        class_names = ['zero', 'one', 'two']
        evaluator = ModelEvaluator(num_classes=3, class_names=class_names)
        
        assert evaluator.num_classes == 3
        assert evaluator.class_names == class_names
    
    def test_basic_metrics_calculation(self):
        """Test basic classification metrics calculation."""
        evaluator = ModelEvaluator(num_classes=3)
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])  # One misclassification
        
        results = evaluator.evaluate_model(y_true, y_pred, model_name="test_model")
        
        assert 'accuracy' in results
        assert 'precision_macro' in results
        assert 'recall_macro' in results
        assert 'f1_macro' in results
        assert 'model_name' in results
        assert results['model_name'] == "test_model"
        
        # Check that accuracy is reasonable
        expected_accuracy = 5/6  # 5 correct out of 6
        assert abs(results['accuracy'] - expected_accuracy) < 0.01
    
    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        evaluator = ModelEvaluator(num_classes=3, class_names=['A', 'B', 'C'])
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])  # Class 0 has one misclassification
        
        results = evaluator.evaluate_model(y_true, y_pred)
        
        # Check per-class metrics exist
        assert 'precision_class_A' in results
        assert 'recall_class_A' in results
        assert 'f1_class_A' in results
        assert 'precision_class_B' in results
        assert 'recall_class_B' in results
        assert 'f1_class_B' in results
        
        # Class B should have perfect precision (all predictions correct)
        # but lower recall (missed one instance)
        assert results['precision_class_B'] == 1.0
        assert results['recall_class_B'] < 1.0
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        evaluator = ModelEvaluator(num_classes=3)
        
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        
        results = evaluator.evaluate_model(y_true, y_pred)
        
        assert 'confusion_matrix' in results
        cm = results['confusion_matrix']
        assert cm.shape == (3, 3)
        
        # Check specific values
        assert cm[0, 0] == 1  # Correct predictions for class 0
        assert cm[0, 1] == 1  # Class 0 misclassified as class 1
        assert cm[1, 1] == 2  # Correct predictions for class 1
    
    def test_probability_metrics(self):
        """Test probability-based metrics."""
        evaluator = ModelEvaluator(num_classes=3)
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        # Create mock probabilities
        y_pred_proba = np.array([
            [0.9, 0.05, 0.05],  # Confident prediction for class 0
            [0.1, 0.8, 0.1],    # Confident prediction for class 1
            [0.2, 0.7, 0.1],    # Predicted class 1, true class 2
            [0.85, 0.1, 0.05],  # Confident prediction for class 0
            [0.1, 0.3, 0.6],    # Predicted class 2, true class 1
            [0.05, 0.1, 0.85]   # Confident prediction for class 2
        ])
        
        results = evaluator.evaluate_model(y_true, y_pred, y_pred_proba)
        
        assert 'roc_auc_macro' in results
        assert 'log_loss' in results
        assert 'mean_confidence' in results
        assert 'std_confidence' in results
        
        # Check confidence statistics
        assert 0 <= results['mean_confidence'] <= 1
        assert results['std_confidence'] >= 0
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        evaluator = ModelEvaluator(num_classes=3, class_names=['A', 'B', 'C'])
        
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 2, 2, 1, 0, 1])  # Some errors
        
        results = evaluator.evaluate_model(y_true, y_pred)
        
        assert 'error_rate' in results
        assert 'most_confused_pair' in results
        assert 'class_error_rates' in results
        assert 'worst_class' in results
        
        # Error rate should be complement of accuracy
        assert abs(results['error_rate'] - (1 - results['accuracy'])) < 0.01
        
        # Check most confused pair structure
        confused_pair = results['most_confused_pair']
        assert 'true_class' in confused_pair
        assert 'predicted_class' in confused_pair
        assert 'count' in confused_pair
    
    def test_classification_report_generation(self):
        """Test classification report generation."""
        evaluator = ModelEvaluator(num_classes=3)
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        report = evaluator.generate_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
    
    @patch('matplotlib.pyplot.show')
    def test_confusion_matrix_plotting(self, mock_show):
        """Test confusion matrix plotting."""
        evaluator = ModelEvaluator(num_classes=3)
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        # Test plotting (won't actually display due to mock)
        evaluator.plot_confusion_matrix(y_true, y_pred)
        
        # Verify show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_per_class_metrics_plotting(self, mock_show):
        """Test per-class metrics plotting."""
        evaluator = ModelEvaluator(num_classes=3)
        
        # Create sample results
        results = {
            'precision_class_0': 0.9,
            'recall_class_0': 0.85,
            'f1_class_0': 0.875,
            'precision_class_1': 0.8,
            'recall_class_1': 0.9,
            'f1_class_1': 0.847,
            'precision_class_2': 0.95,
            'recall_class_2': 0.88,
            'f1_class_2': 0.914
        }
        
        evaluator.plot_per_class_metrics(results)
        
        mock_show.assert_called_once()


class TestBenchmarkEvaluator:
    """Test cases for BenchmarkEvaluator class."""
    
    def test_benchmark_evaluator_initialization(self):
        """Test benchmark evaluator initialization."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        
        assert evaluator.num_classes == 10
        assert isinstance(evaluator.evaluator, ModelEvaluator)
        assert len(evaluator.results) == 0
    
    def test_add_model_results(self):
        """Test adding model results."""
        evaluator = BenchmarkEvaluator(num_classes=3)
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        
        evaluator.add_model_results(
            model_name="test_model",
            y_true=y_true,
            y_pred=y_pred,
            training_time=120.5,
            inference_time=0.01
        )
        
        assert "test_model" in evaluator.results
        results = evaluator.results["test_model"]
        
        assert 'accuracy' in results
        assert 'training_time' in results
        assert 'inference_time' in results
        assert 'inference_speed' in results
        assert results['training_time'] == 120.5
        assert results['inference_time'] == 0.01
    
    def test_comparison_table_generation(self, sample_model_results):
        """Test comparison table generation."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        table = evaluator.get_comparison_table()
        
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2  # Two models
        assert 'Model' in table.columns
        assert 'accuracy' in table.columns
        assert 'precision_macro' in table.columns
        
        # Check model names are present
        model_names = table['Model'].tolist()
        assert 'model_1' in model_names
        assert 'model_2' in model_names
    
    def test_custom_metrics_comparison_table(self, sample_model_results):
        """Test comparison table with custom metrics."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        custom_metrics = ['accuracy', 'f1_macro', 'training_time']
        table = evaluator.get_comparison_table(custom_metrics)
        
        assert isinstance(table, pd.DataFrame)
        assert 'accuracy' in table.columns
        assert 'f1_macro' in table.columns
        assert 'training_time' in table.columns
        assert 'precision_macro' not in table.columns  # Not requested
    
    def test_get_best_models(self, sample_model_results):
        """Test getting best models by metric."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        # Get best models by accuracy
        best_models = evaluator.get_best_models('accuracy', top_k=2)
        
        assert len(best_models) == 2
        assert isinstance(best_models[0], tuple)
        assert len(best_models[0]) == 2  # (model_name, score)
        
        # Should be sorted in descending order
        assert best_models[0][1] >= best_models[1][1]
        
        # Top model should be model_2 (0.97 accuracy)
        assert best_models[0][0] == 'model_2'
        assert best_models[0][1] == 0.97
    
    def test_get_best_models_single(self, sample_model_results):
        """Test getting single best model."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        best_models = evaluator.get_best_models('f1_macro', top_k=1)
        
        assert len(best_models) == 1
        assert best_models[0][0] == 'model_2'  # Better F1 score
    
    def test_get_best_models_invalid_metric(self, sample_model_results):
        """Test getting best models with invalid metric."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        best_models = evaluator.get_best_models('invalid_metric')
        
        assert len(best_models) == 0  # No results for invalid metric
    
    @patch('matplotlib.pyplot.show')
    def test_model_comparison_plotting(self, mock_show, sample_model_results):
        """Test model comparison plotting."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        evaluator.plot_model_comparison()
        
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_model_comparison_custom_metrics(self, mock_show, sample_model_results):
        """Test model comparison plotting with custom metrics."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        custom_metrics = ['accuracy', 'f1_macro']
        evaluator.plot_model_comparison(metrics=custom_metrics)
        
        mock_show.assert_called_once()
    
    def test_summary_report_generation(self, sample_model_results):
        """Test summary report generation."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        evaluator.results = sample_model_results
        
        report = evaluator.generate_summary_report()
        
        assert isinstance(report, str)
        assert "Model Evaluation Summary" in report
        assert "model_1" in report
        assert "model_2" in report
        assert "Best models by accuracy" in report
    
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        
        # Test empty comparison table
        table = evaluator.get_comparison_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 0
        
        # Test empty best models
        best_models = evaluator.get_best_models('accuracy')
        assert len(best_models) == 0
        
        # Test empty summary
        report = evaluator.generate_summary_report()
        assert "No model results available" in report
    
    def test_plot_empty_results(self):
        """Test plotting with empty results."""
        evaluator = BenchmarkEvaluator(num_classes=10)
        
        with patch('builtins.print') as mock_print:
            evaluator.plot_model_comparison()
            mock_print.assert_called_with("No results to plot")


class TestFactoryFunctions:
    """Test factory functions for evaluators."""
    
    def test_create_evaluator(self):
        """Test create_evaluator factory function."""
        evaluator = create_evaluator(num_classes=5)
        
        assert isinstance(evaluator, ModelEvaluator)
        assert evaluator.num_classes == 5
    
    def test_create_benchmark_evaluator(self):
        """Test create_benchmark_evaluator factory function."""
        evaluator = create_benchmark_evaluator(num_classes=8)
        
        assert isinstance(evaluator, BenchmarkEvaluator)
        assert evaluator.num_classes == 8


class TestMetricsIntegration:
    """Integration tests for metrics components."""
    
    def test_end_to_end_evaluation(self):
        """Test end-to-end evaluation workflow."""
        # Create evaluator
        evaluator = BenchmarkEvaluator(num_classes=3)
        
        # Add multiple model results
        np.random.seed(42)  # For reproducibility
        
        # Model 1: Decent performance
        y_true = np.random.randint(0, 3, 100)
        y_pred_1 = y_true.copy()
        # Add some errors
        error_indices = np.random.choice(100, 15, replace=False)
        y_pred_1[error_indices] = np.random.randint(0, 3, 15)
        
        evaluator.add_model_results(
            model_name="model_decent",
            y_true=y_true,
            y_pred=y_pred_1,
            training_time=100.0,
            inference_time=0.005
        )
        
        # Model 2: Better performance
        y_pred_2 = y_true.copy()
        # Add fewer errors
        error_indices = np.random.choice(100, 8, replace=False)
        y_pred_2[error_indices] = np.random.randint(0, 3, 8)
        
        evaluator.add_model_results(
            model_name="model_better",
            y_true=y_true,
            y_pred=y_pred_2,
            training_time=150.0,
            inference_time=0.008
        )
        
        # Test comparison table
        table = evaluator.get_comparison_table()
        assert len(table) == 2
        assert 'model_decent' in table['Model'].values
        assert 'model_better' in table['Model'].values
        
        # Test best models
        best_by_accuracy = evaluator.get_best_models('accuracy', top_k=1)
        assert len(best_by_accuracy) == 1
        # Better model should have higher accuracy
        best_model_name = best_by_accuracy[0][0]
        assert best_model_name == "model_better"
        
        # Test summary report
        report = evaluator.generate_summary_report()
        assert "model_decent" in report
        assert "model_better" in report
    
    def test_probabilistic_evaluation(self):
        """Test evaluation with probability predictions."""
        evaluator = ModelEvaluator(num_classes=3)
        
        # Create consistent true labels and predictions
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2])  # Some errors
        
        # Create probability matrix
        y_pred_proba = np.array([
            [0.9, 0.05, 0.05],   # Correct prediction for class 0
            [0.1, 0.8, 0.1],     # Correct prediction for class 1
            [0.2, 0.7, 0.1],     # Wrong: predicted 1, true 2
            [0.85, 0.1, 0.05],   # Correct prediction for class 0
            [0.1, 0.2, 0.7],     # Wrong: predicted 2, true 1
            [0.05, 0.1, 0.85],   # Correct prediction for class 2
            [0.3, 0.6, 0.1],     # Wrong: predicted 1, true 0
            [0.15, 0.75, 0.1],   # Correct prediction for class 1
            [0.05, 0.1, 0.85]    # Correct prediction for class 2
        ])
        
        results = evaluator.evaluate_model(
            y_true, y_pred, y_pred_proba, model_name="prob_model"
        )
        
        # Check that probability metrics are calculated
        assert 'roc_auc_macro' in results
        assert 'log_loss' in results
        assert 'mean_confidence' in results
        assert 'std_confidence' in results
        
        # Confidence should be reasonable
        assert 0.5 <= results['mean_confidence'] <= 1.0
        assert 0.0 <= results['std_confidence'] <= 0.5
    
    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        evaluator = ModelEvaluator(num_classes=2)
        
        # Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        results = evaluator.evaluate_model(y_true, y_pred)
        
        assert results['accuracy'] == 1.0
        assert results['precision_macro'] == 1.0
        assert results['recall_macro'] == 1.0
        assert results['f1_macro'] == 1.0
        assert results['error_rate'] == 0.0
        
        # All wrong predictions
        y_pred_wrong = np.array([1, 0, 1, 0])
        
        results_wrong = evaluator.evaluate_model(y_true, y_pred_wrong)
        
        assert results_wrong['accuracy'] == 0.0
        assert results_wrong['error_rate'] == 1.0