"""Tests for reporting and visualization functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, mock_open
import tempfile
from pathlib import Path
import json

from mnist_classifier.utils.reporting import (
    ExperimentReporter,
    create_experiment_reporter,
    generate_quick_report
)
from mnist_classifier.utils.metrics import BenchmarkEvaluator


class TestExperimentReporter:
    """Test cases for ExperimentReporter class."""
    
    def test_reporter_initialization(self, temp_output_dir):
        """Test reporter initialization."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        
        assert reporter.output_dir == Path(temp_output_dir)
        assert (reporter.output_dir / "plots").exists()
        assert (reporter.output_dir / "data").exists()
        assert (reporter.output_dir / "html").exists()
        assert hasattr(reporter, 'timestamp')
    
    def test_reporter_default_directory(self):
        """Test reporter with default directory."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            reporter = ExperimentReporter()
            
            assert reporter.output_dir == Path("reports")
            mock_mkdir.assert_called()
    
    def setup_mock_benchmark_evaluator(self):
        """Setup a mock benchmark evaluator with sample results."""
        evaluator = Mock(spec=BenchmarkEvaluator)
        
        # Mock results
        evaluator.results = {
            'mlp_medium': {
                'accuracy': 0.950,
                'precision_macro': 0.940,
                'recall_macro': 0.930,
                'f1_macro': 0.940,
                'training_time': 120.5,
                'inference_time': 0.005,
                'total_parameters': 50000
            },
            'cnn_simple': {
                'accuracy': 0.970,
                'precision_macro': 0.960,
                'recall_macro': 0.970,
                'f1_macro': 0.960,
                'training_time': 200.3,
                'inference_time': 0.008,
                'total_parameters': 34826
            },
            'xgboost_fast': {
                'accuracy': 0.920,
                'precision_macro': 0.910,
                'recall_macro': 0.920,
                'f1_macro': 0.910,
                'training_time': 45.2,
                'inference_time': 0.003
            }
        }
        
        # Mock get_comparison_table method
        def mock_get_comparison_table(metrics=None):
            if metrics is None:
                metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                          'training_time', 'inference_time']
            
            data = []
            for model_name, results in evaluator.results.items():
                row = {'Model': model_name}
                for metric in metrics:
                    row[metric] = results.get(metric, None)
                data.append(row)
            
            return pd.DataFrame(data)
        
        evaluator.get_comparison_table.side_effect = mock_get_comparison_table
        
        # Mock get_best_models method
        def mock_get_best_models(metric, top_k=3):
            model_scores = []
            for model_name, results in evaluator.results.items():
                if metric in results:
                    model_scores.append((model_name, results[metric]))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            return model_scores[:top_k]
        
        evaluator.get_best_models.side_effect = mock_get_best_models
        
        return evaluator
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_all_plots(self, mock_close, mock_savefig, temp_output_dir):
        """Test generating all plots."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        plots_info = reporter._generate_all_plots(evaluator)
        
        assert isinstance(plots_info, dict)
        assert 'accuracy_comparison' in plots_info
        assert 'f1_comparison' in plots_info
        assert 'time_vs_accuracy' in plots_info
        assert 'metrics_heatmap' in plots_info
        assert 'model_complexity' in plots_info
        
        # Check that files have correct extensions
        for plot_path in plots_info.values():
            assert plot_path.endswith('.png')
        
        # Verify matplotlib functions were called
        assert mock_savefig.call_count >= 5  # At least 5 plots
        assert mock_close.call_count >= 5
    
    def test_generate_all_tables(self, temp_output_dir):
        """Test generating all tables."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        tables_info = reporter._generate_all_tables(evaluator)
        
        assert isinstance(tables_info, dict)
        assert 'main_table' in tables_info
        assert 'detailed_table' in tables_info
        assert 'rankings' in tables_info
        
        # Check table structure
        main_table = tables_info['main_table']
        assert isinstance(main_table, pd.DataFrame)
        assert len(main_table) == 3  # Three models
        assert 'Model' in main_table.columns
        assert 'accuracy' in main_table.columns
        
        # Check rankings
        rankings = tables_info['rankings']
        assert 'accuracy' in rankings
        assert 'f1_macro' in rankings
        assert isinstance(rankings['accuracy'], list)
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_generate_interactive_dashboard(self, mock_write_html, temp_output_dir):
        """Test generating interactive dashboard."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        dashboard_path = reporter._generate_interactive_dashboard(evaluator)
        
        assert isinstance(dashboard_path, str)
        assert dashboard_path.endswith('.html')
        assert 'dashboard_' in dashboard_path
        
        # Verify Plotly write_html was called
        mock_write_html.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_generate_full_report(self, mock_write_html, mock_close, mock_savefig, temp_output_dir):
        """Test generating full report."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        experiment_config = {
            'model_types': ['mlp', 'cnn', 'xgboost'],
            'hyperparameter_search': 'focused',
            'total_experiments': 24
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            report_path = reporter.generate_full_report(
                benchmark_evaluator=evaluator,
                experiment_config=experiment_config,
                include_interactive=True
            )
        
        assert isinstance(report_path, str)
        assert report_path.endswith('.html')
        
        # Verify file operations
        mock_file.assert_called()  # HTML report was written
        mock_write_html.assert_called_once()  # Interactive dashboard
        
        # Verify plots were generated
        assert mock_savefig.call_count >= 5
        assert mock_close.call_count >= 5
    
    def test_generate_full_report_no_results(self, temp_output_dir):
        """Test generating report with no results."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = Mock(spec=BenchmarkEvaluator)
        evaluator.results = {}
        
        with pytest.raises(ValueError, match="No model results found"):
            reporter.generate_full_report(evaluator)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_metric_comparison(self, mock_close, mock_savefig, temp_output_dir):
        """Test metric comparison plotting."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        save_path = str(Path(temp_output_dir) / "test_plot.png")
        
        reporter._plot_metric_comparison(
            evaluator, 
            metric='accuracy',
            title='Test Accuracy Comparison',
            save_path=save_path
        )
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_time_vs_accuracy(self, mock_close, mock_savefig, temp_output_dir):
        """Test time vs accuracy plotting."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        save_path = str(Path(temp_output_dir) / "test_scatter.png")
        
        reporter._plot_time_vs_accuracy(evaluator, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        mock_close.assert_called_once()
    
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_metrics_heatmap(self, mock_close, mock_savefig, mock_heatmap, temp_output_dir):
        """Test metrics heatmap plotting."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        save_path = str(Path(temp_output_dir) / "test_heatmap.png")
        
        reporter._plot_metrics_heatmap(evaluator, save_path=save_path)
        
        mock_heatmap.assert_called_once()
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_model_complexity(self, mock_close, mock_savefig, temp_output_dir):
        """Test model complexity plotting."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        save_path = str(Path(temp_output_dir) / "test_complexity.png")
        
        reporter._plot_model_complexity(evaluator, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        mock_close.assert_called_once()
    
    def test_save_experiment_data(self, temp_output_dir):
        """Test saving experiment data."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        experiment_config = {'test': 'config'}
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            reporter._save_experiment_data(evaluator, experiment_config)
            
            # Verify file was opened for writing
            mock_file.assert_called_once()
            
            # Verify JSON was dumped
            mock_json_dump.assert_called_once()
            
            # Check the data structure passed to json.dump
            call_args = mock_json_dump.call_args[0]
            data = call_args[0]
            
            assert 'timestamp' in data
            assert 'experiment_config' in data
            assert 'results' in data
            assert 'summary' in data
            assert data['experiment_config'] == experiment_config
    
    def test_html_report_generation(self, temp_output_dir):
        """Test HTML report generation."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        # Mock components
        plots_info = {
            'accuracy_comparison': f'{temp_output_dir}/plots/accuracy.png',
            'f1_comparison': f'{temp_output_dir}/plots/f1.png'
        }
        
        tables_info = {
            'main_table': pd.DataFrame({
                'Model': ['mlp_medium', 'cnn_simple'],
                'accuracy': [0.95, 0.97]
            }),
            'detailed_table': pd.DataFrame({
                'Model': ['mlp_medium', 'cnn_simple'],
                'accuracy': [0.95, 0.97],
                'f1_macro': [0.94, 0.96]
            }),
            'rankings': {'accuracy': [('cnn_simple', 0.97), ('mlp_medium', 0.95)]}
        }
        
        interactive_info = f'{temp_output_dir}/html/dashboard.html'
        experiment_config = {'models': 2, 'epochs': 10}
        
        with patch('builtins.open', mock_open()) as mock_file:
            report_path = reporter._generate_html_report(
                evaluator, plots_info, tables_info, 
                interactive_info, experiment_config
            )
            
            assert isinstance(report_path, Path)
            assert report_path.name.startswith('experiment_report_')
            assert report_path.suffix == '.html'
            
            # Verify file was written
            mock_file.assert_called_once()
            
            # Check write calls contain expected content
            write_calls = [call for call in mock_file().write.call_args_list]
            html_content = ''.join(call[0][0] for call in write_calls)
            
            assert '<!DOCTYPE html>' in html_content
            assert 'MNIST Classifier Experiment Report' in html_content
            assert 'cnn_simple' in html_content
            assert 'mlp_medium' in html_content


class TestFactoryFunctions:
    """Test factory functions for reporting."""
    
    def test_create_experiment_reporter(self, temp_output_dir):
        """Test create_experiment_reporter factory function."""
        reporter = create_experiment_reporter(output_dir=temp_output_dir)
        
        assert isinstance(reporter, ExperimentReporter)
        assert reporter.output_dir == Path(temp_output_dir)
    
    def test_create_experiment_reporter_default(self):
        """Test factory function with default parameters."""
        with patch('pathlib.Path.mkdir'):
            reporter = create_experiment_reporter()
            
            assert isinstance(reporter, ExperimentReporter)
            assert reporter.output_dir == Path("reports")


class TestQuickReport:
    """Test quick report functionality."""
    
    def test_generate_quick_report(self):
        """Test quick report generation."""
        # Setup mock evaluator
        evaluator = Mock(spec=BenchmarkEvaluator)
        evaluator.results = {
            'model_a': {
                'accuracy': 0.95,
                'precision_macro': 0.94,
                'recall_macro': 0.93,
                'f1_macro': 0.94
            },
            'model_b': {
                'accuracy': 0.97,
                'precision_macro': 0.96,
                'recall_macro': 0.97,
                'f1_macro': 0.96
            }
        }
        
        # Mock methods
        evaluator.get_best_models.return_value = [
            ('model_b', 0.97),
            ('model_a', 0.95)
        ]
        
        evaluator.get_comparison_table.return_value = pd.DataFrame({
            'Model': ['model_a', 'model_b'],
            'accuracy': [0.95, 0.97],
            'precision_macro': [0.94, 0.96]
        })
        
        report = generate_quick_report(evaluator)
        
        assert isinstance(report, str)
        assert "MNIST Classifier Quick Report" in report
        assert "Top Models by Accuracy" in report
        assert "model_a" in report
        assert "model_b" in report
        assert "0.97" in report  # Best accuracy
    
    def test_generate_quick_report_empty(self):
        """Test quick report with no results."""
        evaluator = Mock(spec=BenchmarkEvaluator)
        evaluator.results = {}
        
        report = generate_quick_report(evaluator)
        
        assert isinstance(report, str)
        assert "No model results available" in report


class TestReportingIntegration:
    """Integration tests for reporting components."""
    
    def test_end_to_end_reporting(self, temp_output_dir):
        """Test end-to-end reporting workflow."""
        # Create a real BenchmarkEvaluator with sample data
        from mnist_classifier.utils.metrics import BenchmarkEvaluator
        
        evaluator = BenchmarkEvaluator(num_classes=10)
        
        # Add sample results
        np.random.seed(42)
        y_true = np.random.randint(0, 10, 1000)
        
        # Model 1: 90% accuracy
        y_pred_1 = y_true.copy()
        error_indices = np.random.choice(1000, 100, replace=False)
        y_pred_1[error_indices] = np.random.randint(0, 10, 100)
        
        evaluator.add_model_results(
            "mlp_test", y_true, y_pred_1,
            training_time=120.5, inference_time=0.005
        )
        
        # Model 2: 95% accuracy  
        y_pred_2 = y_true.copy()
        error_indices = np.random.choice(1000, 50, replace=False)
        y_pred_2[error_indices] = np.random.randint(0, 10, 50)
        
        evaluator.add_model_results(
            "cnn_test", y_true, y_pred_2,
            training_time=200.3, inference_time=0.008
        )
        
        # Generate report
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('plotly.graph_objects.Figure.write_html'), \
             patch('builtins.open', mock_open()):
            
            report_path = reporter.generate_full_report(
                benchmark_evaluator=evaluator,
                experiment_config={'test': True},
                include_interactive=True
            )
        
        assert isinstance(report_path, str)
        assert 'experiment_report_' in report_path
        
        # Test quick report
        quick_report = generate_quick_report(evaluator)
        assert "cnn_test" in quick_report
        assert "mlp_test" in quick_report
    
    def test_report_with_probabilities(self, temp_output_dir):
        """Test reporting with probability predictions."""
        from mnist_classifier.utils.metrics import BenchmarkEvaluator
        
        evaluator = BenchmarkEvaluator(num_classes=3)
        
        # Create sample data with probabilities
        y_true = np.array([0, 1, 2, 0, 1, 2] * 10)  # 60 samples
        y_pred = y_true.copy()
        # Add some errors
        y_pred[::10] = (y_pred[::10] + 1) % 3  # 6 errors
        
        # Create probability matrix
        n_samples = len(y_true)
        y_pred_proba = np.random.rand(n_samples, 3)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        
        # Make probabilities align with predictions
        for i in range(n_samples):
            y_pred_proba[i, y_pred[i]] = max(y_pred_proba[i, y_pred[i]], 0.6)
            y_pred_proba[i] = y_pred_proba[i] / y_pred_proba[i].sum()
        
        # Use evaluator's add_model_results which should handle probabilities
        # This would require extending the method to accept probabilities
        # For now, just test that we can generate reports with the existing data
        
        evaluator.add_model_results(
            "prob_model", y_true, y_pred,
            training_time=100.0, inference_time=0.01
        )
        
        # Generate reports
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('plotly.graph_objects.Figure.write_html'), \
             patch('builtins.open', mock_open()):
            
            report_path = reporter.generate_full_report(evaluator)
        
        assert isinstance(report_path, str)
        
        quick_report = generate_quick_report(evaluator)
        assert "prob_model" in quick_report


class TestReportingErrorHandling:
    """Test error handling in reporting components."""
    
    def test_reporter_invalid_directory(self):
        """Test reporter with invalid directory path."""
        with patch('pathlib.Path.mkdir', side_effect=PermissionError):
            with pytest.raises(PermissionError):
                ExperimentReporter(output_dir="/invalid/path")
    
    def test_plot_generation_failure(self, temp_output_dir):
        """Test handling of plot generation failures."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        with patch('matplotlib.pyplot.savefig', side_effect=Exception("Plot error")):
            # Should not raise exception, but handle gracefully
            plots_info = reporter._generate_all_plots(evaluator)
            
            # Should still return dictionary structure
            assert isinstance(plots_info, dict)
    
    def test_html_generation_failure(self, temp_output_dir):
        """Test handling of HTML generation failures."""
        reporter = ExperimentReporter(output_dir=temp_output_dir)
        evaluator = self.setup_mock_benchmark_evaluator()
        
        plots_info = {'test_plot': 'test_path'}
        tables_info = {'main_table': pd.DataFrame(), 'detailed_table': pd.DataFrame(), 'rankings': {}}
        
        with patch('builtins.open', side_effect=IOError("Write error")):
            with pytest.raises(IOError):
                reporter._generate_html_report(
                    evaluator, plots_info, tables_info, None, None
                )