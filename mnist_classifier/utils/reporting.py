"""Performance comparison and reporting system for MNIST classifier experiments."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

from .metrics import BenchmarkEvaluator


class ExperimentReporter:
    """Generates comprehensive experiment reports with visualizations."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize experiment reporter.
        
        Args:
            output_dir: Directory to save reports and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
        
    def generate_full_report(self, benchmark_evaluator: BenchmarkEvaluator,
                           experiment_config: Optional[Dict] = None,
                           include_interactive: bool = True) -> str:
        """
        Generate a comprehensive experiment report.
        
        Args:
            benchmark_evaluator: Evaluator with all model results
            experiment_config: Configuration details of the experiment
            include_interactive: Whether to include interactive plots
            
        Returns:
            Path to the generated report
        """
        if not benchmark_evaluator.results:
            raise ValueError("No model results found in benchmark evaluator")
        
        print("Generating comprehensive experiment report...")
        
        # Generate all components
        plots_info = self._generate_all_plots(benchmark_evaluator)
        tables_info = self._generate_all_tables(benchmark_evaluator)
        
        if include_interactive:
            interactive_info = self._generate_interactive_dashboard(benchmark_evaluator)
        else:
            interactive_info = None
        
        # Generate main HTML report
        report_path = self._generate_html_report(
            benchmark_evaluator, plots_info, tables_info, 
            interactive_info, experiment_config
        )
        
        # Save summary data
        self._save_experiment_data(benchmark_evaluator, experiment_config)
        
        print(f"Report generated: {report_path}")
        return str(report_path)
    
    def _generate_all_plots(self, benchmark_evaluator: BenchmarkEvaluator) -> Dict[str, str]:
        """Generate all static plots and return their paths."""
        plots_dir = self.output_dir / "plots"
        plots_info = {}
        
        # 1. Model accuracy comparison
        accuracy_path = plots_dir / f"accuracy_comparison_{self.timestamp}.png"
        self._plot_metric_comparison(benchmark_evaluator, 'accuracy', 
                                   title="Model Accuracy Comparison", 
                                   save_path=str(accuracy_path))
        plots_info['accuracy_comparison'] = str(accuracy_path)
        
        # 2. F1-score comparison
        f1_path = plots_dir / f"f1_comparison_{self.timestamp}.png"
        self._plot_metric_comparison(benchmark_evaluator, 'f1_macro',
                                   title="F1-Score Comparison", 
                                   save_path=str(f1_path))
        plots_info['f1_comparison'] = str(f1_path)
        
        # 3. Training time vs accuracy scatter
        scatter_path = plots_dir / f"time_vs_accuracy_{self.timestamp}.png"
        self._plot_time_vs_accuracy(benchmark_evaluator, save_path=str(scatter_path))
        plots_info['time_vs_accuracy'] = str(scatter_path)
        
        # 4. Comprehensive metrics heatmap
        heatmap_path = plots_dir / f"metrics_heatmap_{self.timestamp}.png"
        self._plot_metrics_heatmap(benchmark_evaluator, save_path=str(heatmap_path))
        plots_info['metrics_heatmap'] = str(heatmap_path)
        
        # 5. Model complexity comparison
        complexity_path = plots_dir / f"model_complexity_{self.timestamp}.png"
        self._plot_model_complexity(benchmark_evaluator, save_path=str(complexity_path))
        plots_info['model_complexity'] = str(complexity_path)
        
        return plots_info
    
    def _generate_all_tables(self, benchmark_evaluator: BenchmarkEvaluator) -> Dict[str, Any]:
        """Generate all data tables."""
        
        # Main comparison table
        main_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                       'training_time', 'inference_time']
        main_table = benchmark_evaluator.get_comparison_table(main_metrics)
        
        # Detailed metrics table
        detailed_metrics = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
                          'recall_macro', 'recall_micro', 'recall_weighted',
                          'f1_macro', 'f1_micro', 'f1_weighted', 'roc_auc_macro']
        detailed_table = benchmark_evaluator.get_comparison_table(detailed_metrics)
        
        # Performance ranking by different metrics
        rankings = {}
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            rankings[metric] = benchmark_evaluator.get_best_models(metric, top_k=len(benchmark_evaluator.results))
        
        # Save tables
        tables_dir = self.output_dir / "data"
        main_table.to_csv(tables_dir / f"main_results_{self.timestamp}.csv", index=False)
        detailed_table.to_csv(tables_dir / f"detailed_results_{self.timestamp}.csv", index=False)
        
        return {
            'main_table': main_table,
            'detailed_table': detailed_table,
            'rankings': rankings
        }
    
    def _generate_interactive_dashboard(self, benchmark_evaluator: BenchmarkEvaluator) -> str:
        """Generate interactive Plotly dashboard."""
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy Comparison', 'Training Time vs Accuracy',
                           'Metrics Radar Chart', 'Model Performance Overview'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatterpolar"}, {"type": "bar"}]]
        )
        
        models = list(benchmark_evaluator.results.keys())
        results = benchmark_evaluator.results
        
        # 1. Accuracy bar chart
        accuracies = [results[model].get('accuracy', 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy',
                   text=[f'{acc:.3f}' for acc in accuracies], textposition='auto'),
            row=1, col=1
        )
        
        # 2. Training time vs accuracy scatter
        train_times = [results[model].get('training_time', 0) for model in models]
        fig.add_trace(
            go.Scatter(x=train_times, y=accuracies, mode='markers+text',
                      text=models, textposition='top center',
                      name='Models', marker=dict(size=10)),
            row=1, col=2
        )
        
        # 3. Radar chart for top model
        best_model = benchmark_evaluator.get_best_models('accuracy', top_k=1)[0][0]
        best_results = results[best_model]
        radar_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        radar_values = [best_results.get(metric, 0) for metric in radar_metrics]
        
        fig.add_trace(
            go.Scatterpolar(r=radar_values, theta=radar_metrics,
                           fill='toself', name=f'{best_model} (Best)'),
            row=2, col=1
        )
        
        # 4. F1-score comparison
        f1_scores = [results[model].get('f1_macro', 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1-Score',
                   text=[f'{f1:.3f}' for f1 in f1_scores], textposition='auto'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="MNIST Classifier Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save interactive dashboard
        dashboard_path = self.output_dir / "html" / f"dashboard_{self.timestamp}.html"
        fig.write_html(str(dashboard_path))
        
        return str(dashboard_path)
    
    def _plot_metric_comparison(self, benchmark_evaluator: BenchmarkEvaluator, 
                              metric: str, title: str, save_path: str):
        """Plot comparison of a specific metric across models."""
        models = list(benchmark_evaluator.results.keys())
        values = [benchmark_evaluator.results[model].get(metric, 0) for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, alpha=0.8, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_vs_accuracy(self, benchmark_evaluator: BenchmarkEvaluator, save_path: str):
        """Plot training time vs accuracy scatter plot."""
        models = list(benchmark_evaluator.results.keys())
        results = benchmark_evaluator.results
        
        train_times = [results[model].get('training_time', 0) for model in models]
        accuracies = [results[model].get('accuracy', 0) for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(train_times, accuracies, s=100, alpha=0.7, c='coral', edgecolors='black')
        
        # Add model labels
        for i, model in enumerate(models):
            plt.annotate(model, (train_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_heatmap(self, benchmark_evaluator: BenchmarkEvaluator, save_path: str):
        """Plot heatmap of all metrics across models."""
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        models = list(benchmark_evaluator.results.keys())
        
        # Create data matrix
        data = []
        for model in models:
            row = [benchmark_evaluator.results[model].get(metric, 0) for metric in metrics]
            data.append(row)
        
        data_array = np.array(data)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(data_array, annot=True, fmt='.4f', 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=models, cmap='viridis')
        plt.title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_complexity(self, benchmark_evaluator: BenchmarkEvaluator, save_path: str):
        """Plot model complexity comparison."""
        models = list(benchmark_evaluator.results.keys())
        results = benchmark_evaluator.results
        
        # Get model parameters if available
        param_counts = []
        train_times = []
        
        for model in models:
            result = results[model]
            param_counts.append(result.get('total_parameters', 0))
            train_times.append(result.get('training_time', 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Parameters comparison
        if any(p > 0 for p in param_counts):
            bars1 = ax1.bar(models, param_counts, alpha=0.8, color='lightgreen')
            ax1.set_title('Model Complexity (Parameters)')
            ax1.set_ylabel('Number of Parameters')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars1, param_counts):
                if value > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:,}', ha='center', va='bottom', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'Parameter count data not available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Model Complexity (Parameters)')
        
        # Training time comparison
        bars2 = ax2.bar(models, train_times, alpha=0.8, color='lightcoral')
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, train_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.01,
                    f'{value:.1f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, benchmark_evaluator: BenchmarkEvaluator,
                            plots_info: Dict[str, str], tables_info: Dict[str, Any],
                            interactive_info: Optional[str], 
                            experiment_config: Optional[Dict]) -> Path:
        """Generate comprehensive HTML report."""
        
        # Get summary information
        best_models = {
            'accuracy': benchmark_evaluator.get_best_models('accuracy', top_k=3),
            'f1_macro': benchmark_evaluator.get_best_models('f1_macro', top_k=3),
            'training_time': sorted([(model, results.get('training_time', float('inf'))) 
                                   for model, results in benchmark_evaluator.results.items()], 
                                  key=lambda x: x[1])[:3]
        }
        
        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MNIST Classifier Experiment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .subsection {{ margin: 20px 0; }}
        .table-container {{ overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .metric-value {{ font-weight: bold; color: #2563eb; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .summary-box {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .best-model {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        .config-item {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MNIST Classifier Experiment Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Models Evaluated: {len(benchmark_evaluator.results)}</p>
    </div>
    
    <div class="section">
        <h2>🏆 Executive Summary</h2>
        <div class="summary-box">
        """
        
        # Add best models summary
        for metric, models in best_models.items():
            if models:
                html_content += f"<h3>Best {metric.replace('_', ' ').title()}</h3>"
                for i, (model, score) in enumerate(models[:3], 1):
                    html_content += f'<div class="best-model">{i}. <strong>{model}</strong>: {score:.4f}</div>'
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Performance Comparison</h2>
        """
        
        # Add main results table
        main_table_html = tables_info['main_table'].to_html(index=False, classes='', escape=False)
        html_content += f"""
        <div class="subsection">
            <h3>Main Results</h3>
            <div class="table-container">
                {main_table_html}
            </div>
        </div>
        """
        
        # Add plots
        html_content += '<div class="subsection"><h3>Performance Visualizations</h3>'
        
        for plot_name, plot_path in plots_info.items():
            plot_filename = Path(plot_path).name
            html_content += f"""
            <div class="plot-container">
                <h4>{plot_name.replace('_', ' ').title()}</h4>
                <img src="plots/{plot_filename}" alt="{plot_name}">
            </div>
            """
        
        html_content += '</div>'
        
        # Add detailed metrics
        detailed_table_html = tables_info['detailed_table'].to_html(index=False, classes='', escape=False)
        html_content += f"""
        <div class="subsection">
            <h3>Detailed Metrics</h3>
            <div class="table-container">
                {detailed_table_html}
            </div>
        </div>
    </div>
    """
        
        # Add experiment configuration if provided
        if experiment_config:
            html_content += """
            <div class="section">
                <h2>⚙️ Experiment Configuration</h2>
                <div class="summary-box">
            """
            for key, value in experiment_config.items():
                html_content += f'<div class="config-item"><strong>{key}:</strong> {value}</div>'
            html_content += '</div></div>'
        
        # Add interactive dashboard link if available
        if interactive_info:
            dashboard_filename = Path(interactive_info).name
            html_content += f"""
            <div class="section">
                <h2>📈 Interactive Dashboard</h2>
                <p><a href="html/{dashboard_filename}" target="_blank">Open Interactive Dashboard</a></p>
            </div>
            """
        
        # Close HTML
        html_content += """
    <div class="section">
        <h2>📝 Notes</h2>
        <ul>
            <li>All metrics are calculated on the test set</li>
            <li>Training times include full hyperparameter optimization</li>
            <li>Models are compared using consistent evaluation protocols</li>
            <li>Interactive dashboard provides additional exploration capabilities</li>
        </ul>
    </div>
</body>
</html>
        """
        
        # Save HTML report
        report_path = self.output_dir / f"experiment_report_{self.timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _save_experiment_data(self, benchmark_evaluator: BenchmarkEvaluator, 
                            experiment_config: Optional[Dict]):
        """Save experiment data in JSON format."""
        
        # Prepare data for JSON serialization
        experiment_data = {
            'timestamp': self.timestamp,
            'experiment_config': experiment_config or {},
            'results': {},
            'summary': {
                'total_models': len(benchmark_evaluator.results),
                'best_accuracy': 0,
                'best_model': None
            }
        }
        
        # Process results (convert numpy types to Python types)
        best_accuracy = 0
        best_model = None
        
        for model_name, results in benchmark_evaluator.results.items():
            processed_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    processed_results[key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32)):
                    processed_results[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    processed_results[key] = float(value)
                else:
                    processed_results[key] = value
            
            experiment_data['results'][model_name] = processed_results
            
            # Track best model
            if results.get('accuracy', 0) > best_accuracy:
                best_accuracy = results.get('accuracy', 0)
                best_model = model_name
        
        experiment_data['summary']['best_accuracy'] = best_accuracy
        experiment_data['summary']['best_model'] = best_model
        
        # Save JSON data
        json_path = self.output_dir / "data" / f"experiment_data_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)


def create_experiment_reporter(output_dir: str = "reports") -> ExperimentReporter:
    """Factory function to create experiment reporter."""
    return ExperimentReporter(output_dir)


def generate_quick_report(benchmark_evaluator: BenchmarkEvaluator) -> str:
    """Generate a quick text summary report."""
    if not benchmark_evaluator.results:
        return "No model results available."
    
    report = "MNIST Classifier Quick Report\n"
    report += "=" * 40 + "\n\n"
    
    # Best models by accuracy
    best_models = benchmark_evaluator.get_best_models('accuracy', top_k=3)
    report += "🏆 Top Models by Accuracy:\n"
    for i, (model, score) in enumerate(best_models, 1):
        report += f"  {i}. {model}: {score:.4f}\n"
    
    report += "\n📊 All Model Results:\n"
    comparison_df = benchmark_evaluator.get_comparison_table()
    report += comparison_df.to_string(index=False)
    
    return report