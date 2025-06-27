"""MLflow integration utilities for experiment tracking."""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import joblib


class MLflowTracker:
    """Handles MLflow experiment tracking and model logging."""
    
    def __init__(self, experiment_name: str = "mnist_classifier", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local file store)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (defaults to local mlruns directory)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local file store in project directory
            mlruns_path = Path.cwd() / "mlruns"
            mlruns_path.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            print(f"Error setting up MLflow experiment: {e}")
            raise
        
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Optional tags to add to the run
            
        Returns:
            Run ID
        """
        run = mlflow.start_run(run_name=run_name, tags=tags)
        print(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric to current run."""
        mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file) to current run."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model, model_type: str, model_name: str = "model", 
                  signature=None, input_example=None):
        """
        Log a model to current run.
        
        Args:
            model: The model to log
            model_type: Type of model ("pytorch", "sklearn", "xgboost")
            model_name: Name for the model artifact
            signature: Model signature
            input_example: Example input for the model
        """
        if model_type.lower() == "pytorch":
            mlflow.pytorch.log_model(
                model, model_name, signature=signature, input_example=input_example
            )
        elif model_type.lower() == "sklearn":
            mlflow.sklearn.log_model(
                model, model_name, signature=signature, input_example=input_example
            )
        elif model_type.lower() == "xgboost":
            mlflow.xgboost.log_model(
                model, model_name, signature=signature, input_example=input_example
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        print("Ended MLflow run")
    
    def get_experiment_runs(self) -> list:
        """Get all runs from the current experiment."""
        return mlflow.search_runs(experiment_ids=[self.experiment_id])
    
    def get_best_run(self, metric_name: str, ascending: bool = False):
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: If True, lower values are better
            
        Returns:
            Best run data
        """
        runs = self.get_experiment_runs()
        if runs.empty:
            return None
        
        runs = runs.dropna(subset=[f"metrics.{metric_name}"])
        if runs.empty:
            return None
        
        best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmin() if ascending 
                           else runs[f"metrics.{metric_name}"].idxmax()]
        return best_run


def setup_mlflow_experiment(experiment_name: str = "mnist_classifier") -> MLflowTracker:
    """
    Set up MLflow experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(experiment_name)


def log_training_metrics(tracker: MLflowTracker, epoch: int, train_loss: float, 
                        train_acc: float, val_loss: float = None, val_acc: float = None):
    """
    Log training metrics for an epoch.
    
    Args:
        tracker: MLflowTracker instance
        epoch: Current epoch number
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss (optional)
        val_acc: Validation accuracy (optional)
    """
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_acc
    }
    
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if val_acc is not None:
        metrics["val_accuracy"] = val_acc
    
    tracker.log_metrics(metrics, step=epoch)


def log_final_metrics(tracker: MLflowTracker, test_metrics: Dict[str, float]):
    """
    Log final test metrics.
    
    Args:
        tracker: MLflowTracker instance
        test_metrics: Dictionary of test metrics
    """
    # Add "test_" prefix to metrics
    final_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    tracker.log_metrics(final_metrics)