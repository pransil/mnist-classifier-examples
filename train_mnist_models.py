#!/usr/bin/env python
"""Train all MNIST classifier models using real MNIST data."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from mnist_classifier.data.loader import MNISTDataLoader
from mnist_classifier.models.mlp import create_mlp_model
from mnist_classifier.models.cnn import create_cnn_model
from mnist_classifier.models.xgboost_model import create_xgboost_model
from mnist_classifier.training.trainer import PyTorchTrainer, XGBoostTrainer
from mnist_classifier.utils.metrics import BenchmarkEvaluator
from mnist_classifier.utils.reporting import ExperimentReporter


def main():
    """Main training pipeline."""
    print("🚀 MNIST Classifier - Full Model Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    results_dir = Path("training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize components
    evaluator = BenchmarkEvaluator()
    reporter = ExperimentReporter(output_dir=str(results_dir))
    
    # Load MNIST data
    print("\n📁 Loading MNIST dataset...")
    data_loader = MNISTDataLoader(data_root="data", batch_size=64, val_split=0.1)
    
    # Get PyTorch data loaders
    train_loader, val_loader, test_loader = data_loader.get_pytorch_loaders(use_augmentation=False)
    
    # Get numpy arrays for XGBoost
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_numpy_data()
    
    print(f"✅ Data loaded successfully!")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Track all results
    all_results = []
    
    # Train MLP models
    print("\n🧠 Training MLP Models")
    print("-" * 40)
    
    mlp_configs = [
        {'variant': 'small', 'epochs': 10, 'lr': 0.001},
        {'variant': 'medium', 'epochs': 15, 'lr': 0.001},
        {'variant': 'large', 'epochs': 20, 'lr': 0.0005},
    ]
    
    for config in mlp_configs:
        model_name = f"mlp_{config['variant']}"
        print(f"\n🔄 Training {model_name}...")
        
        start_time = time.time()
        
        # Create model
        model = create_mlp_model(variant=config['variant'])
        model = model.to(device)
        
        # Create trainer
        trainer = PyTorchTrainer(
            model=model,
            device=device
        )
        
        # Setup training
        trainer.setup_training(
            lr=config['lr'],
            weight_decay=1e-4,
            optimizer_type='adam',
            scheduler_type='cosine',
            epochs=config['epochs']
        )
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name
        )
        
        # Evaluate
        test_loss, test_acc = trainer.evaluate(test_loader)
        training_time = time.time() - start_time
        
        # Get detailed metrics
        model.eval()
        with torch.no_grad():
            y_pred = []
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
        
        y_pred = np.array(y_pred)
        metrics = evaluator.evaluate_model(y_test, y_pred, model_name)
        
        print(f"✅ {model_name} Complete: {test_acc:.4f} accuracy ({training_time:.1f}s)")
        
        result = {
            'model': model_name,
            'accuracy': test_acc,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
    
    # Train CNN models
    print("\n\n🖼️ Training CNN Models")
    print("-" * 40)
    
    cnn_configs = [
        {'variant': 'simple', 'epochs': 10, 'lr': 0.001},
        {'variant': 'medium', 'epochs': 15, 'lr': 0.001},
        {'variant': 'deep', 'epochs': 20, 'lr': 0.0005},
        {'variant': 'modern', 'epochs': 20, 'lr': 0.0005},
    ]
    
    for config in cnn_configs:
        model_name = f"cnn_{config['variant']}"
        print(f"\n🔄 Training {model_name}...")
        
        start_time = time.time()
        
        # Create model
        model = create_cnn_model(variant=config['variant'])
        model = model.to(device)
        
        # Create trainer
        trainer = PyTorchTrainer(
            model=model,
            device=device
        )
        
        # Setup training
        trainer.setup_training(
            lr=config['lr'],
            weight_decay=1e-4,
            optimizer_type='adam',
            scheduler_type='cosine',
            epochs=config['epochs']
        )
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name
        )
        
        # Evaluate
        test_loss, test_acc = trainer.evaluate(test_loader)
        training_time = time.time() - start_time
        
        # Get detailed metrics
        model.eval()
        with torch.no_grad():
            y_pred = []
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
        
        y_pred = np.array(y_pred)
        metrics = evaluator.evaluate_model(y_test, y_pred, model_name)
        
        print(f"✅ {model_name} Complete: {test_acc:.4f} accuracy ({training_time:.1f}s)")
        
        result = {
            'model': model_name,
            'accuracy': test_acc,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
    
    # Train XGBoost models
    print("\n\n🌳 Training XGBoost Models")
    print("-" * 40)
    
    xgb_configs = [
        {'variant': 'fast', 'n_estimators': 50},
        {'variant': 'balanced', 'n_estimators': 100},
        {'variant': 'deep', 'n_estimators': 200},
    ]
    
    for config in xgb_configs:
        model_name = f"xgboost_{config['variant']}"
        print(f"\n🔄 Training {model_name}...")
        
        start_time = time.time()
        
        # Create model
        model = create_xgboost_model(variant=config['variant'])
        
        # Create trainer
        trainer = XGBoostTrainer(
            model=model
        )
        
        # Train model
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name
        )
        
        # Evaluate
        y_pred = trainer.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        training_time = time.time() - start_time
        
        # Get detailed metrics
        metrics = evaluator.evaluate_model(y_test, y_pred, model_name)
        
        print(f"✅ {model_name} Complete: {accuracy:.4f} accuracy ({training_time:.1f}s)")
        
        result = {
            'model': model_name,
            'accuracy': accuracy,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
    
    # Generate final report
    print("\n\n📊 FINAL RESULTS")
    print("=" * 60)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n🏆 Model Performance Ranking:")
    print(results_df[['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'training_time']].to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
    results_df.to_csv(results_dir / f"training_results_{timestamp}.csv", index=False)
    
    # Generate comprehensive report
    reporter.create_experiment_report(
        results=all_results,
        experiment_name="MNIST Full Training",
        dataset_info={
            'name': 'MNIST',
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'num_classes': 10
        }
    )
    
    print(f"\n✅ Training complete!")
    print(f"📄 Results saved to: {results_dir}")
    print(f"🤖 Models saved to: {models_dir}")
    
    # Best model
    best_model = results_df.iloc[0]
    print(f"\n🥇 Best Model: {best_model['model']}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   F1 Score: {best_model['f1_macro']:.4f}")
    print(f"   Training Time: {best_model['training_time']:.1f}s")


if __name__ == "__main__":
    main()