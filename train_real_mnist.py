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


def evaluate_pytorch_model(model, test_loader, device):
    """Evaluate PyTorch model and return predictions."""
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.numpy())
    
    return np.array(y_true), np.array(y_pred)


def main():
    """Main training pipeline."""
    print("🚀 MNIST Classifier - Full Model Training with Real Data")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    results_dir = Path("training_results") 
    results_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator()
    
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
        {'variant': 'small', 'epochs': 10, 'lr': 0.001, 'name': 'mlp_small'},
        {'variant': 'medium', 'epochs': 15, 'lr': 0.001, 'name': 'mlp_medium'},
        {'variant': 'large', 'epochs': 20, 'lr': 0.0005, 'name': 'mlp_large'},
    ]
    
    for config in mlp_configs:
        model_name = config['name']
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
        
        # Train model with parameters passed directly
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=1e-4,
            optimizer_type='adam',
            scheduler_type='cosine',
            patience=10,
            save_best=True,
            verbose=True
        )
        
        # Evaluate
        y_true, y_pred = evaluate_pytorch_model(model, test_loader, device)
        accuracy = np.mean(y_pred == y_true)
        training_time = time.time() - start_time
        
        # Get detailed metrics
        metrics = evaluator.evaluate_model(y_true, y_pred, model_name)
        
        print(f"✅ {model_name} Complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall: {metrics['recall_macro']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        result = {
            'model': model_name,
            'accuracy': accuracy,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
        
        # Save model
        torch.save(model.state_dict(), models_dir / f"{model_name}.pth")
    
    # Train CNN models
    print("\n\n🖼️ Training CNN Models")
    print("-" * 40)
    
    cnn_configs = [
        {'variant': 'simple', 'epochs': 10, 'lr': 0.001, 'name': 'cnn_simple'},
        {'variant': 'medium', 'epochs': 15, 'lr': 0.001, 'name': 'cnn_medium'},
        {'variant': 'deep', 'epochs': 20, 'lr': 0.0005, 'name': 'cnn_deep'},
        {'variant': 'modern', 'epochs': 20, 'lr': 0.0005, 'name': 'cnn_modern'},
    ]
    
    for config in cnn_configs:
        model_name = config['name']
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
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=1e-4,
            optimizer_type='adam',
            scheduler_type='cosine',
            patience=10,
            save_best=True,
            verbose=True
        )
        
        # Evaluate
        y_true, y_pred = evaluate_pytorch_model(model, test_loader, device)
        accuracy = np.mean(y_pred == y_true)
        training_time = time.time() - start_time
        
        # Get detailed metrics
        metrics = evaluator.evaluate_model(y_true, y_pred, model_name)
        
        print(f"✅ {model_name} Complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall: {metrics['recall_macro']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        result = {
            'model': model_name,
            'accuracy': accuracy,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
        
        # Save model
        torch.save(model.state_dict(), models_dir / f"{model_name}.pth")
    
    # Train XGBoost models
    print("\n\n🌳 Training XGBoost Models")
    print("-" * 40)
    
    xgb_configs = [
        {'variant': 'fast', 'name': 'xgboost_fast'},
        {'variant': 'balanced', 'name': 'xgboost_balanced'},
        {'variant': 'deep', 'name': 'xgboost_deep'},
    ]
    
    for config in xgb_configs:
        model_name = config['name']
        print(f"\n🔄 Training {model_name}...")
        
        start_time = time.time()
        
        # Create model
        model = create_xgboost_model(variant=config['variant'])
        
        # Create trainer
        trainer = XGBoostTrainer(model=model)
        
        # Train model
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            verbose=True
        )
        
        # Evaluate
        model.fit(X_train, y_train)  # Fit on full training data
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        training_time = time.time() - start_time
        
        # Get detailed metrics
        metrics = evaluator.evaluate_model(y_test, y_pred, model_name)
        
        print(f"✅ {model_name} Complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall: {metrics['recall_macro']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        result = {
            'model': model_name,
            'accuracy': accuracy,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
        
        # Save model
        import joblib
        joblib.dump(model, models_dir / f"{model_name}.pkl")
    
    # Generate final report
    print("\n\n📊 FINAL RESULTS")
    print("=" * 60)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n🏆 Model Performance Ranking:")
    print("-" * 100)
    print(f"{'Model':20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time (s)':>10}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:20} {row['accuracy']:10.4f} {row['precision_macro']:10.4f} "
              f"{row['recall_macro']:10.4f} {row['f1_macro']:10.4f} {row['training_time']:10.1f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
    results_file = results_dir / f"training_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   Models Trained: {len(results_df)}")
    print(f"   Best Accuracy: {results_df['accuracy'].max():.4f}")
    print(f"   Average Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"   Total Training Time: {results_df['training_time'].sum():.1f}s")
    
    # Best model
    best_model = results_df.iloc[0]
    print(f"\n🥇 Best Model: {best_model['model']}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   F1 Score: {best_model['f1_macro']:.4f}")
    print(f"   Training Time: {best_model['training_time']:.1f}s")
    
    print(f"\n✅ Training complete!")
    print(f"📄 Results saved to: {results_file}")
    print(f"🤖 Models saved to: {models_dir}/")


if __name__ == "__main__":
    main()