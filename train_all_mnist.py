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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from mnist_classifier.utils.notifications import prompt_with_bell


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


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }


def main():
    """Main training pipeline."""
    print("🚀 MNIST Classifier - Training All Models with Real Data")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    results_dir = Path("training_results") 
    results_dir.mkdir(exist_ok=True)
    
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
    
    # MLP Models
    print("\n" + "="*60)
    print("🧠 TRAINING MLP MODELS")
    print("="*60)
    
    mlp_configs = [
        {'variant': 'small', 'epochs': 5, 'lr': 0.001, 'name': 'mlp_small'},
        {'variant': 'medium', 'epochs': 5, 'lr': 0.001, 'name': 'mlp_medium'},
        {'variant': 'large', 'epochs': 5, 'lr': 0.0005, 'name': 'mlp_large'},
    ]
    
    for config in mlp_configs:
        model_name = config['name']
        print(f"\n{'='*40}")
        print(f"🔄 Training {model_name}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        # Create model
        model = create_mlp_model(variant=config['variant'])
        model = model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
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
        metrics = calculate_metrics(y_true, y_pred)
        training_time = time.time() - start_time
        
        print(f"\n✅ {model_name} RESULTS:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall: {metrics['recall_macro']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        result = {
            'model': model_name,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
        
        # Save model
        torch.save(model.state_dict(), models_dir / f"{model_name}.pth")
    
    # CNN Models
    print("\n\n" + "="*60)
    print("🖼️ TRAINING CNN MODELS")
    print("="*60)
    
    cnn_configs = [
        {'variant': 'simple', 'epochs': 5, 'lr': 0.001, 'name': 'cnn_simple'},
        {'variant': 'medium', 'epochs': 5, 'lr': 0.001, 'name': 'cnn_medium'},
        {'variant': 'deep', 'epochs': 5, 'lr': 0.0005, 'name': 'cnn_deep'},
        {'variant': 'modern', 'epochs': 5, 'lr': 0.0005, 'name': 'cnn_modern'},
    ]
    
    for config in cnn_configs:
        model_name = config['name']
        print(f"\n{'='*40}")
        print(f"🔄 Training {model_name}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        # Create model
        model = create_cnn_model(variant=config['variant'])
        model = model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
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
        metrics = calculate_metrics(y_true, y_pred)
        training_time = time.time() - start_time
        
        print(f"\n✅ {model_name} RESULTS:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall: {metrics['recall_macro']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        result = {
            'model': model_name,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
        
        # Save model
        torch.save(model.state_dict(), models_dir / f"{model_name}.pth")
    
    # XGBoost Models
    print("\n\n" + "="*60)
    print("🌳 TRAINING XGBOOST MODELS")
    print("="*60)
    
    xgb_configs = [
        {'variant': 'fast', 'name': 'xgboost_fast'},
        {'variant': 'balanced', 'name': 'xgboost_balanced'},
        {'variant': 'deep', 'name': 'xgboost_deep'},
    ]
    
    for config in xgb_configs:
        model_name = config['name']
        print(f"\n{'='*40}")
        print(f"🔄 Training {model_name}")
        print(f"{'='*40}")
        
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
        metrics = calculate_metrics(y_test, y_pred)
        training_time = time.time() - start_time
        
        print(f"\n✅ {model_name} RESULTS:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision_macro']:.4f}")
        print(f"   Recall: {metrics['recall_macro']:.4f}")
        print(f"   F1-Score: {metrics['f1_macro']:.4f}")
        print(f"   Training Time: {training_time:.1f}s")
        
        result = {
            'model': model_name,
            'training_time': training_time,
            **metrics
        }
        all_results.append(result)
        
        # Save model
        import joblib
        joblib.dump(model, models_dir / f"{model_name}.pkl")
    
    # Final Report
    print("\n\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n🏆 Model Performance Ranking:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Model':20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time (s)':>10}")
    print("-" * 100)
    
    for idx, row in results_df.iterrows():
        rank = list(results_df.index).index(idx) + 1
        print(f"{rank:<6} {row['model']:20} {row['accuracy']:10.4f} {row['precision_macro']:10.4f} "
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
    
    # Best models by category
    print("\n🥇 Best Models by Category:")
    for model_type in ['mlp', 'cnn', 'xgboost']:
        type_df = results_df[results_df['model'].str.startswith(model_type)]
        if not type_df.empty:
            best = type_df.iloc[0]
            print(f"   {model_type.upper()}: {best['model']} (Accuracy: {best['accuracy']:.4f})")
    
    print(f"\n✅ All training complete!")
    print(f"📄 Results saved to: {results_file}")
    print(f"🤖 Models saved to: {models_dir}/")
    
    print("\n" + "="*60)
    print("🎉 MNIST CLASSIFIER TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()