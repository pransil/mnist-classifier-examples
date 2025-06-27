#!/usr/bin/env python
"""Quick training of all MNIST models with fewer epochs for demonstration."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mnist_classifier.data.loader import MNISTDataLoader
from mnist_classifier.models.mlp import create_mlp_model
from mnist_classifier.models.cnn import create_cnn_model
from mnist_classifier.models.xgboost_model import create_xgboost_model
from mnist_classifier.training.trainer import PyTorchTrainer, XGBoostTrainer


def evaluate_pytorch_model(model, test_loader, device):
    """Evaluate PyTorch model."""
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
    """Calculate metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1_score': f1_score(y_true, y_pred, average='macro')
    }


print("🚀 MNIST Classifier - Quick Training Demo (All Models)")
print("="*60)

# Load data
print("\n📁 Loading MNIST data...")
data_loader = MNISTDataLoader(data_root="data", batch_size=128, val_split=0.1)
train_loader, val_loader, test_loader = data_loader.get_pytorch_loaders()
X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_numpy_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

results = []

# Train each model type with just 2 epochs for quick demo
models = [
    ('mlp_small', 'mlp', 'small'),
    ('mlp_medium', 'mlp', 'medium'),
    ('mlp_large', 'mlp', 'large'),
    ('cnn_simple', 'cnn', 'simple'),
    ('cnn_medium', 'cnn', 'medium'),
    ('cnn_deep', 'cnn', 'deep'),
    ('cnn_modern', 'cnn', 'modern'),
]

print("\n🔄 Training PyTorch models (2 epochs each for demo)...")
for name, model_type, variant in models:
    print(f"\nTraining {name}...")
    start = time.time()
    
    if model_type == 'mlp':
        model = create_mlp_model(variant=variant)
    else:
        model = create_cnn_model(variant=variant)
    
    model = model.to(device)
    trainer = PyTorchTrainer(model=model, device=device)
    
    # Quick training with just 2 epochs
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=0.001,
        verbose=False
    )
    
    # Evaluate
    y_true, y_pred = evaluate_pytorch_model(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    train_time = time.time() - start
    
    print(f"✅ {name}: Accuracy={metrics['accuracy']:.4f}, Time={train_time:.1f}s")
    
    results.append({
        'model': name,
        'training_time': train_time,
        **metrics
    })

# Train XGBoost models
print("\n🔄 Training XGBoost models...")
xgb_models = [
    ('xgboost_fast', 'fast'),
    ('xgboost_balanced', 'balanced'),
    ('xgboost_deep', 'deep'),
]

for name, variant in xgb_models:
    print(f"\nTraining {name}...")
    start = time.time()
    
    model = create_xgboost_model(variant=variant)
    trainer = XGBoostTrainer(model=model)
    
    # Train (limiting to subset for speed)
    subset_size = 10000
    trainer.train(
        X_train=X_train[:subset_size],
        y_train=y_train[:subset_size],
        X_val=X_val[:2000],
        y_val=y_val[:2000],
        verbose=False
    )
    
    # Fit on larger dataset and evaluate
    model.fit(X_train[:subset_size], y_train[:subset_size])
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    train_time = time.time() - start
    
    print(f"✅ {name}: Accuracy={metrics['accuracy']:.4f}, Time={train_time:.1f}s")
    
    results.append({
        'model': name,
        'training_time': train_time,
        **metrics
    })

# Final report
print("\n\n📊 FINAL RESULTS")
print("="*60)

df = pd.DataFrame(results).sort_values('accuracy', ascending=False)

print("\n🏆 Model Rankings:")
print("-"*80)
print(f"{'Model':20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(s)':>10}")
print("-"*80)

for _, row in df.iterrows():
    print(f"{row['model']:20} {row['accuracy']:10.4f} {row['precision']:10.4f} "
          f"{row['recall']:10.4f} {row['f1_score']:10.4f} {row['training_time']:10.1f}")

print(f"\n✅ Training complete! Best model: {df.iloc[0]['model']} "
      f"with {df.iloc[0]['accuracy']:.4f} accuracy")