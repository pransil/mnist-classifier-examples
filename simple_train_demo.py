#!/usr/bin/env python
"""Simple training demo for all MNIST models."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mnist_classifier.data.loader import MNISTDataLoader
from mnist_classifier.models.mlp import create_mlp_model
from mnist_classifier.models.cnn import create_cnn_model
from mnist_classifier.models.xgboost_model import create_xgboost_model


def simple_train_pytorch(model, train_loader, val_loader, epochs=2, lr=0.001, device='cpu'):
    """Simple PyTorch training loop."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return model


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate PyTorch model."""
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            y_pred.extend(pred.cpu().numpy().flatten())
            y_true.extend(target.numpy())
    
    return np.array(y_true), np.array(y_pred)


def main():
    print("🚀 MNIST Classifier - Simple Training Demo")
    print("=" * 50)
    
    # Load data with single-threaded loading
    print("\n📁 Loading MNIST data...")
    data_loader = MNISTDataLoader(data_root="data", batch_size=64, val_split=0.1)
    
    # Get data loaders with num_workers=0 to avoid multiprocessing
    train_dataset = data_loader.train_dataset
    val_dataset = data_loader.val_dataset  
    test_dataset = data_loader.test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Get numpy data for XGBoost
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_numpy_data()
    
    device = 'cpu'  # Force CPU to avoid CUDA issues
    print(f"Device: {device}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    results = []
    
    # Train MLP Models
    print("\n🧠 Training MLP Models (1 epoch each)")
    print("-" * 40)
    
    mlp_variants = ['small', 'medium', 'large']
    for variant in mlp_variants:
        print(f"\n🔄 Training MLP {variant}...")
        start = time.time()
        
        model = create_mlp_model(variant=variant)
        model = simple_train_pytorch(model, train_loader, val_loader, epochs=1, device=device)
        
        y_true, y_pred = evaluate_model(model, test_loader, device)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        train_time = time.time() - start
        
        print(f"✅ MLP {variant}: Accuracy={accuracy:.4f}, Time={train_time:.1f}s")
        
        results.append({
            'model': f'mlp_{variant}',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': train_time
        })
    
    # Train CNN Models
    print("\n🖼️ Training CNN Models (1 epoch each)")
    print("-" * 40)
    
    cnn_variants = ['simple', 'medium', 'deep', 'modern']
    for variant in cnn_variants:
        print(f"\n🔄 Training CNN {variant}...")
        start = time.time()
        
        model = create_cnn_model(variant=variant)
        model = simple_train_pytorch(model, train_loader, val_loader, epochs=1, device=device)
        
        y_true, y_pred = evaluate_model(model, test_loader, device)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        train_time = time.time() - start
        
        print(f"✅ CNN {variant}: Accuracy={accuracy:.4f}, Time={train_time:.1f}s")
        
        results.append({
            'model': f'cnn_{variant}',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': train_time
        })
    
    # Train XGBoost Models
    print("\n🌳 Training XGBoost Models")
    print("-" * 40)
    
    xgb_variants = ['fast', 'balanced', 'deep']
    for variant in xgb_variants:
        print(f"\n🔄 Training XGBoost {variant}...")
        start = time.time()
        
        model = create_xgboost_model(variant=variant)
        
        # Use subset for faster training
        subset_size = 5000
        model.fit(X_train[:subset_size], y_train[:subset_size])
        
        y_pred = model.predict(X_test[:1000])  # Test on subset too
        y_test_subset = y_test[:1000]
        
        accuracy = accuracy_score(y_test_subset, y_pred)
        precision = precision_score(y_test_subset, y_pred, average='macro')
        recall = recall_score(y_test_subset, y_pred, average='macro')
        f1 = f1_score(y_test_subset, y_pred, average='macro')
        
        train_time = time.time() - start
        
        print(f"✅ XGBoost {variant}: Accuracy={accuracy:.4f}, Time={train_time:.1f}s")
        
        results.append({
            'model': f'xgboost_{variant}',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': train_time
        })
    
    # Final Results
    print("\n\n📊 FINAL RESULTS")
    print("=" * 80)
    
    df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    print("\n🏆 Model Performance Ranking:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(s)':>10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i:<5} {row['model']:15} {row['accuracy']:10.4f} {row['precision']:10.4f} "
              f"{row['recall']:10.4f} {row['f1_score']:10.4f} {row['training_time']:10.1f}")
    
    print(f"\n✅ Training complete!")
    print(f"🥇 Best model: {df.iloc[0]['model']} with {df.iloc[0]['accuracy']:.4f} accuracy")
    print(f"⏱️ Total time: {df['training_time'].sum():.1f}s")
    
    # Save results
    results_file = f"quick_results_{datetime.now().strftime('%Y-%m-%d-%H:%M')}.csv"
    df.to_csv(results_file, index=False)
    print(f"📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()