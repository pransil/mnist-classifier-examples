#!/usr/bin/env python
"""Demo training individual models one at a time."""

import sys
import time
import numpy as np
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

from mnist_classifier.data.loader import MNISTDataLoader
from mnist_classifier.models.mlp import create_mlp_model
from mnist_classifier.models.cnn import create_cnn_model
from mnist_classifier.models.xgboost_model import create_xgboost_model


def train_one_pytorch_model(model, train_loader, device='cpu', epochs=1):
    """Train one PyTorch model quickly."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 200 == 0:
                print(f'    Batch {i}, Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')
    
    return model


def evaluate_pytorch(model, test_loader, device='cpu'):
    """Evaluate PyTorch model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def main():
    print("🚀 MNIST Model Training Demo - Individual Models")
    print("=" * 50)
    
    # Load data
    print("\n📁 Loading MNIST...")
    loader = MNISTDataLoader(data_root="data", batch_size=128)
    
    # Use smaller batch size and no multiprocessing
    from torch.utils.data import DataLoader
    train_loader = DataLoader(loader.train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(loader.test_dataset, batch_size=128, shuffle=False, num_workers=0) 
    
    # Get numpy data for XGBoost
    X_train, y_train, _, _, X_test, y_test = loader.get_numpy_data()
    
    device = 'cpu'
    print(f"Using device: {device}")
    
    results = []
    
    # Test MLP Small
    print("\n" + "="*50)
    print("🧠 Training MLP Small (1 epoch)")
    print("="*50)
    start = time.time()
    model = create_mlp_model(variant='small')
    model = train_one_pytorch_model(model, train_loader, device, epochs=1)
    accuracy = evaluate_pytorch(model, test_loader, device)
    train_time = time.time() - start
    print(f"✅ MLP Small: {accuracy:.4f} accuracy in {train_time:.1f}s")
    results.append(('mlp_small', accuracy, train_time))
    
    # Test MLP Medium  
    print("\n" + "="*50)
    print("🧠 Training MLP Medium (1 epoch)")
    print("="*50)
    start = time.time()
    model = create_mlp_model(variant='medium')
    model = train_one_pytorch_model(model, train_loader, device, epochs=1)
    accuracy = evaluate_pytorch(model, test_loader, device)
    train_time = time.time() - start
    print(f"✅ MLP Medium: {accuracy:.4f} accuracy in {train_time:.1f}s")
    results.append(('mlp_medium', accuracy, train_time))
    
    # Test CNN Simple
    print("\n" + "="*50)
    print("🖼️ Training CNN Simple (1 epoch)")
    print("="*50)
    start = time.time()
    model = create_cnn_model(variant='simple')
    model = train_one_pytorch_model(model, train_loader, device, epochs=1)
    accuracy = evaluate_pytorch(model, test_loader, device)
    train_time = time.time() - start
    print(f"✅ CNN Simple: {accuracy:.4f} accuracy in {train_time:.1f}s")
    results.append(('cnn_simple', accuracy, train_time))
    
    # Test XGBoost Fast
    print("\n" + "="*50)
    print("🌳 Training XGBoost Fast")
    print("="*50)
    start = time.time()
    model = create_xgboost_model(variant='fast')
    
    # Train on subset for speed
    subset_size = 5000
    print(f"Training on {subset_size} samples...")
    model.fit(X_train[:subset_size], y_train[:subset_size])
    
    # Test on subset too
    y_pred = model.predict(X_test[:1000])
    accuracy = accuracy_score(y_test[:1000], y_pred)
    train_time = time.time() - start
    print(f"✅ XGBoost Fast: {accuracy:.4f} accuracy in {train_time:.1f}s")
    results.append(('xgboost_fast', accuracy, train_time))
    
    # Final Summary
    print("\n\n📊 TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':15} {'Accuracy':>10} {'Time (s)':>10}")
    print("-" * 40)
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    
    for model_name, accuracy, train_time in results:
        print(f"{model_name:15} {accuracy:10.4f} {train_time:10.1f}")
    
    print(f"\n🥇 Best Model: {results[0][0]} with {results[0][1]:.4f} accuracy")
    print(f"⏱️ Total Training Time: {sum(r[2] for r in results):.1f}s")
    
    print("\n✅ Demo complete! All model types successfully trained and tested.")


if __name__ == "__main__":
    main()