"""Train and evaluate all MNIST classifier models systematically."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from mnist_classifier.data.loader import MNISTDataLoader, create_data_loaders
from mnist_classifier.models.mlp import create_mlp_model
from mnist_classifier.models.cnn import create_cnn_model
from mnist_classifier.models.xgboost_model import create_xgboost_model
from mnist_classifier.training.trainer import PyTorchTrainer, XGBoostTrainer
from mnist_classifier.training.hyperparams import create_hyperparameter_manager
from mnist_classifier.utils.metrics import BenchmarkEvaluator
from mnist_classifier.utils.reporting import ExperimentReporter
from mnist_classifier.utils.mlflow_utils import MLflowTracker
from mnist_classifier.utils.notifications import data_fallback_warning

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available - skipping PyTorch models")
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - skipping XGBoost models")
    XGBOOST_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  MLflow not available - proceeding without experiment tracking")
    MLFLOW_AVAILABLE = False


def setup_data():
    """Set up MNIST data for training and evaluation."""
    print("📁 Setting up MNIST data...")
    
    data_dir = Path("../data/MNIST")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing data, create synthetic if not available
    try:
        loader = MNISTDataLoader(data_dir=str(data_dir))
        
        # Check if MNIST data exists
        train_images_file = data_dir / "train_images.npy"
        if not train_images_file.exists():
            if not data_fallback_warning("MNIST data files not found"):
                return None
            
            print("🔄 Creating synthetic dataset...")
            create_synthetic_mnist_data(data_dir)
        
        loader.load_data()
        print(f"✅ Data loaded: {len(loader.train_data[0])} train, {len(loader.test_data[0])} test samples")
        
        return loader
        
    except Exception as e:
        if not data_fallback_warning(f"Error loading MNIST data: {e}"):
            return None
            
        print("🔄 Creating synthetic dataset for demonstration...")
        create_synthetic_mnist_data(data_dir)
        
        loader = MNISTDataLoader(data_dir=str(data_dir))
        loader.load_data()
        return loader


def create_synthetic_mnist_data(data_dir: Path):
    """Create synthetic MNIST-like data for demonstration."""
    print("🎲 Generating synthetic MNIST-like data...")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic training data
    n_train = 6000  # Smaller dataset for faster training
    n_test = 1000
    
    # Create synthetic images with some structure
    train_images = []
    train_labels = []
    
    for digit in range(10):
        n_samples = n_train // 10
        for _ in range(n_samples):
            # Create digit-like patterns
            img = np.zeros((28, 28), dtype=np.uint8)
            
            if digit == 0:  # Circle-like
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if 6 < dist < 10:
                            img[i, j] = np.random.randint(150, 255)
            
            elif digit == 1:  # Vertical line
                col = np.random.randint(10, 18)
                for i in range(5, 23):
                    img[i, col:col+2] = np.random.randint(150, 255)
            
            elif digit == 2:  # S-curve
                for i in range(28):
                    j = int(14 + 6 * np.sin(i * np.pi / 14))
                    if 0 <= j < 28:
                        img[i, j] = np.random.randint(150, 255)
            
            else:  # Random patterns for other digits
                n_points = np.random.randint(50, 150)
                for _ in range(n_points):
                    i, j = np.random.randint(0, 28, 2)
                    img[i, j] = np.random.randint(100, 255)
            
            # Add noise
            noise = np.random.randint(0, 50, (28, 28))
            img = np.clip(img + noise, 0, 255)
            
            train_images.append(img)
            train_labels.append(digit)
    
    # Generate test data similarly
    test_images = []
    test_labels = []
    
    for digit in range(10):
        n_samples = n_test // 10
        for _ in range(n_samples):
            # Similar pattern generation as training
            img = np.zeros((28, 28), dtype=np.uint8)
            
            if digit == 0:
                center = (14, 14)
                for i in range(28):
                    for j in range(28):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                        if 6 < dist < 10:
                            img[i, j] = np.random.randint(150, 255)
            elif digit == 1:
                col = np.random.randint(10, 18)
                for i in range(5, 23):
                    img[i, col:col+2] = np.random.randint(150, 255)
            else:
                n_points = np.random.randint(30, 100)
                for _ in range(n_points):
                    i, j = np.random.randint(0, 28, 2)
                    img[i, j] = np.random.randint(100, 255)
            
            noise = np.random.randint(0, 30, (28, 28))
            img = np.clip(img + noise, 0, 255)
            
            test_images.append(img)
            test_labels.append(digit)
    
    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Save data
    np.save(data_dir / "train_images.npy", train_images)
    np.save(data_dir / "train_labels.npy", train_labels)
    np.save(data_dir / "test_images.npy", test_images)
    np.save(data_dir / "test_labels.npy", test_labels)
    
    print(f"✅ Synthetic data created: {len(train_images)} train, {len(test_images)} test samples")


def train_pytorch_models(data_loader, evaluator, models_dir):
    """Train all PyTorch models (MLP and CNN)."""
    if not TORCH_AVAILABLE:
        print("⏭️  Skipping PyTorch models - not available")
        return
    
    print("\n🧠 Training PyTorch Models")
    print("=" * 40)
    
    # Get data loaders for PyTorch
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_loader.data_dir,
        batch_size=32,
        validation_split=0.2
    )
    
    # Get test data for evaluation
    (X_test, y_test), _ = data_loader.get_numpy_arrays()
    y_test_torch = torch.from_numpy(y_test).long()
    
    # PyTorch model configurations
    pytorch_configs = [
        # MLP models
        {'type': 'mlp', 'variant': 'small', 'epochs': 3},
        {'type': 'mlp', 'variant': 'medium', 'epochs': 3},
        
        # CNN models  
        {'type': 'cnn', 'variant': 'simple', 'epochs': 3},
        {'type': 'cnn', 'variant': 'medium', 'epochs': 3},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    for config in pytorch_configs:
        model_name = f"{config['type']}_{config['variant']}"
        print(f"\n🔄 Training {model_name}...")
        
        start_time = time.time()
        
        try:
            # Create model
            if config['type'] == 'mlp':
                model = create_mlp_model(variant=config['variant'])
            else:  # cnn
                model = create_cnn_model(variant=config['variant'])
            
            model = model.to(device)
            
            # Create trainer
            trainer = PyTorchTrainer(
                model=model,
                device=device,
                save_dir=models_dir
            )
            
            # Setup training with simple hyperparameters
            trainer.setup_training(
                lr=0.001,
                weight_decay=1e-4,
                optimizer_type='adam',
                scheduler_type='cosine',
                epochs=config['epochs']
            )
            
            # Train model
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config['epochs'],
                model_name=model_name
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on test set
            model.eval()
            predictions = []
            inference_start = time.time()
            
            with torch.no_grad():
                for batch_images, _ in test_loader:
                    batch_images = batch_images.to(device)
                    outputs = model(batch_images)
                    batch_preds = torch.argmax(outputs, dim=1)
                    predictions.extend(batch_preds.cpu().numpy())
            
            inference_time = (time.time() - inference_start) / len(y_test)
            
            # Add results to evaluator
            evaluator.add_model_results(
                model_name=model_name,
                y_true=y_test,
                y_pred=np.array(predictions),
                training_time=training_time,
                inference_time=inference_time
            )
            
            # Save model
            save_path = trainer.save_model(
                model_name, 
                accuracy=evaluator.results[model_name]['accuracy']
            )
            
            accuracy = evaluator.results[model_name]['accuracy']
            print(f"✅ {model_name}: {accuracy:.4f} accuracy ({training_time:.1f}s training)")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue


def train_xgboost_models(data_loader, evaluator, models_dir):
    """Train all XGBoost models."""
    if not XGBOOST_AVAILABLE:
        print("⏭️  Skipping XGBoost models - not available")
        return
    
    print("\n🌲 Training XGBoost Models")
    print("=" * 40)
    
    # Get data for XGBoost (flattened)
    (X_train, y_train), (X_test, y_test) = data_loader.get_numpy_arrays()
    
    # Create validation split
    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # XGBoost configurations
    xgboost_configs = [
        {
            'variant': 'fast',
            'params': {
                'n_estimators': 20,  # Reduced for faster training
                'max_depth': 4,
                'learning_rate': 0.3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0
            }
        },
        {
            'variant': 'balanced',
            'params': {
                'n_estimators': 50,  # Reduced for faster training
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        }
    ]
    
    for config in xgboost_configs:
        model_name = f"xgboost_{config['variant']}"
        print(f"\n🔄 Training {model_name}...")
        
        start_time = time.time()
        
        try:
            # Create trainer
            trainer = XGBoostTrainer(save_dir=models_dir)
            
            # Train model
            history = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                params=config['params'],
                model_name=model_name
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on test set
            inference_start = time.time()
            predictions = trainer.predict(X_test)
            inference_time = (time.time() - inference_start) / len(X_test)
            
            # Add results to evaluator
            evaluator.add_model_results(
                model_name=model_name,
                y_true=y_test,
                y_pred=predictions,
                training_time=training_time,
                inference_time=inference_time
            )
            
            # Save model
            save_path = trainer.save_model(
                model_name,
                accuracy=evaluator.results[model_name]['accuracy']
            )
            
            accuracy = evaluator.results[model_name]['accuracy']
            print(f"✅ {model_name}: {accuracy:.4f} accuracy ({training_time:.1f}s training)")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue


def main():
    """Main training and evaluation pipeline."""
    print("🚀 MNIST Classifier - Complete Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    
    reports_dir = Path("training_reports")
    reports_dir.mkdir(exist_ok=True)
    
    total_start_time = time.time()
    
    try:
        # Setup data
        data_loader = setup_data()
        
        # Initialize evaluation
        evaluator = BenchmarkEvaluator(num_classes=10)
        
        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            print("📊 Initializing MLflow tracking...")
            mlflow.set_experiment("mnist_classifier_complete_training")
        
        # Train PyTorch models
        train_pytorch_models(data_loader, evaluator, models_dir)
        
        # Train XGBoost models
        train_xgboost_models(data_loader, evaluator, models_dir)
        
        total_training_time = time.time() - total_start_time
        
        # Generate comprehensive report
        print("\n📊 Generating Training Report")
        print("=" * 40)
        
        if len(evaluator.results) == 0:
            print("⚠️  No models were successfully trained!")
            return 1
        
        # Print summary
        print_training_summary(evaluator, total_training_time)
        
        # Generate detailed report
        reporter = ExperimentReporter(output_dir=str(reports_dir))
        
        experiment_config = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_training_time": total_training_time,
            "models_trained": len(evaluator.results),
            "dataset_size": f"{len(data_loader.train_data[0])} train, {len(data_loader.test_data[0])} test",
            "frameworks": ["PyTorch", "XGBoost"],
            "device": "cuda" if torch.cuda.is_available() else "cpu" if TORCH_AVAILABLE else "N/A",
            "training_type": "Accelerated training with reduced epochs for demonstration"
        }
        
        print("\n📄 Generating comprehensive report...")
        report_path = reporter.generate_full_report(
            benchmark_evaluator=evaluator,
            experiment_config=experiment_config,
            include_interactive=True
        )
        
        # Save model summary
        save_model_summary(evaluator, models_dir)
        
        print(f"\n🎉 Training Complete!")
        print(f"📄 Report: {report_path}")
        print(f"💾 Models saved in: {models_dir}")
        print(f"⏱️  Total time: {total_training_time:.1f}s")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def print_training_summary(evaluator, total_time):
    """Print a summary of training results."""
    print("\n🏆 TRAINING RESULTS SUMMARY")
    print("=" * 50)
    
    if len(evaluator.results) == 0:
        print("No models trained successfully.")
        return
    
    # Get comparison table
    comparison_df = evaluator.get_comparison_table()
    print("\n📊 Model Performance:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Best performers
    best_models = evaluator.get_best_models('accuracy', top_k=3)
    print(f"\n🥇 Top Performers:")
    for i, (model, accuracy) in enumerate(best_models, 1):
        train_time = evaluator.results[model]['training_time']
        print(f"  {i}. {model}: {accuracy:.4f} ({accuracy*100:.2f}%) - {train_time:.1f}s")
    
    # Training efficiency
    fastest_models = sorted(
        [(name, results['training_time']) for name, results in evaluator.results.items()],
        key=lambda x: x[1]
    )[:3]
    
    print(f"\n⚡ Fastest Training:")
    for i, (model, train_time) in enumerate(fastest_models, 1):
        accuracy = evaluator.results[model]['accuracy']
        print(f"  {i}. {model}: {train_time:.1f}s - {accuracy:.4f} accuracy")
    
    print(f"\n📈 Overall Statistics:")
    accuracies = [r['accuracy'] for r in evaluator.results.values()]
    times = [r['training_time'] for r in evaluator.results.values()]
    
    print(f"  • Models trained: {len(evaluator.results)}")
    print(f"  • Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")
    print(f"  • Average accuracy: {np.mean(accuracies):.4f}")
    print(f"  • Total training time: {total_time:.1f}s")
    print(f"  • Average time per model: {total_time/len(evaluator.results):.1f}s")


def save_model_summary(evaluator, models_dir):
    """Save a summary of trained models."""
    summary_data = {
        "training_timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    for model_name, results in evaluator.results.items():
        summary_data["models"][model_name] = {
            "accuracy": float(results["accuracy"]),
            "training_time": float(results["training_time"]),
            "inference_time": float(results["inference_time"]),
            "model_file": f"{model_name}.pth" if model_name.startswith(('mlp', 'cnn')) else f"{model_name}.pkl"
        }
    
    summary_file = models_dir / "training_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"💾 Model summary saved: {summary_file}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)