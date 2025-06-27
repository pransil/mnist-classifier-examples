# MNIST Digit Classifier Examples

Complete implementation of multiple ML approaches for MNIST digit classification with comprehensive evaluation and comparison.

## Overview

This project demonstrates different machine learning approaches for handwritten digit recognition:

- **Multi-Layer Perceptron (MLP)**: Neural networks with varying architectures
- **Convolutional Neural Networks (CNN)**: Deep learning with convolutional layers  
- **XGBoost**: Gradient boosting ensemble method

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Train All Models
```bash
python train_all_models.py
```

### Train Individual Models
```bash
# MLP models
python train_mnist_models.py

# CNN models  
python train_real_mnist.py

# XGBoost models
python simple_train_demo.py

# Enhanced demonstration
python enhanced_training_demo.py
```

## Model Performance

Best results achieved:
- **CNN Deep**: 98.77% accuracy
- **MLP Large**: 97.84% accuracy  
- **XGBoost Deep**: 97.23% accuracy

See `final_report/` for detailed results and analysis.

## Project Structure

```
├── mnist_classifier/           # Core ML modules
│   ├── models/                # Model definitions
│   ├── training/              # Training utilities
│   ├── utils/                 # Data handling and reporting
│   └── config.py              # Configuration
├── final_report/              # Results and analysis
├── plots/                     # Training visualizations
├── train_*.py                 # Training scripts
└── requirements.txt           # Dependencies
```

## Features

- **Multiple Architectures**: Compare different ML approaches
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Visualization**: Training curves and performance plots
- **Experiment Tracking**: MLflow integration for reproducibility
- **Batch Processing**: Train multiple models systematically

## Results

The project includes complete experimental results showing:
- Model comparison across architectures
- Training curves and convergence analysis
- Confusion matrices and per-class performance
- Computational efficiency metrics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- XGBoost 1.7+
- scikit-learn 1.3+
- MLflow 2.5+
- See `requirements.txt` for complete list

## Usage Examples

### Basic Model Training
```python
from mnist_classifier.training.trainer import PyTorchTrainer
from mnist_classifier.models.mlp import MLPSmall

model = MLPSmall()
trainer = PyTorchTrainer(model, "mlp_small")
results = trainer.train(train_loader, val_loader, epochs=10)
```

### Batch Evaluation
```python
from train_all_models import main
results = main()  # Trains and evaluates all models
```

### Custom Configuration
```python
from mnist_classifier.config import update_config

update_config({
    'training': {'epochs': 20, 'batch_size': 128},
    'model': {'hidden_size': 512}
})
```

This implementation provides a complete framework for MNIST classification with multiple ML approaches, comprehensive evaluation, and production-ready code structure.