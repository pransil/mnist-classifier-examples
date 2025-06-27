"""Enhanced training demonstration with improved models and realistic performance."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def create_enhanced_mnist_data():
    """Create enhanced MNIST-like data with more realistic patterns."""
    print("🎨 Creating enhanced MNIST-like dataset with realistic patterns...")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate dataset
    n_train = 2000  # 200 samples per digit
    n_test = 400    # 40 samples per digit
    
    def create_digit_pattern(digit, size=(28, 28), noise_level=0.1):
        """Create realistic digit patterns."""
        img = np.zeros(size, dtype=np.float32)
        h, w = size
        cx, cy = h // 2, w // 2
        
        if digit == 0:  # Circle/Oval
            for i in range(h):
                for j in range(w):
                    # Create oval shape
                    dx = (i - cx) / 8.0
                    dy = (j - cy) / 6.0
                    dist = dx*dx + dy*dy
                    if 0.7 < dist < 1.3:
                        img[i, j] = 0.8 + np.random.normal(0, 0.1)
        
        elif digit == 1:  # Vertical line with top serif
            # Main vertical line
            col = cx + np.random.randint(-2, 3)
            for i in range(6, 22):
                img[i, col] = 0.9 + np.random.normal(0, 0.1)
                if col > 0:
                    img[i, col-1] = 0.4 + np.random.normal(0, 0.1)
            
            # Top serif
            for j in range(max(0, col-3), min(w, col+2)):
                img[6, j] = 0.7 + np.random.normal(0, 0.1)
        
        elif digit == 2:  # Curved S shape
            # Top horizontal
            for j in range(6, 22):
                img[8, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Middle curve
            for i in range(9, 15):
                j = int(22 - (i-8) * 2.5)
                if 0 <= j < w:
                    img[i, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Bottom horizontal
            for j in range(6, 22):
                img[19, j] = 0.8 + np.random.normal(0, 0.1)
        
        elif digit == 3:  # Two curves
            # Top curve
            for i in range(8, 14):
                j = int(16 + 4 * np.sin((i-8) * np.pi / 6))
                if 0 <= j < w:
                    img[i, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Bottom curve
            for i in range(14, 20):
                j = int(16 + 4 * np.sin((i-14) * np.pi / 6))
                if 0 <= j < w:
                    img[i, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Middle connection
            for j in range(14, 18):
                img[14, j] = 0.6 + np.random.normal(0, 0.1)
        
        elif digit == 4:  # L with vertical line
            # Vertical line (right)
            for i in range(6, 22):
                img[i, 18] = 0.8 + np.random.normal(0, 0.1)
            
            # Horizontal line (middle)
            for j in range(8, 19):
                img[14, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Diagonal/vertical (left)
            for i in range(6, 15):
                img[i, 8] = 0.7 + np.random.normal(0, 0.1)
        
        elif digit == 5:  # S shape with horizontals
            # Top horizontal
            for j in range(8, 20):
                img[8, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Left vertical (top part)
            for i in range(8, 14):
                img[i, 8] = 0.8 + np.random.normal(0, 0.1)
            
            # Middle horizontal
            for j in range(8, 18):
                img[14, j] = 0.7 + np.random.normal(0, 0.1)
            
            # Right vertical (bottom part)
            for i in range(14, 20):
                img[i, 18] = 0.8 + np.random.normal(0, 0.1)
            
            # Bottom horizontal
            for j in range(8, 19):
                img[19, j] = 0.8 + np.random.normal(0, 0.1)
        
        elif digit == 6:  # Circle with gap at top
            for i in range(h):
                for j in range(w):
                    dx = (i - cx) / 7.0
                    dy = (j - cy) / 6.0
                    dist = dx*dx + dy*dy
                    if 0.7 < dist < 1.2 and i > cx - 2:  # Gap at top
                        img[i, j] = 0.8 + np.random.normal(0, 0.1)
        
        elif digit == 7:  # Top horizontal with diagonal
            # Top horizontal
            for j in range(8, 20):
                img[8, j] = 0.8 + np.random.normal(0, 0.1)
            
            # Diagonal down-left
            for i in range(9, 20):
                j = int(19 - (i-8) * 0.8)
                if 0 <= j < w:
                    img[i, j] = 0.8 + np.random.normal(0, 0.1)
        
        elif digit == 8:  # Two circles
            # Top circle
            for i in range(h//2):
                for j in range(w):
                    dx = (i - cx//2) / 4.0
                    dy = (j - cy) / 5.0
                    dist = dx*dx + dy*dy
                    if 0.6 < dist < 1.1:
                        img[i, j] = 0.7 + np.random.normal(0, 0.1)
            
            # Bottom circle
            for i in range(h//2, h):
                for j in range(w):
                    dx = (i - 3*cx//2) / 4.0
                    dy = (j - cy) / 5.0
                    dist = dx*dx + dy*dy
                    if 0.6 < dist < 1.1:
                        img[i, j] = 0.7 + np.random.normal(0, 0.1)
        
        elif digit == 9:  # Circle with gap at bottom
            for i in range(h):
                for j in range(w):
                    dx = (i - cx) / 7.0
                    dy = (j - cy) / 6.0
                    dist = dx*dx + dy*dy
                    if 0.7 < dist < 1.2 and i < cx + 2:  # Gap at bottom
                        img[i, j] = 0.8 + np.random.normal(0, 0.1)
        
        # Add realistic noise
        noise = np.random.normal(0, noise_level, size)
        img = np.clip(img + noise, 0, 1)
        
        return img
    
    # Generate training data
    X_train = []
    y_train = []
    
    for digit in range(10):
        n_samples = n_train // 10
        for _ in range(n_samples):
            img = create_digit_pattern(digit, noise_level=0.05)
            X_train.append(img.flatten())
            y_train.append(digit)
    
    # Generate test data (with slightly different patterns)
    X_test = []
    y_test = []
    
    for digit in range(10):
        n_samples = n_test // 10
        for _ in range(n_samples):
            img = create_digit_pattern(digit, noise_level=0.08)
            X_test.append(img.flatten())
            y_test.append(digit)
    
    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    # Shuffle the data
    train_indices = np.random.permutation(len(X_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    test_indices = np.random.permutation(len(X_test))
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    
    print(f"✅ Enhanced data created: {len(X_train)} train, {len(X_test)} test samples")
    print(f"   Data shape: {X_train.shape}, Labels: {np.unique(y_train)}")
    
    return (X_train, y_train), (X_test, y_test)


def train_improved_mlp(X_train, y_train, X_test, y_test):
    """Train an improved MLP with better architecture and training."""
    print("\n🧠 Training Improved MLP (2-Hidden Layer Network)")
    print("-" * 50)
    
    class ImprovedMLP:
        def __init__(self, input_size=784, hidden1=256, hidden2=128, output_size=10):
            # He initialization for ReLU
            scale1 = np.sqrt(2.0 / input_size)
            scale2 = np.sqrt(2.0 / hidden1)
            scale3 = np.sqrt(2.0 / hidden2)
            
            self.W1 = np.random.randn(input_size, hidden1) * scale1
            self.b1 = np.zeros(hidden1)
            self.W2 = np.random.randn(hidden1, hidden2) * scale2
            self.b2 = np.zeros(hidden2)
            self.W3 = np.random.randn(hidden2, output_size) * scale3
            self.b3 = np.zeros(output_size)
            
            # Add momentum
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)
            self.v_W3 = np.zeros_like(self.W3)
            self.v_b3 = np.zeros_like(self.b3)
        
        def relu(self, x):
            return np.maximum(0, x)
        
        def softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = self.relu(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = self.relu(self.z2)
            self.z3 = self.a2 @ self.W3 + self.b3
            self.a3 = self.softmax(self.z3)
            return self.a3
        
        def predict(self, X):
            probs = self.forward(X)
            return np.argmax(probs, axis=1)
        
        def accuracy(self, X, y):
            predictions = self.predict(X)
            return np.mean(predictions == y)
        
        def train_step(self, X, y, learning_rate=0.01, momentum=0.9):
            m = X.shape[0]
            
            # Forward pass
            probs = self.forward(X)
            
            # Cross-entropy loss
            loss = -np.sum(np.log(probs[range(m), y] + 1e-8)) / m
            
            # Backward pass
            dZ3 = probs.copy()
            dZ3[range(m), y] -= 1
            dZ3 /= m
            
            dW3 = self.a2.T @ dZ3
            db3 = np.sum(dZ3, axis=0)
            
            dA2 = dZ3 @ self.W3.T
            dZ2 = dA2 * (self.z2 > 0)  # ReLU derivative
            
            dW2 = self.a1.T @ dZ2
            db2 = np.sum(dZ2, axis=0)
            
            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (self.z1 > 0)
            
            dW1 = X.T @ dZ1
            db1 = np.sum(dZ1, axis=0)
            
            # Update with momentum
            self.v_W3 = momentum * self.v_W3 + learning_rate * dW3
            self.v_b3 = momentum * self.v_b3 + learning_rate * db3
            self.v_W2 = momentum * self.v_W2 + learning_rate * dW2
            self.v_b2 = momentum * self.v_b2 + learning_rate * db2
            self.v_W1 = momentum * self.v_W1 + learning_rate * dW1
            self.v_b1 = momentum * self.v_b1 + learning_rate * db1
            
            self.W3 -= self.v_W3
            self.b3 -= self.v_b3
            self.W2 -= self.v_W2
            self.b2 -= self.v_b2
            self.W1 -= self.v_W1
            self.b1 -= self.v_b1
            
            return loss
    
    start_time = time.time()
    
    # Initialize model
    model = ImprovedMLP()
    
    # Training parameters
    epochs = 20
    batch_size = 64
    learning_rate = 0.01
    
    print(f"🔧 Training for {epochs} epochs with batch size {batch_size}")
    print("    Architecture: 784 -> 256 -> 128 -> 10")
    
    best_test_acc = 0
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            loss = model.train_step(batch_X, batch_y, learning_rate)
            total_loss += loss
            n_batches += 1
        
        # Evaluate every few epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            train_acc = model.accuracy(X_train, y_train)
            test_acc = model.accuracy(X_test, y_test)
            avg_loss = total_loss / n_batches
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f} {'*' if test_acc == best_test_acc else ''}")
        
        # Learning rate decay
        if epoch == 10:
            learning_rate *= 0.5
        elif epoch == 15:
            learning_rate *= 0.5
    
    training_time = time.time() - start_time
    final_accuracy = model.accuracy(X_test, y_test)
    
    print(f"✅ Improved MLP Complete: {final_accuracy:.4f} accuracy ({training_time:.1f}s)")
    
    return {
        'model_name': 'improved_mlp',
        'accuracy': final_accuracy,
        'training_time': training_time,
        'best_accuracy': best_test_acc,
        'model': model
    }


def train_ensemble_classifier(X_train, y_train, X_test, y_test):
    """Train an ensemble of simple classifiers."""
    print("\n🤝 Training Ensemble Classifier (Random Forest-like)")
    print("-" * 50)
    
    class SimpleRandomForest:
        def __init__(self, n_trees=10, max_depth=8, feature_subset_ratio=0.7):
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.feature_subset_ratio = feature_subset_ratio
            self.trees = []
            self.feature_subsets = []
        
        def build_tree(self, X, y, max_depth, feature_indices):
            # Simple decision tree
            if max_depth == 0 or len(np.unique(y)) == 1 or len(y) < 5:
                return {'type': 'leaf', 'prediction': np.bincount(y).argmax()}
            
            best_gini = float('inf')
            best_feature = None
            best_threshold = None
            
            # Try random features
            n_try_features = max(1, int(len(feature_indices) * 0.3))
            try_features = np.random.choice(feature_indices, n_try_features, replace=False)
            
            for feature_idx in try_features:
                feature_values = X[:, feature_idx]
                # Try random thresholds
                thresholds = np.random.choice(feature_values, min(5, len(feature_values)), replace=False)
                
                for threshold in thresholds:
                    left_mask = feature_values <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                        continue
                    
                    # Calculate weighted Gini impurity
                    left_gini = self._gini_impurity(y[left_mask])
                    right_gini = self._gini_impurity(y[right_mask])
                    weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / len(y)
                    
                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_feature = feature_idx
                        best_threshold = threshold
            
            if best_feature is None:
                return {'type': 'leaf', 'prediction': np.bincount(y).argmax()}
            
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            
            return {
                'type': 'split',
                'feature': best_feature,
                'threshold': best_threshold,
                'left': self.build_tree(X[left_mask], y[left_mask], max_depth - 1, feature_indices),
                'right': self.build_tree(X[right_mask], y[right_mask], max_depth - 1, feature_indices)
            }
        
        def _gini_impurity(self, y):
            if len(y) == 0:
                return 0
            counts = np.bincount(y, minlength=10)  # Ensure length 10 for digits
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)
        
        def fit(self, X, y):
            n_features = X.shape[1]
            subset_size = int(n_features * self.feature_subset_ratio)
            
            for i in range(self.n_trees):
                # Bootstrap sample
                boot_indices = np.random.choice(len(X), len(X), replace=True)
                X_boot = X[boot_indices]
                y_boot = y[boot_indices]
                
                # Random feature subset
                feature_subset = np.random.choice(n_features, subset_size, replace=False)
                self.feature_subsets.append(feature_subset)
                
                # Build tree
                tree = self.build_tree(X_boot, y_boot, self.max_depth, feature_subset)
                self.trees.append(tree)
                
                if i % 3 == 0:
                    print(f"    Built tree {i+1}/{self.n_trees}")
        
        def predict_tree(self, x, tree):
            if tree['type'] == 'leaf':
                return tree['prediction']
            
            if x[tree['feature']] <= tree['threshold']:
                return self.predict_tree(x, tree['left'])
            else:
                return self.predict_tree(x, tree['right'])
        
        def predict(self, X):
            predictions = np.zeros((len(X), self.n_trees))
            
            for i, tree in enumerate(self.trees):
                for j, x in enumerate(X):
                    predictions[j, i] = self.predict_tree(x, tree)
            
            # Majority vote
            final_predictions = []
            for j in range(len(X)):
                votes = np.bincount(predictions[j].astype(int), minlength=10)
                final_predictions.append(np.argmax(votes))
            
            return np.array(final_predictions)
        
        def accuracy(self, X, y):
            predictions = self.predict(X)
            return np.mean(predictions == y)
    
    start_time = time.time()
    
    # Train ensemble
    model = SimpleRandomForest(n_trees=15, max_depth=10)
    print(f"🔧 Training Random Forest with 15 trees, max depth 10")
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    final_accuracy = model.accuracy(X_test, y_test)
    
    print(f"✅ Ensemble Complete: {final_accuracy:.4f} accuracy ({training_time:.1f}s)")
    
    return {
        'model_name': 'random_forest',
        'accuracy': final_accuracy,
        'training_time': training_time,
        'model': model
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train multinomial logistic regression."""
    print("\n📈 Training Multinomial Logistic Regression")
    print("-" * 50)
    
    class LogisticRegression:
        def __init__(self, n_classes=10, n_features=784):
            # Initialize weights
            self.W = np.random.randn(n_features, n_classes) * 0.01
            self.b = np.zeros(n_classes)
        
        def softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        def forward(self, X):
            return self.softmax(X @ self.W + self.b)
        
        def predict(self, X):
            probs = self.forward(X)
            return np.argmax(probs, axis=1)
        
        def accuracy(self, X, y):
            predictions = self.predict(X)
            return np.mean(predictions == y)
        
        def fit(self, X, y, epochs=50, learning_rate=0.01, batch_size=64):
            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                total_loss = 0
                n_batches = 0
                
                for i in range(0, len(X), batch_size):
                    batch_X = X_shuffled[i:i+batch_size]
                    batch_y = y_shuffled[i:i+batch_size]
                    m = batch_X.shape[0]
                    
                    # Forward pass
                    probs = self.forward(batch_X)
                    
                    # Cross-entropy loss
                    loss = -np.sum(np.log(probs[range(m), batch_y] + 1e-8)) / m
                    total_loss += loss
                    n_batches += 1
                    
                    # Backward pass
                    dZ = probs.copy()
                    dZ[range(m), batch_y] -= 1
                    dZ /= m
                    
                    dW = batch_X.T @ dZ
                    db = np.sum(dZ, axis=0)
                    
                    # Update weights
                    self.W -= learning_rate * dW
                    self.b -= learning_rate * db
                
                # Print progress
                if epoch % 10 == 0 or epoch == epochs - 1:
                    avg_loss = total_loss / n_batches
                    train_acc = self.accuracy(X, y)
                    test_acc = self.accuracy(X_test, y_test)
                    print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    start_time = time.time()
    
    # Train model
    model = LogisticRegression()
    print("🔧 Training multinomial logistic regression")
    model.fit(X_train, y_train, epochs=30)
    
    training_time = time.time() - start_time
    
    # Evaluate
    final_accuracy = model.accuracy(X_test, y_test)
    
    print(f"✅ Logistic Regression Complete: {final_accuracy:.4f} accuracy ({training_time:.1f}s)")
    
    return {
        'model_name': 'logistic_regression',
        'accuracy': final_accuracy,
        'training_time': training_time,
        'model': model
    }


def main():
    """Main enhanced training demonstration."""
    print("🚀 MNIST Classifier - Enhanced Training Demonstration")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start_time = time.time()
    
    try:
        # Create enhanced demo data
        (X_train, y_train), (X_test, y_test) = create_enhanced_mnist_data()
        
        # Train all models
        results = []
        
        print("\n🎯 Training multiple ML models on enhanced MNIST-like data...")
        
        # Train Improved MLP
        mlp_result = train_improved_mlp(X_train, y_train, X_test, y_test)
        results.append(mlp_result)
        
        # Train Ensemble Classifier
        ensemble_result = train_ensemble_classifier(X_train, y_train, X_test, y_test)
        results.append(ensemble_result)
        
        # Train Logistic Regression
        lr_result = train_logistic_regression(X_train, y_train, X_test, y_test)
        results.append(lr_result)
        
        total_time = time.time() - total_start_time
        
        # Generate comprehensive report
        print("\n📊 ENHANCED TRAINING RESULTS")
        print("=" * 60)
        
        # Create results DataFrame
        data = []
        for result in results:
            data.append({
                'Model': result['model_name'].replace('_', ' ').title(),
                'Accuracy': result['accuracy'],
                'Accuracy %': result['accuracy'] * 100,
                'Training Time (s)': result['training_time'],
                'Efficiency': result['accuracy'] / result['training_time']  # Accuracy per second
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy', ascending=False)
        
        print("\n🏆 Final Model Rankings:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Best performer analysis
        best_model = df.iloc[0]
        print(f"\n🥇 Champion Model: {best_model['Model']}")
        print(f"   🎯 Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy %']:.2f}%)")
        print(f"   ⏱️  Training Time: {best_model['Training Time (s)']:.1f}s")
        print(f"   ⚡ Efficiency: {best_model['Efficiency']:.4f} accuracy/second")
        
        # Speed champion
        fastest = df.loc[df['Training Time (s)'].idxmin()]
        print(f"\n🏃 Speed Champion: {fastest['Model']}")
        print(f"   ⏱️  Training Time: {fastest['Training Time (s)']:.1f}s")
        print(f"   🎯 Accuracy: {fastest['Accuracy']:.4f} ({fastest['Accuracy %']:.2f}%)")
        
        # Overall insights
        print(f"\n📈 Performance Insights:")
        accuracy_range = df['Accuracy'].max() - df['Accuracy'].min()
        time_range = df['Training Time (s)'].max() - df['Training Time (s)'].min()
        
        print(f"   📊 Accuracy Range: {accuracy_range:.3f} ({accuracy_range*100:.1f} percentage points)")
        print(f"   ⏰ Time Range: {time_range:.1f}s ({time_range/df['Training Time (s)'].min():.1f}x difference)")
        print(f"   🔄 Total Experiments: {len(results)} models")
        print(f"   ⏱️  Total Time: {total_time:.1f}s")
        print(f"   📈 Average Accuracy: {df['Accuracy'].mean():.4f} ({df['Accuracy %'].mean():.1f}%)")
        
        # Save results
        output_dir = Path("enhanced_training_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "best_model": {
                "name": best_model['Model'],
                "accuracy": float(best_model['Accuracy']),
                "training_time": float(best_model['Training Time (s)'])
            },
            "all_models": [
                {
                    "name": row['Model'],
                    "accuracy": float(row['Accuracy']),
                    "training_time": float(row['Training Time (s)']),
                    "efficiency": float(row['Efficiency'])
                }
                for _, row in df.iterrows()
            ]
        }
        
        import json
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}/")
        
        print(f"\n🎉 ENHANCED TRAINING COMPLETE!")
        print(f"🏆 Best accuracy achieved: {df['Accuracy'].max():.1%}")
        print(f"⚡ Fastest training: {df['Training Time (s)'].min():.1f}s")
        print(f"🎯 All models successfully trained and evaluated!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)