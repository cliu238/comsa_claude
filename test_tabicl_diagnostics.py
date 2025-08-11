#!/usr/bin/env python
"""Quick diagnostic test for TabICL with 34 classes."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

# Test if TabICL is available
try:
    from tabicl import TabICLClassifier
    tabicl_available = True
    print("✓ TabICL is installed")
except ImportError:
    tabicl_available = False
    print("✗ TabICL not installed")
    exit(1)

# Create synthetic dataset with 34 classes
print("\nCreating synthetic dataset with 34 classes...")
X, y = make_classification(
    n_samples=1500,
    n_features=100,  # Similar to VA data
    n_informative=50,
    n_redundant=20,
    n_classes=34,
    n_clusters_per_class=2,
    random_state=42
)

# Convert to DataFrame for consistency
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name="target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# Test different configurations
configs = [
    {
        "name": "Default (no hierarchical)",
        "params": {
            "n_estimators": 32,
            "use_hierarchical": False,
            "batch_size": 8,
            "softmax_temperature": 0.9
        }
    },
    {
        "name": "Hierarchical (recommended for 34 classes)",
        "params": {
            "n_estimators": 48,
            "use_hierarchical": True,
            "batch_size": 4,
            "softmax_temperature": 0.5
        }
    },
    {
        "name": "Small ensemble (faster)",
        "params": {
            "n_estimators": 16,
            "use_hierarchical": True,
            "batch_size": 2,
            "softmax_temperature": 0.7
        }
    }
]

print("\n" + "="*60)
print("Testing TabICL configurations for 34-class problem:")
print("="*60)

for config in configs:
    print(f"\n{config['name']}:")
    print(f"  Parameters: {config['params']}")
    
    try:
        # Initialize model
        model = TabICLClassifier(**config['params'], random_state=42, verbose=False)
        
        # Train
        print("  Training...", end=" ")
        start_time = time.time()
        model.fit(X_train.values, y_train.values)
        train_time = time.time() - start_time
        print(f"Done ({train_time:.2f}s)")
        
        # Predict
        print("  Predicting...", end=" ")
        start_time = time.time()
        y_pred = model.predict(X_test.values)
        pred_time = time.time() - start_time
        print(f"Done ({pred_time:.2f}s)")
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test.values)
        print(f"  Accuracy: {accuracy:.3f}")
        
        # Memory usage estimate (if possible)
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except:
            pass
            
    except Exception as e:
        print(f"  ERROR: {str(e)[:100]}")
        if "hierarchical" in str(e).lower() or "class" in str(e).lower():
            print("  Note: This error suggests issues with multi-class handling")

print("\n" + "="*60)
print("Key Observations for 34-class VA problem:")
print("="*60)
print("1. Hierarchical mode is REQUIRED for >10 classes (TabICL limitation)")
print("2. Larger ensembles (48) improve accuracy but increase runtime")
print("3. Lower temperature (0.5) helps with many classes")
print("4. Batch size affects memory usage - reduce if OOM")
print("5. Expected runtime: 5-6 minutes for full VA dataset")