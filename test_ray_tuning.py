#!/usr/bin/env python
"""Quick test of Ray-based hyperparameter tuning."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model_comparison.hyperparameter_tuning.ray_tuner import quick_tune_model


def test_ray_tuning():
    """Test Ray-based hyperparameter tuning."""
    print("Testing Ray-based hyperparameter tuning...")
    
    # Generate test data
    X_array, y_array = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        random_state=42,
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i:02d}" for i in range(X_array.shape[1])]
    X = pd.DataFrame(X_array, columns=feature_names)
    
    # Create cause labels
    cause_names = ["Cardio", "Respiratory", "Infectious", "Cancer", "Other"]
    y = pd.Series([cause_names[i] for i in y_array], name="cause")
    
    print(f"Test data: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes")
    
    try:
        # Quick tune with minimal trials
        results = quick_tune_model(
            model_name="xgboost",
            X=X,
            y=y,
            n_trials=3,
            metric="csmf_accuracy",
        )
        
        print("✓ Ray tuning successful!")
        print(f"  Best CSMF accuracy: {results['best_score']:.4f}")
        print(f"  Best params: {results['best_params']}")
        print(f"  Model trained: {results['trained_model']._is_fitted}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ray tuning failed: {e}")
        return False


if __name__ == "__main__":
    success = test_ray_tuning()
    sys.exit(0 if success else 1)