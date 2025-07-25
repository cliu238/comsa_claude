#!/usr/bin/env python
"""
Example usage of CategoricalNB baseline model for VA cause-of-death prediction

This script demonstrates how to use the CategoricalNB model with VA-like categorical data.
The model handles Y/N/./DK/missing values natively and provides interpretable results.

Expected runtime: ~5 seconds
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from baseline.models.categorical_nb_model import CategoricalNBModel
from baseline.models.categorical_nb_config import CategoricalNBConfig
from baseline.models.hyperparameter_tuning import quick_tune_categorical_nb


def main():
    """Demonstrate CategoricalNB model usage."""
    print("CategoricalNB Baseline Model Example")
    print("=" * 40)
    
    # Create sample VA-like categorical data
    print("1. Creating sample VA categorical data...")
    X = pd.DataFrame({
        'fever': ['Y', 'N', '.', 'Y', 'DK', 'N', 'Y', '.', 'N', 'Y'] * 5,
        'cough': ['N', 'Y', 'Y', '.', 'N', 'DK', 'Y', 'N', 'Y', '.'] * 5,
        'difficulty_breathing': ['.', 'DK', 'Y', 'N', 'Y', 'Y', 'N', '.', 'DK', 'Y'] * 5,
        'chest_pain': ['Y', 'N', 'N', 'Y', '.', 'N', 'Y', 'DK', 'N', 'Y'] * 5,
        'weight_loss': ['N', '.', 'Y', 'N', 'DK', 'Y', 'N', 'Y', '.', 'N'] * 5,
    })
    
    # Create labels representing different causes of death
    causes = ['Pneumonia', 'Tuberculosis', 'Heart_Disease', 'Cancer', 'Other']
    y = pd.Series(np.random.choice(causes, size=50, p=[0.3, 0.2, 0.2, 0.15, 0.15]))
    
    print(f"   - Data shape: {X.shape}")
    print(f"   - Classes: {sorted(y.unique())}")
    
    # 2. Basic model usage
    print("\n2. Training CategoricalNB model...")
    config = CategoricalNBConfig(alpha=1.0, fit_prior=True)
    model = CategoricalNBModel(config=config)
    model.fit(X, y)
    print("   ✓ Model trained successfully")
    
    # 3. Make predictions
    print("\n3. Making predictions...")
    predictions = model.predict(X[:10])
    probabilities = model.predict_proba(X[:10])
    print(f"   - Predictions for first 10 samples: {predictions}")
    print(f"   - Probability shape: {probabilities.shape}")
    print(f"   - Sample probabilities sum to 1: {np.allclose(probabilities.sum(axis=1), 1.0)}")
    
    # 4. Feature importance
    print("\n4. Analyzing feature importance...")
    importance = model.get_feature_importance()
    print("   Top 3 most important features:")
    for i, (_, row) in enumerate(importance.head(3).iterrows()):
        print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # 5. Cross-validation
    print("\n5. Running cross-validation...")
    cv_results = model.cross_validate(X, y, cv=3)
    print(f"   - CSMF Accuracy: {cv_results['csmf_accuracy_mean']:.3f} ± {cv_results['csmf_accuracy_std']:.3f}")
    print(f"   - COD Accuracy: {cv_results['cod_accuracy_mean']:.3f} ± {cv_results['cod_accuracy_std']:.3f}")
    
    # 6. Hyperparameter tuning (quick example)
    print("\n6. Quick hyperparameter tuning...")
    try:
        best_model = quick_tune_categorical_nb(X, y, n_trials=5)
        print(f"   - Best alpha: {best_model.config.alpha:.3f}")
        print(f"   - Best fit_prior: {best_model.config.fit_prior}")
        print("   ✓ Hyperparameter tuning completed")
    except Exception as e:
        print(f"   - Hyperparameter tuning skipped: {str(e)}")
    
    # 7. Demonstrate categorical encoding
    print("\n7. Testing categorical encoding robustness...")
    X_mixed = pd.DataFrame({
        'symptom1': ['Y', 'yes', 1, True, 'N', 'no', 0, False, '.', np.nan],
        'symptom2': ['DK', 'unknown', '', ' ', 'missing', 'Y', 'N', '.', 'y', 'n']
    })
    encoded = model._prepare_categorical_features(X_mixed)
    print("   Mixed categorical values encoded successfully:")
    print(f"   - Input shape: {X_mixed.shape}")
    print(f"   - Encoded shape: {encoded.shape}")
    print(f"   - Unique encoded values: {sorted(np.unique(encoded))}")
    
    print("\n" + "=" * 40)
    print("✓ CategoricalNB model demonstration completed!")
    print("\nKey advantages of CategoricalNB for VA data:")
    print("- Native handling of categorical features (no extensive preprocessing)")
    print("- Robust to missing data through categorical encoding")
    print("- Fast training and inference")
    print("- Interpretable probability estimates")
    print("- Good baseline performance expected: 70-85% CSMF accuracy")


if __name__ == "__main__":
    main()