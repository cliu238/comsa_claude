"""Demo script to compare XGBoost with and without medical priors.

This is a simplified demo to show the prior enhancement functionality.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from baseline.models import XGBoostModel, XGBoostPriorEnhanced
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_prior_config import XGBoostPriorConfig


def generate_va_data(n_samples=500, n_features=50, n_classes=10):
    """Generate synthetic VA-like data."""
    np.random.seed(42)
    
    # Generate binary symptom data
    X = np.random.binomial(1, 0.2, size=(n_samples, n_features))
    
    # Create structured labels
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Simple rules to create structure
        if X[i, :5].sum() >= 3:
            y[i] = 0  # Cause 0 associated with first 5 symptoms
        elif X[i, 5:10].sum() >= 3:
            y[i] = 1  # Cause 1 associated with symptoms 5-10
        elif X[i, 10:15].sum() >= 3:
            y[i] = 2
        else:
            y[i] = np.random.randint(0, n_classes)
    
    # Convert to DataFrame/Series
    feature_names = [f"symptom_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="cause")
    
    return X_df, y_series


def main():
    print("=== XGBoost Prior Enhancement Demo ===\n")
    
    # Generate data
    X, y = generate_va_data()
    print(f"Generated data: X shape={X.shape}, y shape={y.shape}")
    print(f"Number of classes: {len(np.unique(y))}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Vanilla XGBoost
    print("1. Training vanilla XGBoost...")
    vanilla_config = XGBoostConfig(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.3
    )
    vanilla_model = XGBoostModel(vanilla_config)
    vanilla_model.fit(X_train, y_train)
    vanilla_csmf = vanilla_model.calculate_csmf_accuracy(X_test, y_test)
    print(f"   CSMF accuracy: {vanilla_csmf:.3f}\n")
    
    # 2. XGBoost with prior features
    print("2. Training XGBoost with prior features...")
    feature_config = XGBoostPriorConfig(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.3,
        use_medical_priors=True,
        prior_method="feature_engineering",
        feature_prior_weight=1.0
    )
    feature_model = XGBoostPriorEnhanced(feature_config)
    feature_model.fit(X_train, y_train)
    
    # Test predictions
    pred_proba = feature_model.predict_proba(X_test)
    print(f"   Prediction shape: {pred_proba.shape}")
    print(f"   Model fitted: {feature_model._is_fitted}")
    
    # Calculate accuracy
    feature_csmf = feature_model.calculate_csmf_accuracy(X_test, y_test)
    print(f"   CSMF accuracy: {feature_csmf:.3f}")
    
    # Get prior influence
    influence = feature_model.get_prior_influence_report()
    print(f"   Prior influence: {influence.get('avg_contribution', 'N/A'):.3f}\n")
    
    # 3. Summary
    print("=== Summary ===")
    print(f"Vanilla XGBoost:    {vanilla_csmf:.3f}")
    print(f"With Prior Features: {feature_csmf:.3f}")
    print(f"Improvement:         {(feature_csmf - vanilla_csmf):.3f}")


if __name__ == "__main__":
    main()