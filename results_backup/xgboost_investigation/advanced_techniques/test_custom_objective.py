#!/usr/bin/env python
"""Test XGBoost with custom CSMF-optimized objective."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from baseline.models.xgboost_advanced_model import XGBoostAdvancedModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.data.data_loader import VADataProcessor
from model_comparison.metrics.comparison_metrics import calculate_metrics
from sklearn.preprocessing import LabelEncoder


def preprocess_features(X):
    """Preprocess features for XGBoost - convert categorical to numeric."""
    X_processed = X.copy()
    
    # Encode categorical features
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            # Fill missing values with a placeholder
            X_processed[col] = X_processed[col].fillna('missing')
            # Use label encoding
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
    
    # Fill any remaining missing values with 0
    X_processed = X_processed.fillna(0)
    
    return X_processed


# Load data
data_path = sys.argv[1]
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

# Load data using VADataProcessor
processor = VADataProcessor()
df = processor.load_data(data_path)

# Ensure we have the expected columns
if "va34" in df.columns and "cause" not in df.columns:
    df["cause"] = df["va34"].astype(str)  # Convert to string for consistency

# Test different objective functions
objectives = ["csmf_weighted", "focal", "standard"]
sites = ["Mexico", "AP", "UP", "Pemba", "Bohol", "Dar"]

results = []

for objective in objectives:
    print(f"\nTesting objective: {objective}")
    
    for train_site in sites[:3]:  # Use subset for faster testing
        for test_site in sites:
            if train_site == test_site:
                continue
                
            # Prepare data
            train_df = df[df['site'] == train_site]
            test_df = df[df['site'] == test_site]
            
            # Drop label columns - must match the pattern from ray_tasks.py
            label_columns = [
                "cause", "site", "module",
                "gs_code34", "gs_text34", "va34",
                "gs_code46", "gs_text46", "va46",  
                "gs_code55", "gs_text55", "va55",
                "gs_comorbid1", "gs_comorbid2",
                "gs_level", "cod5", "newid"
            ]
            
            # Prepare features and labels
            y_train = train_df["cause"]
            y_test = test_df["cause"]
            
            columns_to_drop = [col for col in label_columns if col in train_df.columns]
            X_train = train_df.drop(columns=columns_to_drop)
            X_test = test_df.drop(columns=columns_to_drop)
            
            # Preprocess features for XGBoost
            X_train = preprocess_features(X_train)
            X_test = preprocess_features(X_test)
            
            # Train model
            if objective == "standard":
                model = XGBoostAdvancedModel(
                    config=XGBoostEnhancedConfig(),
                    use_custom_objective=False,
                )
            else:
                model = XGBoostAdvancedModel(
                    config=XGBoostEnhancedConfig(),
                    use_custom_objective=True,
                    objective_type=objective,
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            
            results.append({
                'objective': objective,
                'train_site': train_site,
                'test_site': test_site,
                'csmf_accuracy': metrics['csmf_accuracy'],
                'cod_accuracy': metrics['cod_accuracy'],
            })
            
            print(f"  {train_site} → {test_site}: CSMF={metrics['csmf_accuracy']:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_dir / "custom_objective_results.csv", index=False)

# Summary by objective
print("\n=== Summary by Objective ===")
summary = results_df.groupby('objective').agg({
    'csmf_accuracy': ['mean', 'std'],
    'cod_accuracy': ['mean', 'std']
})
print(summary)
