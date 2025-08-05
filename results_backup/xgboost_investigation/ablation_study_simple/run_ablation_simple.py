#!/usr/bin/env python
"""Simplified ablation study for XGBoost improvements."""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from model_comparison.metrics.comparison_metrics import calculate_metrics

# Load data
print("Loading data...")
df = pd.read_csv(sys.argv[1])
output_dir = Path(sys.argv[2])

# Basic preprocessing
# Remove non-feature columns
drop_cols = ['gs_code34', 'site', 'module', 'gs_text34', 'va34', 
            'gs_code46', 'gs_text46', 'va46', 'gs_code55', 'gs_text55', 'va55',
            'gs_comorbid1', 'gs_comorbid2', 'gs_level']

# Prepare label encoder on all data
le = LabelEncoder()
y_all = le.fit_transform(df['gs_code34'])

# Simple configurations to test
configs = {
    "baseline": XGBoostConfig(),
    "enhanced": XGBoostEnhancedConfig(),
    "conservative": XGBoostEnhancedConfig.conservative(),
    "optimized_subsampling": XGBoostEnhancedConfig.optimized_subsampling(),
}

results = []

for config_name, config in configs.items():
    print(f"\nTesting: {config_name}")
    
    # Test on Mexico data (good site)
    mexico_df = df[df['site'] == 'Mexico'].copy()
    X_mexico = mexico_df.drop(columns=drop_cols, errors='ignore')
    y_mexico = pd.Series(le.transform(mexico_df['gs_code34']), index=mexico_df.index)
    
    # Convert categorical to numeric
    for col in X_mexico.columns:
        if X_mexico[col].dtype == 'object':
            X_mexico[col] = pd.Categorical(X_mexico[col]).codes
    
    # Create and evaluate model
    model = XGBoostModel(config=config)
    
    # Simple cross-validation
    try:
        scores = cross_val_score(model, X_mexico, y_mexico, cv=3, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"  CV Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        
        results.append({
            'config': config_name,
            'cv_mean': mean_score,
            'cv_std': std_score,
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
        })
    except Exception as e:
        print(f"  ERROR: {str(e)}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(output_dir / "ablation_results_simple.csv", index=False)

# Print summary
print("\n" + "="*60)
print("ABLATION STUDY SUMMARY")
print("="*60)

# Sort by performance
results_df = results_df.sort_values('cv_mean', ascending=False)

print("\nConfiguration Performance:")
print("-"*40)
for _, row in results_df.iterrows():
    print(f"{row['config']:20s}: {row['cv_mean']:.4f} (+/- {row['cv_std']:.4f})")

# Calculate improvements
baseline_score = results_df[results_df['config'] == 'baseline']['cv_mean'].values[0]
print("\nImprovements over baseline:")
print("-"*40)
for _, row in results_df.iterrows():
    if row['config'] != 'baseline':
        improvement = row['cv_mean'] - baseline_score
        pct_improvement = (improvement / baseline_score) * 100
        print(f"{row['config']:20s}: {improvement:+.4f} ({pct_improvement:+.1f}%)")

print(f"\nResults saved to: {output_dir}")
