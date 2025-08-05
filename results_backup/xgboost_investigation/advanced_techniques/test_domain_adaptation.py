#!/usr/bin/env python
"""Test domain-adaptive XGBoost model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
from baseline.models.xgboost_domain_adaptive import XGBoostDomainAdaptive
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.data.data_loader import VADataProcessor
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

# Prepare data by site
sites = ["Mexico", "AP", "UP", "Pemba"]
data_by_site = {}

# Drop label columns - must match the pattern from ray_tasks.py
label_columns = [
    "cause", "site", "module",
    "gs_code34", "gs_text34", "va34",
    "gs_code46", "gs_text46", "va46",  
    "gs_code55", "gs_text55", "va55",
    "gs_comorbid1", "gs_comorbid2",
    "gs_level", "cod5", "newid"
]

for site in sites:
    site_df = df[df['site'] == site]
    if len(site_df) < 50:  # Skip sites with too little data
        continue
        
    # Prepare features and labels
    y = site_df["cause"]
    columns_to_drop = [col for col in label_columns if col in site_df.columns]
    X = site_df.drop(columns=columns_to_drop)
    
    # Preprocess features for XGBoost
    X = preprocess_features(X)
    
    data_by_site[site] = (X, y)

# Test different adaptation strategies
strategies = ["multi_task", "feature_align", "instance_weight"]
results = []

for strategy in strategies:
    print(f"\nTesting strategy: {strategy}")
    
    # Create and train model
    model = XGBoostDomainAdaptive(
        base_config=XGBoostEnhancedConfig(),
        adaptation_strategy=strategy,
        feature_alignment=(strategy != "instance_weight"),
        instance_weighting=(strategy == "instance_weight"),
    )
    
    model.fit(data_by_site)
    
    # Evaluate cross-domain performance
    eval_results = model.cross_domain_evaluate(data_by_site)
    eval_results['strategy'] = strategy
    results.append(eval_results)
    
    # Print summary
    in_domain = eval_results[eval_results['is_in_domain']]
    out_domain = eval_results[~eval_results['is_in_domain']]
    
    print(f"  In-domain CSMF: {in_domain['csmf_accuracy'].mean():.4f}")
    print(f"  Out-domain CSMF: {out_domain['csmf_accuracy'].mean():.4f}")
    print(f"  Generalization gap: {in_domain['csmf_accuracy'].mean() - out_domain['csmf_accuracy'].mean():.4f}")

# Save results
all_results = pd.concat(results, ignore_index=True)
all_results.to_csv(output_dir / "domain_adaptation_results.csv", index=False)

# Summary comparison
print("\n=== Strategy Comparison ===")
summary = all_results.groupby(['strategy', 'is_in_domain']).agg({
    'csmf_accuracy': ['mean', 'std'],
    'cod_accuracy': ['mean', 'std']
})
print(summary)
