#!/usr/bin/env python
"""Comprehensive ablation study for XGBoost improvements."""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from itertools import product
from sklearn.preprocessing import LabelEncoder

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.models.xgboost_advanced_model import XGBoostAdvancedModel
from baseline.models.xgboost_domain_adaptive import XGBoostDomainAdaptive
from model_comparison.metrics.comparison_metrics import calculate_metrics

# Load data
print("Loading data...")
df = pd.read_csv(sys.argv[1])
output_dir = Path(sys.argv[2])

# Basic data preprocessing
print(f"Loaded {len(df)} records from {len(df['site'].unique())} sites")

# Filter to specified sites
sites_to_use = sys.argv[3].split() if len(sys.argv) > 3 else ["Mexico", "AP", "Pemba"]
df = df[df['site'].isin(sites_to_use)]
print(f"Filtered to {len(df)} records from sites: {sites_to_use}")

# Define configurations for ablation study
ablations = {
    "baseline": {
        "description": "Baseline XGBoost configuration",
        "config_class": XGBoostConfig,
        "config_params": {},
        "model_class": XGBoostModel,
        "model_params": {},
    },
    "enhanced": {
        "description": "Enhanced regularization configuration",
        "config_class": XGBoostEnhancedConfig,
        "config_params": {},
        "model_class": XGBoostModel,
        "model_params": {},
    },
    "conservative": {
        "description": "Fixed conservative parameters (no tuning)",
        "config_class": XGBoostEnhancedConfig,
        "config_params": "conservative",  # Use class method
        "model_class": XGBoostModel,
        "model_params": {},
    },
    "optimized_subsampling": {
        "description": "Optimized subsampling configuration",
        "config_class": XGBoostEnhancedConfig,
        "config_params": "optimized_subsampling",  # Use class method
        "model_class": XGBoostModel,
        "model_params": {},
    },
    "domain_adaptive": {
        "description": "Domain adaptive XGBoost",
        "config_class": XGBoostEnhancedConfig,
        "config_params": {},
        "model_class": XGBoostDomainAdaptive,
        "model_params": {},
    },
}

# Run ablation study
sites = sites_to_use
results = []

print("\nRunning ablation study across configurations...")
print("=" * 60)

for config_name, config_spec in ablations.items():
    print(f"\nTesting: {config_name}")
    print(f"Description: {config_spec['description']}")
    
    config_results = []
    
    # Test each site pair
    for train_site in sites:
        for test_site in sites:
            # Prepare data
            train_df = df[df['site'] == train_site]
            test_df = df[df['site'] == test_site]
            
            # Separate features and target
            target_column = 'gs_code34'
            # Drop non-feature columns
            drop_cols = [target_column, 'site', 'module', 'gs_text34', 'va34', 
                        'gs_code46', 'gs_text46', 'va46', 'gs_code55', 'gs_text55', 'va55',
                        'gs_comorbid1', 'gs_comorbid2', 'gs_level']
            X_train = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])
            y_test = test_df[target_column]
            
            # Encode labels if needed
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            
            # Convert all features to numeric
            for col in X_train.columns:
                # Check if column has mixed types or is object type
                if X_train[col].dtype == 'object' or X_train[col].isnull().any():
                    le = LabelEncoder()
                    # Convert to string to handle mixed types
                    X_train[col] = X_train[col].fillna('missing').astype(str)
                    X_test[col] = X_test[col].fillna('missing').astype(str)
                    # Fit on train and transform both
                    X_train[col] = le.fit_transform(X_train[col])
                    # Handle unseen categories in test set
                    test_values = X_test[col].unique()
                    train_values = set(le.classes_)
                    unseen = set(test_values) - train_values
                    if unseen:
                        # Map unseen values to 'missing'
                        X_test[col] = X_test[col].apply(lambda x: x if x in train_values else 'missing')
                    X_test[col] = le.transform(X_test[col])
                else:
                    # For numeric columns, just fill missing values
                    X_train[col] = X_train[col].fillna(0)
                    X_test[col] = X_test[col].fillna(0)
            
            try:
                # Create configuration
                if isinstance(config_spec['config_params'], str):
                    # Use class method
                    config = getattr(config_spec['config_class'], config_spec['config_params'])()
                else:
                    config = config_spec['config_class'](**config_spec['config_params'])
                
                # Create and train model
                if config_spec['model_class'] == XGBoostDomainAdaptive:
                    # Special handling for domain adaptive model
                    model = XGBoostDomainAdaptive(
                        base_config=config,
                        **config_spec['model_params']
                    )
                    
                    # Prepare data by site
                    data_by_site = {train_site: (X_train, y_train)}
                    model.fit(data_by_site)
                    y_pred = model.predict(X_test, source_domain=train_site)
                else:
                    model = config_spec['model_class'](
                        config=config,
                        **config_spec['model_params']
                    )
                    
                    # Add monotonic constraints if needed
                    if config_spec['model_params'].get('use_monotonic_constraints'):
                        model.fit(X_train, y_train)
                        # Create simple medical constraints
                        constraints = {}
                        for col in X_train.columns:
                            if 'fever' in col or 'difficulty' in col:
                                constraints[col] = 1
                            elif 'access' in col or 'vaccine' in col:
                                constraints[col] = -1
                        
                        if constraints:
                            model.fit(X_train, y_train, monotonic_constraints=constraints)
                    else:
                        model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred)
                
                result = {
                    'config': config_name,
                    'train_site': train_site,
                    'test_site': test_site,
                    'is_in_domain': train_site == test_site,
                    'csmf_accuracy': metrics['csmf_accuracy'],
                    'cod_accuracy': metrics['cod_accuracy'],
                }
                
                config_results.append(result)
                results.append(result)
                
                print(f"  {train_site} → {test_site}: CSMF={metrics['csmf_accuracy']:.4f}")
                
            except Exception as e:
                print(f"  ERROR {train_site} → {test_site}: {str(e)}")
                continue
    
    # Calculate summary for this configuration
    if config_results:
        results_df = pd.DataFrame(config_results)
        in_domain = results_df[results_df['is_in_domain']]['csmf_accuracy'].mean()
        out_domain = results_df[~results_df['is_in_domain']]['csmf_accuracy'].mean()
        gap = in_domain - out_domain
        
        print(f"\n  Summary for {config_name}:")
        print(f"    In-domain CSMF: {in_domain:.4f}")
        print(f"    Out-domain CSMF: {out_domain:.4f}")
        print(f"    Generalization gap: {gap:.4f}")

# Save detailed results
results_df = pd.DataFrame(results)
results_df.to_csv(output_dir / "ablation_results_detailed.csv", index=False)

# Create summary analysis
print("\n" + "=" * 60)
print("ABLATION STUDY SUMMARY")
print("=" * 60)

# Summary by configuration
summary = results_df.groupby(['config', 'is_in_domain']).agg({
    'csmf_accuracy': ['mean', 'std', 'count'],
    'cod_accuracy': ['mean', 'std']
}).round(4)

print("\nPerformance by Configuration:")
print(summary)

# Calculate improvement over baseline
baseline_out = results_df[
    (results_df['config'] == 'baseline') & 
    (~results_df['is_in_domain'])
]['csmf_accuracy'].mean()

improvements = []
for config in ablations.keys():
    if config == 'baseline':
        continue
    
    config_out = results_df[
        (results_df['config'] == config) & 
        (~results_df['is_in_domain'])
    ]['csmf_accuracy'].mean()
    
    improvement = config_out - baseline_out
    improvements.append({
        'config': config,
        'description': ablations[config]['description'],
        'out_domain_csmf': config_out,
        'improvement': improvement,
        'percent_improvement': (improvement / baseline_out * 100) if baseline_out > 0 else 0
    })

improvements_df = pd.DataFrame(improvements).sort_values('improvement', ascending=False)
improvements_df.to_csv(output_dir / "ablation_improvements.csv", index=False)

print("\nImprovements over Baseline:")
print(improvements_df.to_string(index=False))

# Identify best individual technique
best_single = improvements_df.iloc[0]
print(f"\nBest Single Technique: {best_single['config']}")
print(f"  Description: {best_single['description']}")
print(f"  Improvement: {best_single['improvement']:.4f} ({best_single['percent_improvement']:.1f}%)")

# Save configuration metadata
with open(output_dir / "ablation_configurations.json", "w") as f:
    json.dump(ablations, f, indent=2)

print(f"\nResults saved to: {output_dir}")
