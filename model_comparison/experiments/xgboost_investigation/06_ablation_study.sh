#!/bin/bash
# Experiment 6: Comprehensive Ablation Study
# Test each improvement technique individually to understand contribution

set -e

echo "==========================================="
echo "XGBoost Ablation Study Experiment"
echo "==========================================="

# Configuration
DATA_PATH="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
OUTPUT_BASE="results/xgboost_investigation/ablation_study"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Use representative sites for ablation
SITES="Mexico AP Pemba"  # Good, medium, and challenging sites

echo "Starting ablation study at $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Sites: $SITES"

# Create comprehensive ablation study script
cat > "$OUTPUT_BASE/run_ablation_study.py" << 'EOF'
#!/usr/bin/env python
"""Comprehensive ablation study for XGBoost improvements."""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import json
from itertools import product

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.models.xgboost_advanced_model import XGBoostAdvancedModel
from baseline.models.xgboost_domain_adaptive import XGBoostDomainAdaptive
from data.va_data_handler import VADataHandler
from model_comparison.utils.metrics import calculate_metrics

# Load data
print("Loading data...")
va_handler = VADataHandler()
df = va_handler.load_va_data(sys.argv[1])
output_dir = Path(sys.argv[2])

# Define only top 4 configurations based on prior results
ablations = {
    "enhanced": {
        "description": "Enhanced regularization configuration",
        "config_class": XGBoostEnhancedConfig,
        "config_params": {},
        "model_class": XGBoostModel,
        "model_params": {},
    },
    "fixed_conservative": {
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
    "optimized_tuning": {
        "description": "Optimized with focused tuning",
        "config_class": XGBoostEnhancedConfig,
        "config_params": "optimized_tuning",  # Use class method
        "model_class": XGBoostModel,
        "model_params": {},
    },
}

# Run ablation study
sites = ["Mexico", "AP", "Pemba"]
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
            
            X_train, y_train = va_handler.prepare_features(train_df)
            X_test, y_test = va_handler.prepare_features(test_df)
            
            # Remove site column
            if 'site' in X_train.columns:
                X_train = X_train.drop('site', axis=1)
                X_test = X_test.drop('site', axis=1)
            
            try:
                # Create configuration
                if isinstance(config_spec['config_params'], str):
                    # Use class method
                    config = getattr(config_spec['config_class'], config_spec['config_params'])()
                else:
                    config = config_spec['config_class'](**config_spec['config_params'])
                
                # Create and train model
                if config_spec['model_class'] == "XGBoostDomainAdaptive":
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
EOF

# Make script executable
chmod +x "$OUTPUT_BASE/run_ablation_study.py"

# Run ablation study
poetry run python "$OUTPUT_BASE/run_ablation_study.py" \
    "$DATA_PATH" \
    "$OUTPUT_BASE" \
    2>&1 | tee "$OUTPUT_BASE/ablation_study_${TIMESTAMP}.log"

# Generate visualization of results
echo ""
echo ">>> Generating Ablation Study Visualization"

poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_BASE')

# Load results
results_df = pd.read_csv(output_dir / 'ablation_results_detailed.csv')
improvements_df = pd.read_csv(output_dir / 'ablation_improvements.csv')

print('\n' + '='*70)
print('ABLATION STUDY VISUALIZATION')
print('='*70)

# Create ASCII bar chart of improvements
print('\nImprovement over Baseline (Out-Domain CSMF):')
print('-' * 60)

max_improvement = improvements_df['improvement'].max()
min_improvement = improvements_df['improvement'].min()

for _, row in improvements_df.iterrows():
    config = row['config'][:20].ljust(20)
    improvement = row['improvement']
    
    # Create bar
    if improvement >= 0:
        bar_length = int(improvement / max_improvement * 30) if max_improvement > 0 else 0
        bar = '█' * bar_length
        print(f'{config} |{bar} +{improvement:.4f}')
    else:
        bar_length = int(abs(improvement) / abs(min_improvement) * 30) if min_improvement < 0 else 0
        bar = '░' * bar_length
        print(f'{config} |{bar} {improvement:.4f}')

# Show synergy effects
print('\n\nSynergy Analysis:')
print('-' * 60)

# Compare combined vs individual
baseline = results_df[results_df['config'] == 'baseline']['csmf_accuracy'].mean()
enhanced_reg = improvements_df[improvements_df['config'] == 'enhanced_regularization']['improvement'].values[0]
custom_obj = improvements_df[improvements_df['config'] == 'custom_objective']['improvement'].values[0]
combined_basic = improvements_df[improvements_df['config'] == 'combined_basic']['improvement'].values[0]

expected_combined = enhanced_reg + custom_obj
actual_combined = combined_basic
synergy = actual_combined - expected_combined

print(f'Enhanced Regularization alone: +{enhanced_reg:.4f}')
print(f'Custom Objective alone: +{custom_obj:.4f}')
print(f'Expected combined improvement: +{expected_combined:.4f}')
print(f'Actual combined improvement: +{actual_combined:.4f}')
print(f'Synergy effect: {synergy:+.4f}')

# Site-specific analysis
print('\n\nSite-Specific Performance:')
print('-' * 60)

for site in ['Mexico', 'AP', 'Pemba']:
    site_out = results_df[
        (~results_df['is_in_domain']) & 
        (results_df['test_site'] == site)
    ]
    
    baseline_site = site_out[site_out['config'] == 'baseline']['csmf_accuracy'].mean()
    best_site = site_out.groupby('config')['csmf_accuracy'].mean().max()
    best_config = site_out.groupby('config')['csmf_accuracy'].mean().idxmax()
    
    print(f'\n{site} (as test site):')
    print(f'  Baseline CSMF: {baseline_site:.4f}')
    print(f'  Best CSMF: {best_site:.4f} ({best_config})')
    print(f'  Improvement: {best_site - baseline_site:.4f}')
"

echo ""
echo "Ablation study completed at $(date)"
echo "Results saved to: $OUTPUT_BASE"