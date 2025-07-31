#!/bin/bash
# Experiment 5: Advanced XGBoost Techniques
# Test custom objectives, domain adaptation, and monotonic constraints

set -e

echo "==========================================="
echo "Advanced XGBoost Techniques Experiment"
echo "==========================================="

# Configuration
DATA_PATH="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
OUTPUT_BASE="results/xgboost_investigation/advanced_techniques"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Use all sites for comprehensive evaluation
SITES="Mexico AP UP Pemba Bohol Dar"

echo "Starting experiments at $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Sites: $SITES"

# Experiment 5a: Custom CSMF-Optimized Objective
echo ""
echo ">>> Experiment 5a: Custom CSMF-Optimized Objective"
echo "Testing XGBoost with custom objective function optimized for CSMF accuracy"

# Create output directory
mkdir -p "$OUTPUT_BASE/custom_objective_${TIMESTAMP}"

# Create test script for custom objective
cat > "$OUTPUT_BASE/test_custom_objective.py" << 'EOF'
#!/usr/bin/env python
"""Test XGBoost with custom CSMF-optimized objective."""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from baseline.models.xgboost_advanced_model import XGBoostAdvancedModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from data.va_data_handler import VADataHandler
from model_comparison.utils.metrics import calculate_metrics

# Load data
data_path = sys.argv[1]
output_dir = Path(sys.argv[2])

va_handler = VADataHandler()
df = va_handler.load_va_data(data_path)

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
            
            X_train, y_train = va_handler.prepare_features(train_df)
            X_test, y_test = va_handler.prepare_features(test_df)
            
            # Remove site column
            if 'site' in X_train.columns:
                X_train = X_train.drop('site', axis=1)
                X_test = X_test.drop('site', axis=1)
            
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
EOF

poetry run python "$OUTPUT_BASE/test_custom_objective.py" \
    "$DATA_PATH" \
    "$OUTPUT_BASE/custom_objective_${TIMESTAMP}" \
    2>&1 | tee "$OUTPUT_BASE/custom_objective_${TIMESTAMP}.log"

# Experiment 5b: Domain Adaptation
echo ""
echo ">>> Experiment 5b: Domain Adaptation with Multi-Task Learning"
echo "Testing domain-adaptive XGBoost model"

# Create output directory
mkdir -p "$OUTPUT_BASE/domain_adaptation_${TIMESTAMP}"

# Create test script for domain adaptation
cat > "$OUTPUT_BASE/test_domain_adaptation.py" << 'EOF'
#!/usr/bin/env python
"""Test domain-adaptive XGBoost model."""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
from baseline.models.xgboost_domain_adaptive import XGBoostDomainAdaptive
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from data.va_data_handler import VADataHandler

# Load data
data_path = sys.argv[1]
output_dir = Path(sys.argv[2])

va_handler = VADataHandler()
df = va_handler.load_va_data(data_path)

# Prepare data by site
sites = ["Mexico", "AP", "UP", "Pemba"]
data_by_site = {}

for site in sites:
    site_df = df[df['site'] == site]
    X, y = va_handler.prepare_features(site_df)
    if 'site' in X.columns:
        X = X.drop('site', axis=1)
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
EOF

poetry run python "$OUTPUT_BASE/test_domain_adaptation.py" \
    "$DATA_PATH" \
    "$OUTPUT_BASE/domain_adaptation_${TIMESTAMP}" \
    2>&1 | tee "$OUTPUT_BASE/domain_adaptation_${TIMESTAMP}.log"

# Experiment 5c: Ensemble Methods
echo ""
echo ">>> Experiment 5c: Ensemble Methods for Robust Predictions"
echo "Testing ensemble of XGBoost models"

# Create output directory
mkdir -p "$OUTPUT_BASE/ensemble_${TIMESTAMP}"

poetry run python -c "
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from baseline.models.xgboost_advanced_model import XGBoostAdvancedModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from data.va_data_handler import VADataHandler
from model_comparison.utils.metrics import calculate_metrics

# Load data
va_handler = VADataHandler()
df = va_handler.load_va_data('$DATA_PATH')

# Test ensemble on challenging transfer
train_site = 'Mexico'
test_sites = ['Pemba', 'AP']

# Prepare training data
train_df = df[df['site'] == train_site]
X_train, y_train = va_handler.prepare_features(train_df)
if 'site' in X_train.columns:
    X_train = X_train.drop('site', axis=1)

# Create base model
base_model = XGBoostAdvancedModel(
    config=XGBoostEnhancedConfig(),
    use_custom_objective=True,
    objective_type='csmf_weighted',
)

print('Training ensemble models...')
ensemble_models = base_model.fit_ensemble(
    X_train, y_train,
    n_models=5,
    subsample_data=0.8,
    subsample_features=0.8,
)

# Evaluate on test sites
results = []

for test_site in test_sites:
    test_df = df[df['site'] == test_site]
    X_test, y_test = va_handler.prepare_features(test_df)
    if 'site' in X_test.columns:
        X_test = X_test.drop('site', axis=1)
    
    # Single model prediction
    base_model.fit(X_train, y_train)
    y_pred_single = base_model.predict(X_test)
    metrics_single = calculate_metrics(y_test, y_pred_single)
    
    # Ensemble prediction
    y_pred_ensemble = base_model.predict_ensemble(ensemble_models, X_test)
    metrics_ensemble = calculate_metrics(y_test, y_pred_ensemble)
    
    # With uncertainty
    y_pred_uncertain, uncertainties = base_model.predict_with_uncertainty(X_test, n_iterations=10)
    
    results.append({
        'test_site': test_site,
        'single_csmf': metrics_single['csmf_accuracy'],
        'ensemble_csmf': metrics_ensemble['csmf_accuracy'],
        'improvement': metrics_ensemble['csmf_accuracy'] - metrics_single['csmf_accuracy'],
        'avg_uncertainty': uncertainties.mean(),
    })
    
    print(f'\n{train_site} → {test_site}:')
    print(f'  Single model CSMF: {metrics_single[\"csmf_accuracy\"]:.4f}')
    print(f'  Ensemble CSMF: {metrics_ensemble[\"csmf_accuracy\"]:.4f}')
    print(f'  Improvement: {metrics_ensemble[\"csmf_accuracy\"] - metrics_single[\"csmf_accuracy\"]:+.4f}')
    print(f'  Avg uncertainty: {uncertainties.mean():.4f}')

# Save results
output_dir = Path('$OUTPUT_BASE/ensemble_${TIMESTAMP}')
pd.DataFrame(results).to_csv(output_dir / 'ensemble_results.csv', index=False)
"

# Generate final comparison report
echo ""
echo ">>> Generating Advanced Techniques Report"

poetry run python -c "
import pandas as pd
from pathlib import Path

base_path = Path('$OUTPUT_BASE')

# Find latest results for each experiment
experiments = {
    'custom_objective': 'Custom CSMF Objective',
    'domain_adaptation': 'Domain Adaptation',
    'ensemble': 'Ensemble Methods'
}

print('\n=== Advanced XGBoost Techniques Summary ===\n')

for exp_key, exp_name in experiments.items():
    dirs = [d for d in base_path.glob(f'{exp_key}_*') if d.is_dir()]
    if dirs:
        latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
        
        # Load appropriate results file
        if exp_key == 'custom_objective':
            results_file = latest_dir / 'custom_objective_results.csv'
            if results_file.exists():
                df = pd.read_csv(results_file)
                best_obj = df.groupby('objective')['csmf_accuracy'].mean().idxmax()
                best_score = df.groupby('objective')['csmf_accuracy'].mean().max()
                print(f'{exp_name}:')
                print(f'  Best objective: {best_obj}')
                print(f'  Out-domain CSMF: {best_score:.4f}')
                print()
                
        elif exp_key == 'domain_adaptation':
            results_file = latest_dir / 'domain_adaptation_results.csv'
            if results_file.exists():
                df = pd.read_csv(results_file)
                out_domain = df[~df['is_in_domain']]
                summary = out_domain.groupby('strategy')['csmf_accuracy'].mean()
                best_strategy = summary.idxmax()
                print(f'{exp_name}:')
                print(f'  Best strategy: {best_strategy}')
                print(f'  Out-domain CSMF: {summary[best_strategy]:.4f}')
                print()
                
        elif exp_key == 'ensemble':
            results_file = latest_dir / 'ensemble_results.csv'
            if results_file.exists():
                df = pd.read_csv(results_file)
                avg_improvement = df['improvement'].mean()
                print(f'{exp_name}:')
                print(f'  Average improvement: {avg_improvement:+.4f}')
                print(f'  Best single CSMF: {df[\"single_csmf\"].mean():.4f}')
                print(f'  Best ensemble CSMF: {df[\"ensemble_csmf\"].mean():.4f}')
                print()

print('\n>>> RECOMMENDATION: Combine custom CSMF objective with domain adaptation')
print('    and ensemble methods for maximum generalization improvement')
"

echo ""
echo "Experiment completed at $(date)"
echo "Results saved to: $OUTPUT_BASE"