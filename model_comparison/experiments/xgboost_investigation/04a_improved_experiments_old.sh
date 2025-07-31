#!/bin/bash
# Improved XGBoost experiments leveraging all fixes and enhancements
# This script runs experiments with the corrected configurations

set -e

echo "=============================================="
echo "Improved XGBoost Experiments with Fixes"
echo "=============================================="

# Configuration
DATA_PATH="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
OUTPUT_BASE="results/xgboost_investigation/improved_experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Use all sites for comprehensive testing
SITES="AP Bohol Dar Kenya Mexico Pemba UP"

echo "Starting improved experiments at $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Sites: $SITES"

# Experiment 1: Fixed Conservative Configuration
echo ""
echo ">>> Experiment 1: Fixed Conservative Configuration (with proper instantiation)"
echo "Using XGBoostEnhancedConfig.conservative() with all parameters properly passed"

mkdir -p "$OUTPUT_BASE/fixed_conservative_proper_${TIMESTAMP}"

# Run with conservative config flag
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 100 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8270 \
    --output-dir "$OUTPUT_BASE/fixed_conservative_proper_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    --use-conservative-xgboost \
    2>&1 | tee "$OUTPUT_BASE/fixed_conservative_proper_${TIMESTAMP}.log"

# Experiment 2: Cross-Domain Tuned Configuration
echo ""
echo ">>> Experiment 2: Cross-Domain Tuned Configuration"
echo "Using CrossDomainTuner for hyperparameter optimization"

mkdir -p "$OUTPUT_BASE/cross_domain_tuned_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 100 \
    --enable-tuning \
    --tuning-trials 100 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --tuning-cpus-per-trial 1.0 \
    --use-cross-domain-cv \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8271 \
    --output-dir "$OUTPUT_BASE/cross_domain_tuned_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/cross_domain_tuned_${TIMESTAMP}.log"

# Experiment 3: Adaptive Configuration
echo ""
echo ">>> Experiment 3: Adaptive Configuration"
echo "Using site-adaptive configurations based on data characteristics"

mkdir -p "$OUTPUT_BASE/adaptive_config_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 100 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8272 \
    --output-dir "$OUTPUT_BASE/adaptive_config_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    --use-adaptive-xgboost \
    2>&1 | tee "$OUTPUT_BASE/adaptive_config_${TIMESTAMP}.log"

# Experiment 4: Ensemble Approach
echo ""
echo ">>> Experiment 4: Multi-Site Ensemble"
echo "Training separate models for each source site and ensembling"

mkdir -p "$OUTPUT_BASE/ensemble_${TIMESTAMP}"

poetry run python model_comparison/experiments/xgboost_investigation/run_ensemble_experiment.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --n-bootstrap 100 \
    --output-dir "$OUTPUT_BASE/ensemble_${TIMESTAMP}" \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/ensemble_${TIMESTAMP}.log"

# Generate comparison report
echo ""
echo ">>> Generating Improved Experiments Report"

poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

base_path = Path('$OUTPUT_BASE')
results = []
configs = ['fixed_conservative_proper', 'cross_domain_tuned', 'adaptive_config', 'ensemble']

print('\\n=== Improved XGBoost Experiments Results ===\\n')

# Load baseline for comparison
baseline_path = Path('results/xgboost_investigation/regularization_comparison/regularization_comparison_combined.csv')
if baseline_path.exists():
    baseline_df = pd.read_csv(baseline_path)
    baseline_xgb = baseline_df[baseline_df['model'].str.contains('xgboost') & (baseline_df['regularization_config'] == 'enhanced')]
    if len(baseline_xgb) > 0:
        baseline_in = baseline_xgb[baseline_xgb['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
        baseline_out = baseline_xgb[baseline_xgb['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
        baseline_gap = ((baseline_in - baseline_out) / baseline_in * 100) if baseline_in > 0 else 0
        print(f'Baseline (Enhanced Config): In-domain={baseline_in:.4f}, Out-domain={baseline_out:.4f}, Gap={baseline_gap:.1f}%')
        print()

# Load results from each experiment
for config in configs:
    dirs = list(base_path.glob(f'{config}_*'))
    if dirs:
        latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
        csv_path = latest_dir / 'va34_comparison_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['experiment_config'] = config
            results.append(df)
            
            # Calculate metrics
            xgb_df = df[df['model'].str.contains('xgboost')]
            if len(xgb_df) > 0:
                in_domain = xgb_df[xgb_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
                out_domain = xgb_df[xgb_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
                gap = ((in_domain - out_domain) / in_domain * 100) if in_domain > 0 else 0
                
                print(f'{config.upper()} Configuration:')
                print(f'  In-domain CSMF: {in_domain:.4f}')
                print(f'  Out-domain CSMF: {out_domain:.4f}')
                print(f'  Performance gap: {gap:.1f}%')
                if gap < 30:
                    print(f'  ✓ TARGET ACHIEVED! (Gap < 30%)')
                elif gap < 40:
                    print(f'  → Good improvement (Gap < 40%)')
                print()

if results:
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_csv(base_path / 'improved_experiments_combined.csv', index=False)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Performance gap comparison
    gap_data = []
    for config in configs:
        config_df = combined_df[combined_df['experiment_config'] == config]
        xgb_df = config_df[config_df['model'].str.contains('xgboost')]
        if len(xgb_df) > 0:
            in_domain = xgb_df[xgb_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
            out_domain = xgb_df[xgb_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
            gap = ((in_domain - out_domain) / in_domain * 100) if in_domain > 0 else 0
            gap_data.append({'Config': config, 'Gap': gap})
    
    if gap_data:
        gap_df = pd.DataFrame(gap_data)
        colors = ['green' if g < 30 else 'orange' if g < 40 else 'red' for g in gap_df['Gap']]
        ax1.bar(gap_df['Config'], gap_df['Gap'], color=colors, alpha=0.8)
        ax1.axhline(y=30, color='green', linestyle='--', label='Target: 30%')
        ax1.axhline(y=53.8, color='red', linestyle='--', label='Original: 53.8%')
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Performance Gap (%)')
        ax1.set_title('Generalization Gap by Configuration')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
    
    # Site-specific performance heatmap
    site_performance = []
    for config in configs:
        config_df = combined_df[combined_df['experiment_config'] == config]
        xgb_df = config_df[config_df['model'].str.contains('xgboost')]
        for _, row in xgb_df.iterrows():
            if row['experiment_type'] == 'out_domain':
                site_performance.append({
                    'Config': config,
                    'Train Site': row['train_site'],
                    'Test Site': row['test_site'],
                    'CSMF': row['csmf_accuracy']
                })
    
    if site_performance:
        site_df = pd.DataFrame(site_performance)
        # Average across configs for heatmap
        pivot_df = site_df.groupby(['Train Site', 'Test Site'])['CSMF'].mean().reset_index()
        pivot_matrix = pivot_df.pivot(index='Test Site', columns='Train Site', values='CSMF')
        
        sns.heatmap(pivot_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                    center=0.5, vmin=0.2, vmax=0.8, ax=ax2)
        ax2.set_title('Average Out-Domain CSMF Accuracy (Improved Configs)')
    
    plt.tight_layout()
    plt.savefig(base_path / 'improved_experiments_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('\\n✓ Report generated successfully!')
    print(f'  Combined results: {base_path}/improved_experiments_combined.csv')
    print(f'  Visualization: {base_path}/improved_experiments_comparison.png')
else:
    print('\\nNo results found to analyze.')
"

echo ""
echo "Improved experiments completed at $(date)"
echo "Results saved to: $OUTPUT_BASE"