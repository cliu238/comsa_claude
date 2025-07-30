#!/bin/bash
# Experiment 2: Cross-Domain Tuning Comparison
# Compare models tuned for in-domain vs out-of-domain performance

set -e

echo "==========================================="
echo "Cross-Domain Tuning Experiment"
echo "==========================================="

# Configuration
VENV_DIR="venv_linux"
DATA_PATH="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
OUTPUT_BASE="results/xgboost_investigation/cross_domain_tuning"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Use all sites for comprehensive cross-domain evaluation
SITES="Mexico AP UP Pemba Bohol Dar"

echo "Starting experiments at $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Sites: $SITES"

# Experiment 2a: In-Domain Tuning (Standard)
echo ""
echo ">>> Experiment 2a: In-Domain Tuning (Standard Approach)"
echo "Optimizing for in-domain CSMF accuracy"

source $VENV_DIR/bin/activate
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost \
    --n-bootstrap 30 \
    --enable-tuning \
    --tuning-trials 75 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --tuning-cpus-per-trial 1.0 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 30 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8265 \
    --output-dir "$OUTPUT_BASE/in_domain_tuning_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    --save-trained-models \
    2>&1 | tee "$OUTPUT_BASE/in_domain_tuning_${TIMESTAMP}.log"

# Experiment 2b: Cross-Domain Tuning
echo ""
echo ">>> Experiment 2b: Cross-Domain Tuning"
echo "Optimizing for out-of-domain performance using leave-one-site-out CV"

# Note: This requires implementing --use-cross-domain-cv flag
# For now, we'll simulate by using a balanced metric
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost \
    --n-bootstrap 30 \
    --enable-tuning \
    --tuning-trials 75 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --use-cross-domain-cv \
    --tuning-cpus-per-trial 1.0 \
    --use-conservative-space \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 30 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8266 \
    --output-dir "$OUTPUT_BASE/cross_domain_tuning_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    --save-trained-models \
    2>&1 | tee "$OUTPUT_BASE/cross_domain_tuning_${TIMESTAMP}.log"

# Experiment 2c: Multi-Objective Tuning
echo ""
echo ">>> Experiment 2c: Multi-Objective Tuning Simulation"
echo "Training separate models for each objective and comparing"

# Train a model optimized for Mexico->Others transfer
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites Mexico \
    --test-sites AP UP Pemba Bohol Dar \
    --models xgboost \
    --n-bootstrap 30 \
    --enable-tuning \
    --tuning-trials 50 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --use-conservative-space \
    --tuning-cpus-per-trial 1.0 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 30 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8267 \
    --output-dir "$OUTPUT_BASE/mexico_transfer_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    --save-trained-models \
    2>&1 | tee "$OUTPUT_BASE/mexico_transfer_${TIMESTAMP}.log"

# Generate comparison report
echo ""
echo ">>> Generating Cross-Domain Tuning Report"

poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Load results
base_path = Path('$OUTPUT_BASE')
results = {}

# Define experiment configurations
configs = {
    'in_domain_tuning': 'In-Domain Optimization',
    'cross_domain_tuning': 'Cross-Domain Optimization',
    'mexico_transfer': 'Mexico Transfer Optimization'
}

for config_key, config_name in configs.items():
    dirs = list(base_path.glob(f'{config_key}_*'))
    if dirs:
        latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
        csv_path = latest_dir / 'va34_comparison_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            results[config_name] = df

if results:
    print('\\n=== Cross-Domain Tuning Strategy Comparison ===\\n')
    
    comparison_data = []
    
    for config_name, df in results.items():
        xgb_df = df[df['model'] == 'xgboost']
        
        if len(xgb_df) > 0:
            # Calculate metrics
            in_domain_df = xgb_df[xgb_df['experiment_type'] == 'in_domain']
            out_domain_df = xgb_df[xgb_df['experiment_type'] == 'out_domain']
            
            metrics = {
                'Strategy': config_name,
                'In-Domain CSMF': in_domain_df['csmf_accuracy'].mean() if len(in_domain_df) > 0 else np.nan,
                'Out-Domain CSMF': out_domain_df['csmf_accuracy'].mean() if len(out_domain_df) > 0 else np.nan,
                'In-Domain COD': in_domain_df['cod_accuracy'].mean() if len(in_domain_df) > 0 else np.nan,
                'Out-Domain COD': out_domain_df['cod_accuracy'].mean() if len(out_domain_df) > 0 else np.nan,
                'Tuning Time (s)': xgb_df['tuning_time_seconds'].mean() if 'tuning_time_seconds' in xgb_df else 0,
                'Total Time (s)': xgb_df['execution_time_seconds'].mean()
            }
            
            # Calculate performance gap
            if metrics['In-Domain CSMF'] > 0:
                metrics['CSMF Gap (%)'] = ((metrics['In-Domain CSMF'] - metrics['Out-Domain CSMF']) / 
                                          metrics['In-Domain CSMF'] * 100)
            else:
                metrics['CSMF Gap (%)'] = np.nan
                
            comparison_data.append(metrics)
            
            # Print detailed results
            print(f'{config_name}:')
            print(f'  In-domain CSMF: {metrics[\"In-Domain CSMF\"]:.4f}')
            print(f'  Out-domain CSMF: {metrics[\"Out-Domain CSMF\"]:.4f}')
            print(f'  Performance gap: {metrics[\"CSMF Gap (%)\"]:.1f}%')
            print(f'  Tuning time: {metrics[\"Tuning Time (s)\"]:.1f}s')
            print()
            
            # Analyze specific site transfers
            if len(out_domain_df) > 0:
                print(f'  Top performing transfers:')
                top_transfers = out_domain_df.nlargest(5, 'csmf_accuracy')[
                    ['train_site', 'test_site', 'csmf_accuracy', 'cod_accuracy']
                ]
                for _, row in top_transfers.iterrows():
                    print(f'    {row[\"train_site\"]} → {row[\"test_site\"]}: '
                          f'CSMF={row[\"csmf_accuracy\"]:.4f}, COD={row[\"cod_accuracy\"]:.4f}')
                print()
    
    # Save comparison data
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(base_path / 'tuning_strategy_comparison.csv', index=False)
        print(f'\\nComparison saved to: {base_path}/tuning_strategy_comparison.csv')
        
        # Identify best strategy
        best_gap_idx = comparison_df['CSMF Gap (%)'].idxmin()
        best_strategy = comparison_df.loc[best_gap_idx, 'Strategy']
        best_gap = comparison_df.loc[best_gap_idx, 'CSMF Gap (%)']
        
        print(f'\\n>>> RECOMMENDATION: {best_strategy} shows the best generalization')
        print(f'    with only {best_gap:.1f}% performance gap between domains')
else:
    print('No results found to compare')
"

echo ""
echo "Experiment completed at $(date)"
echo "Results saved to: $OUTPUT_BASE"