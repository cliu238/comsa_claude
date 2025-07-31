#!/bin/bash
# Experiment 1: XGBoost Regularization Comparison
# Compare different regularization strategies to improve out-of-domain generalization

set -e

echo "==========================================="
echo "XGBoost Regularization Comparison Experiment"
echo "==========================================="

# Configuration
DATA_PATH="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
OUTPUT_BASE="results/xgboost_investigation/regularization_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Use focused subset of sites for faster experimentation
# Mexico (best source), AP/Pemba (key transfer pairs)
SITES="Mexico AP Pemba"

echo "Starting experiments at $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Sites: $SITES"

# Experiment 1a: Standard Enhanced Configuration
echo ""
echo ">>> Experiment 1a: Standard Enhanced Configuration"
echo "Using enhanced search space with moderate regularization"

# Create output directory
mkdir -p "$OUTPUT_BASE/enhanced_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 10 \
    --enable-tuning \
    --tuning-trials 25 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --tuning-cpus-per-trial 1.0 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8265 \
    --output-dir "$OUTPUT_BASE/enhanced_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/enhanced_${TIMESTAMP}.log"

# Check if experiment succeeded (use PIPESTATUS to get exit code of python command, not tee)
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Experiment 1a failed!"
    exit 1
fi

# Experiment 1b: Conservative Configuration
echo ""
echo ">>> Experiment 1b: Conservative Configuration"
echo "Using conservative search space with strong regularization"

# Create output directory
mkdir -p "$OUTPUT_BASE/conservative_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 10 \
    --enable-tuning \
    --tuning-trials 25 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --tuning-cpus-per-trial 1.0 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8266 \
    --output-dir "$OUTPUT_BASE/conservative_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/conservative_${TIMESTAMP}.log"

# Experiment 1c: No Tuning - Fixed Conservative Parameters
echo ""
echo ">>> Experiment 1c: Fixed Conservative Parameters (No Tuning)"
echo "Using XGBoostEnhancedConfig.conservative() without tuning"

# Create output directory
mkdir -p "$OUTPUT_BASE/fixed_conservative_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 10 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8267 \
    --output-dir "$OUTPUT_BASE/fixed_conservative_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/fixed_conservative_${TIMESTAMP}.log"

# Generate comparison report
echo ""
echo ">>> Generating Comparison Report"

poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Load results
results = []
configs = ['enhanced', 'conservative', 'fixed_conservative']
base_path = Path('$OUTPUT_BASE')

print(f'Looking for results in: {base_path}')

for config in configs:
    # Find the most recent directory for this config (exclude .log files)
    dirs = [d for d in base_path.glob(f'{config}_*') if d.is_dir()]
    if dirs:
        latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
        csv_path = latest_dir / 'va34_comparison_results.csv'
        if csv_path.exists():
            print(f'Found results for {config}: {csv_path}')
            df = pd.read_csv(csv_path)
            df['regularization_config'] = config
            results.append(df)
        else:
            print(f'WARNING: No results file found for {config} at {csv_path}')
    else:
        print(f'WARNING: No directory found for {config}')

if results:
    # Combine results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Calculate performance gaps
    print('\\n=== Regularization Strategy Comparison ===\\n')
    
    for config in configs:
        config_df = combined_df[combined_df['regularization_config'] == config]
        if len(config_df) > 0:
            xgb_df = config_df[config_df['model'].str.contains('xgboost')]
            if len(xgb_df) > 0:
                # In-domain vs out-domain
                in_domain = xgb_df[xgb_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
                out_domain = xgb_df[xgb_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
                gap = ((in_domain - out_domain) / in_domain * 100) if in_domain > 0 else 0
                
                print(f'{config.upper()} Configuration:')
                print(f'  In-domain CSMF: {in_domain:.4f}')
                print(f'  Out-domain CSMF: {out_domain:.4f}')
                print(f'  Performance gap: {gap:.1f}%')
                print(f'  Execution time: {xgb_df[\"execution_time_seconds\"].mean():.1f}s')
                print()
    
    # Save combined results
    combined_df.to_csv(base_path / 'regularization_comparison_combined.csv', index=False)
    print(f'\\nCombined results saved to: {base_path}/regularization_comparison_combined.csv')
else:
    print('\\nERROR: No results found to compare!')
    print('This may be due to experiments failing to save results.')
    print('Check the log files for errors.')
    sys.exit(1)
"

echo ""
echo "Experiment completed at $(date)"
echo "Results saved to: $OUTPUT_BASE"