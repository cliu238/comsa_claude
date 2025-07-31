#!/bin/bash
# Experiment 4: Optimized Subsampling Configuration
# Test the new optimized subsampling configuration that should improve both in-domain and out-domain performance

set -e

echo "==========================================="
echo "XGBoost Optimized Subsampling Experiment"
echo "==========================================="

# Configuration
DATA_PATH="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
OUTPUT_BASE="results/xgboost_investigation/optimized_subsampling"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Use focused subset of sites for faster experimentation
SITES="Mexico AP Pemba"

echo "Starting experiments at $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Sites: $SITES"

# Experiment 4a: Baseline (Current Enhanced Configuration)
echo ""
echo ">>> Experiment 4a: Baseline Enhanced Configuration"
echo "Using current enhanced configuration for baseline comparison"

# Create output directory
mkdir -p "$OUTPUT_BASE/baseline_enhanced_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 30 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8265 \
    --output-dir "$OUTPUT_BASE/baseline_enhanced_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/baseline_enhanced_${TIMESTAMP}.log"

# Check if experiment succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Experiment 4a failed!"
    exit 1
fi

# Experiment 4b: Optimized Subsampling Configuration
echo ""
echo ">>> Experiment 4b: Optimized Subsampling Configuration"
echo "Using new optimized subsampling parameters"

# Create output directory
mkdir -p "$OUTPUT_BASE/optimized_subsampling_${TIMESTAMP}"

# Create a wrapper script that properly configures XGBoost
cat > "$OUTPUT_BASE/run_with_optimized_config.py" << 'EOF'
#!/usr/bin/env python
"""Run distributed comparison with optimized subsampling configuration."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

# Import and configure the model factory
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.models.model_factory import ModelFactory

# Override the default XGBoost configuration
original_create_model = ModelFactory.create_model

def create_model_with_optimized_config(model_type, **kwargs):
    if model_type == "xgboost":
        # Use optimized subsampling configuration
        kwargs['config'] = XGBoostEnhancedConfig.optimized_subsampling()
    return original_create_model(model_type, **kwargs)

ModelFactory.create_model = staticmethod(create_model_with_optimized_config)

# Now run the main script
from model_comparison.scripts.run_distributed_comparison import main

if __name__ == "__main__":
    main()
EOF

poetry run python "$OUTPUT_BASE/run_with_optimized_config.py" \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 30 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8266 \
    --output-dir "$OUTPUT_BASE/optimized_subsampling_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/optimized_subsampling_${TIMESTAMP}.log"

# Check if experiment succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Experiment 4b failed!"
    exit 1
fi

# Experiment 4c: Optimized Subsampling with Tuning
echo ""
echo ">>> Experiment 4c: Optimized Subsampling with Tuning"
echo "Using optimized search space with narrower ranges"

# Create output directory
mkdir -p "$OUTPUT_BASE/optimized_tuning_${TIMESTAMP}"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites $SITES \
    --models xgboost insilico \
    --n-bootstrap 30 \
    --enable-tuning \
    --tuning-trials 20 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --tuning-cpus-per-trial 1.0 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 25 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8267 \
    --output-dir "$OUTPUT_BASE/optimized_tuning_${TIMESTAMP}" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee "$OUTPUT_BASE/optimized_tuning_${TIMESTAMP}.log"

# Generate comparison report
echo ""
echo ">>> Generating Optimized Subsampling Report"

poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Load results
results = []
configs = ['baseline_enhanced', 'optimized_subsampling', 'optimized_tuning']
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
            df['subsampling_config'] = config
            results.append(df)
        else:
            print(f'WARNING: No results file found for {config} at {csv_path}')
    else:
        print(f'WARNING: No directory found for {config}')

if results:
    # Combine results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Calculate performance improvements
    print('\n=== Optimized Subsampling Results ===\n')
    
    for config in configs:
        config_df = combined_df[combined_df['subsampling_config'] == config]
        if len(config_df) > 0:
            xgb_df = config_df[config_df['model'].str.contains('xgboost')]
            if len(xgb_df) > 0:
                # Calculate metrics for both in-domain and out-domain
                in_domain = xgb_df[xgb_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
                out_domain = xgb_df[xgb_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
                
                # Also calculate COD accuracy
                in_domain_cod = xgb_df[xgb_df['experiment_type'] == 'in_domain']['cod_accuracy'].mean()
                out_domain_cod = xgb_df[xgb_df['experiment_type'] == 'out_domain']['cod_accuracy'].mean()
                
                # Overall average (what user cares about)
                overall_csmf = xgb_df['csmf_accuracy'].mean()
                overall_cod = xgb_df['cod_accuracy'].mean()
                
                print(f'{config.upper()} Configuration:')
                print(f'  In-domain CSMF: {in_domain:.4f} (COD: {in_domain_cod:.4f})')
                print(f'  Out-domain CSMF: {out_domain:.4f} (COD: {out_domain_cod:.4f})')
                print(f'  Overall CSMF: {overall_csmf:.4f} (COD: {overall_cod:.4f})')
                print(f'  Execution time: {xgb_df[\"execution_time_seconds\"].mean():.1f}s')
                print()
    
    # Calculate improvements
    baseline_df = combined_df[combined_df['subsampling_config'] == 'baseline_enhanced']
    optimized_df = combined_df[combined_df['subsampling_config'] == 'optimized_subsampling']
    
    if len(baseline_df) > 0 and len(optimized_df) > 0:
        baseline_xgb = baseline_df[baseline_df['model'].str.contains('xgboost')]
        optimized_xgb = optimized_df[optimized_df['model'].str.contains('xgboost')]
        
        if len(baseline_xgb) > 0 and len(optimized_xgb) > 0:
            # Calculate overall improvements
            baseline_overall = baseline_xgb['csmf_accuracy'].mean()
            optimized_overall = optimized_xgb['csmf_accuracy'].mean()
            improvement = ((optimized_overall - baseline_overall) / baseline_overall * 100)
            
            print('\n>>> IMPROVEMENT SUMMARY:')
            print(f'Overall CSMF improvement: {improvement:+.1f}%')
            print(f'Baseline: {baseline_overall:.4f} → Optimized: {optimized_overall:.4f}')
            
            # Compare to InSilicoVA
            insilico_df = combined_df[combined_df['model'] == 'insilico']
            if len(insilico_df) > 0:
                insilico_overall = insilico_df['csmf_accuracy'].mean()
                baseline_gap = ((insilico_overall - baseline_overall) / insilico_overall * 100)
                optimized_gap = ((insilico_overall - optimized_overall) / insilico_overall * 100)
                print(f'\nGap to InSilicoVA:')
                print(f'Baseline gap: {baseline_gap:.1f}%')
                print(f'Optimized gap: {optimized_gap:.1f}%')
                print(f'Gap reduction: {baseline_gap - optimized_gap:.1f} percentage points')
    
    # Save combined results
    combined_df.to_csv(base_path / 'optimized_subsampling_comparison.csv', index=False)
    print(f'\nCombined results saved to: {base_path}/optimized_subsampling_comparison.csv')
else:
    print('\nERROR: No results found to compare!')
    sys.exit(1)
"

echo ""
echo "Experiment completed at $(date)"
echo "Results saved to: $OUTPUT_BASE"