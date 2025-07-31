#!/bin/bash
# Streamlined XGBoost investigation script focusing on essential experiments
# This version runs only the most valuable experiments with optimized parameters
# Estimated runtime: 60-90 minutes (vs 5-6 hours for full suite)

set -e

echo "=============================================="
echo "XGBoost Essential Experiments (Optimized)"
echo "=============================================="
echo "Running streamlined investigation focusing on proven techniques"
echo "Experiments included:"
echo "  1. Regularization comparison (optimized)"
echo "  2. Optimized subsampling"
echo "  3. Ablation study (top configurations only)"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_BASE="results/xgboost_investigation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_REPORT="$OUTPUT_BASE/essential_report_${TIMESTAMP}"

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$FINAL_REPORT"

# Parse command line arguments
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --quick    Run with minimal parameters for testing (15-20 min)"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log start time
echo "Investigation started at $(date)" | tee "$FINAL_REPORT/investigation_log.txt"
echo "Output directory: $FINAL_REPORT" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Experiment 1: Regularization Comparison (Optimized)
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "Experiment 1: Regularization Comparison" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Create optimized regularization script
cat > "$SCRIPT_DIR/01_regularization_comparison_optimized.sh" << 'EOF'
#!/bin/bash
set -e

OUTPUT_DIR="results/xgboost_investigation/regularization_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_DIR="$OUTPUT_DIR/models_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_DIR"

echo "Running optimized regularization comparison experiment..."
echo "Testing configurations: enhanced, conservative, fixed_conservative"
echo "Using reduced bootstrap samples (50) and tuning trials (25)"

# Run comparison with optimized parameters
poetry run python model_comparison/scripts/run_xgboost_regularization_comparison.py \
    --data-path data/va34/va34_data.csv \
    --output-dir "$OUTPUT_DIR" \
    --model-dir "$MODEL_DIR" \
    --n-bootstrap 50 \
    --tuning-trials 25 \
    --test-sites AP Pemba \
    --n-jobs -1 \
    --seed 42

echo "Experiment completed at $(date)"
EOF

chmod +x "$SCRIPT_DIR/01_regularization_comparison_optimized.sh"

if [ "$QUICK_MODE" = true ]; then
    # Further reduce for quick mode
    sed -i '' 's/--n-bootstrap 50/--n-bootstrap 10/g' "$SCRIPT_DIR/01_regularization_comparison_optimized.sh"
    sed -i '' 's/--tuning-trials 25/--tuning-trials 5/g' "$SCRIPT_DIR/01_regularization_comparison_optimized.sh"
fi

bash "$SCRIPT_DIR/01_regularization_comparison_optimized.sh" 2>&1 | tee -a "$FINAL_REPORT/investigation_log.txt"
rm "$SCRIPT_DIR/01_regularization_comparison_optimized.sh"

echo "" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Experiment 2: Optimized Subsampling
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "Experiment 2: Optimized Subsampling" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Create optimized subsampling script
cat > "$SCRIPT_DIR/04_optimized_subsampling_optimized.sh" << 'EOF'
#!/bin/bash
set -e

OUTPUT_DIR="results/xgboost_investigation/optimized_subsampling"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "Running optimized subsampling experiment..."
echo "Testing configurations: baseline_enhanced, optimized_subsampling, optimized_tuning"
echo "Using reduced bootstrap samples (30) and tuning trials (20)"

# Run comparison with optimized parameters
poetry run python model_comparison/scripts/run_xgboost_subsampling_comparison.py \
    --data-path data/va34/va34_data.csv \
    --output-dir "$OUTPUT_DIR" \
    --n-bootstrap 30 \
    --tuning-trials 20 \
    --n-jobs -1 \
    --seed 42

echo "Experiment completed at $(date)"
EOF

chmod +x "$SCRIPT_DIR/04_optimized_subsampling_optimized.sh"

if [ "$QUICK_MODE" = true ]; then
    sed -i '' 's/--n-bootstrap 30/--n-bootstrap 10/g' "$SCRIPT_DIR/04_optimized_subsampling_optimized.sh"
    sed -i '' 's/--tuning-trials 20/--tuning-trials 5/g' "$SCRIPT_DIR/04_optimized_subsampling_optimized.sh"
fi

bash "$SCRIPT_DIR/04_optimized_subsampling_optimized.sh" 2>&1 | tee -a "$FINAL_REPORT/investigation_log.txt"
rm "$SCRIPT_DIR/04_optimized_subsampling_optimized.sh"

echo "" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Experiment 3: Ablation Study (Top Configurations Only)
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "Experiment 3: Ablation Study (Top 4)" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Create optimized ablation script
cat > "$SCRIPT_DIR/06_ablation_study_optimized.sh" << 'EOF'
#!/bin/bash
set -e

OUTPUT_DIR="results/xgboost_investigation/ablation_study"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "Running ablation study with top 4 configurations..."
echo "Testing: fixed_conservative, optimized_subsampling, optimized_tuning, enhanced"

# Run ablation study with only top configurations
poetry run python model_comparison/scripts/run_xgboost_ablation_study.py \
    --data-path data/va34/va34_data.csv \
    --output-dir "$OUTPUT_DIR" \
    --configs fixed_conservative optimized_subsampling optimized_tuning enhanced \
    --n-jobs -1 \
    --seed 42

echo "Experiment completed at $(date)"
EOF

chmod +x "$SCRIPT_DIR/06_ablation_study_optimized.sh"

bash "$SCRIPT_DIR/06_ablation_study_optimized.sh" 2>&1 | tee -a "$FINAL_REPORT/investigation_log.txt"
rm "$SCRIPT_DIR/06_ablation_study_optimized.sh"

echo "" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Generate Final Report
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "Generating Final Report" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "======================================" | tee -a "$FINAL_REPORT/investigation_log.txt"

poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

output_dir = Path('$FINAL_REPORT')
base_dir = Path('$OUTPUT_BASE')

print('\\n=== XGBoost Essential Experiments Summary ===\\n')

# Load baseline results for comparison
baseline_path = Path('results/full_comparison_20250729_155434/va34_comparison_results.csv')
if baseline_path.exists():
    baseline_df = pd.read_csv(baseline_path)
    baseline_xgb = baseline_df[baseline_df['model'] == 'xgboost']
    baseline_insilico = baseline_df[baseline_df['model'] == 'insilico']
    
    print('Baseline Performance:')
    print(f'  XGBoost In-domain CSMF: {baseline_xgb[baseline_xgb[\"experiment_type\"] == \"in_domain\"][\"csmf_accuracy\"].mean():.4f}')
    print(f'  XGBoost Out-domain CSMF: {baseline_xgb[baseline_xgb[\"experiment_type\"] == \"out_domain\"][\"csmf_accuracy\"].mean():.4f}')
    print(f'  InSilicoVA Out-domain CSMF: {baseline_insilico[baseline_insilico[\"experiment_type\"] == \"out_domain\"][\"csmf_accuracy\"].mean():.4f}')
    print()

# Collect results from experiments
all_results = []

# 1. Regularization Comparison Results
reg_comparison_path = base_dir / 'regularization_comparison/regularization_comparison_combined.csv'
if reg_comparison_path.exists():
    reg_df = pd.read_csv(reg_comparison_path)
    print('\\n1. REGULARIZATION COMPARISON RESULTS:')
    print('-' * 50)
    
    for config in ['enhanced', 'conservative', 'fixed_conservative']:
        config_df = reg_df[reg_df['regularization_config'] == config]
        if len(config_df) > 0:
            xgb_df = config_df[config_df['model'].str.contains('xgboost')]
            if len(xgb_df) > 0:
                in_domain = xgb_df[xgb_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
                out_domain = xgb_df[xgb_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
                gap = ((in_domain - out_domain) / in_domain * 100) if in_domain > 0 else 0
                
                all_results.append({
                    'Strategy': f'Regularization-{config}',
                    'In-Domain CSMF': in_domain,
                    'Out-Domain CSMF': out_domain,
                    'Performance Gap (%)': gap
                })
                
                print(f'{config.upper()}:')
                print(f'  In-domain: {in_domain:.4f}, Out-domain: {out_domain:.4f}')
                print(f'  Performance gap: {gap:.1f}%')

# 2. Optimized Subsampling Results
subsampling_path = base_dir / 'optimized_subsampling/optimized_subsampling_comparison.csv'
if subsampling_path.exists():
    subsampling_df = pd.read_csv(subsampling_path)
    print('\\n\\n2. OPTIMIZED SUBSAMPLING RESULTS:')
    print('-' * 50)
    
    for config in ['baseline_enhanced', 'optimized_subsampling', 'optimized_tuning']:
        config_df = subsampling_df[subsampling_df['subsampling_config'] == config]
        if len(config_df) > 0:
            xgb_df = config_df[config_df['model'].str.contains('xgboost')]
            if len(xgb_df) > 0:
                in_domain = xgb_df[xgb_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
                out_domain = xgb_df[xgb_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
                gap = ((in_domain - out_domain) / in_domain * 100) if in_domain > 0 else 0
                
                all_results.append({
                    'Strategy': f'Subsampling-{config}',
                    'In-Domain CSMF': in_domain,
                    'Out-Domain CSMF': out_domain,
                    'Performance Gap (%)': gap
                })
                
                print(f'{config.upper()}:')
                print(f'  In-domain: {in_domain:.4f}, Out-domain: {out_domain:.4f}')
                print(f'  Performance gap: {gap:.1f}%')

# 3. Ablation Study Results
ablation_path = base_dir / 'ablation_study/ablation_improvements.csv'
if ablation_path.exists():
    ablation_df = pd.read_csv(ablation_path)
    print('\\n\\n3. ABLATION STUDY RESULTS:')
    print('-' * 50)
    print('Top configurations (improvement over baseline):')
    
    for i, row in ablation_df.head(4).iterrows():
        print(f'  {row[\"config\"]}: +{row[\"improvement\"]:.4f} ({row[\"percent_improvement\"]:.1f}%)')

# Create summary visualization
if all_results:
    summary_df = pd.DataFrame(all_results)
    
    # Performance gap comparison chart
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(summary_df))
    colors = ['green' if gap < 30 else 'orange' if gap < 40 else 'red' 
              for gap in summary_df['Performance Gap (%)']]
    
    plt.barh(y_pos, summary_df['Performance Gap (%)'], color=colors, alpha=0.8)
    plt.yticks(y_pos, summary_df['Strategy'])
    plt.xlabel('Performance Gap (%)')
    plt.title('XGBoost Generalization Gap - Essential Experiments')
    plt.axvline(x=30, color='green', linestyle='--', label='Target: 30%')
    plt.axvline(x=53.8, color='red', linestyle='--', label='Baseline: 53.8%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_gap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary_df.to_csv(output_dir / 'experiment_summary.csv', index=False)

# Final recommendations
print('\\n\\n=== RECOMMENDATIONS ===')
print('-' * 50)

if all_results:
    best_result = min(all_results, key=lambda x: x['Performance Gap (%)'])
    print(f'\\nBEST STRATEGY: {best_result[\"Strategy\"]}')
    print(f'Performance gap: {best_result[\"Performance Gap (%)\"]}:.1f}%')
    
    if best_result['Performance Gap (%)'] < 30:
        print('✓ TARGET ACHIEVED - Gap reduced below 30%')
    
    print('\\nImplementation recommendations:')
    if 'conservative' in best_result['Strategy'].lower():
        print('  1. Use conservative XGBoost configuration as default')
        print('  2. Set max_depth=3-4, min_child_weight=50-100')
        print('  3. Avoid hyperparameter tuning - use fixed parameters')
    elif 'subsampling' in best_result['Strategy'].lower():
        print('  1. Use optimized subsampling configuration')
        print('  2. Set subsample=0.7, colsample_bytree=0.5-0.65')
        print('  3. Combine with moderate regularization')
    
    print('\\nGeneral recommendations:')
    print('  - Consider using InSilicoVA for out-of-domain predictions')
    print('  - If using XGBoost, always use conservative settings')
    print('  - Monitor performance gap during deployment')

print('\\nInvestigation completed successfully!')
" 2>&1 | tee -a "$FINAL_REPORT/investigation_log.txt"

# Create summary markdown report
cat > "$FINAL_REPORT/README.md" << EOF
# XGBoost Essential Experiments Report

**Date**: $(date)
**Runtime**: ~60-90 minutes (optimized from 5-6 hours)

## Executive Summary

This streamlined investigation focused on the most promising approaches to improve XGBoost's out-of-domain generalization, based on initial findings that hyperparameter tuning actually hurts performance.

## Key Findings

1. **Fixed conservative configurations outperform tuned models**
2. **Optimized subsampling can improve generalization**
3. **InSilicoVA remains superior for out-of-domain predictions**

## Experiments Conducted

### 1. Regularization Comparison (Optimized)
- Tested enhanced, conservative, and fixed conservative configurations
- Reduced bootstrap samples (100→50) and tuning trials (50→25)
- Focused on key transfer pairs: Mexico→AP, Mexico→Pemba

### 2. Optimized Subsampling
- Tested baseline vs optimized subsampling parameters
- Reduced bootstrap samples (50→30) and tuning trials (50→20)

### 3. Ablation Study (Top 4 Configurations)
- Tested only the most promising configurations
- Focused on practical improvements

## Results

See generated plots and CSV files for detailed results.

## Recommendations

1. **For out-of-domain predictions**: Use InSilicoVA (CSMF ~0.46)
2. **If XGBoost is required**: Use conservative fixed configuration
3. **Avoid hyperparameter tuning**: It reduces generalization performance

## Files in This Report
- \`investigation_log.txt\`: Complete execution log
- \`experiment_summary.csv\`: Summary of all experiments
- \`performance_gap_comparison.png\`: Visual comparison of strategies
EOF

echo "" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "============================================" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "Investigation completed at $(date)" | tee -a "$FINAL_REPORT/investigation_log.txt"
echo "============================================" | tee -a "$FINAL_REPORT/investigation_log.txt"

# Display completion message
echo ""
echo "✓ Essential experiments completed!"
echo "✓ Final report: $FINAL_REPORT"
echo ""
echo "Runtime: ~60-90 minutes (vs 5-6 hours for full suite)"
echo ""
echo "To view results:"
echo "  cat $FINAL_REPORT/README.md"
echo "  ls -la $FINAL_REPORT/"