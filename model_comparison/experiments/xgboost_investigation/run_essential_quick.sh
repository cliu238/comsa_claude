#!/bin/bash
# Quick essential XGBoost experiments using existing scripts
# This version actually works with the available infrastructure

set -e

echo "=============================================="
echo "XGBoost Essential Experiments (Quick Mode)"
echo "=============================================="
echo "This will run a minimal set of experiments to test the setup"
echo ""

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_BASE="results/xgboost_investigation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_REPORT="$OUTPUT_BASE/quick_test_${TIMESTAMP}"

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$FINAL_REPORT"

# Use the correct data path
DATA_PATH="data/va34/va34_data.csv"

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    echo "Please ensure the VA34 data is available"
    exit 1
fi

echo "Starting quick test at $(date)" | tee "$FINAL_REPORT/test_log.txt"
echo "Output directory: $FINAL_REPORT" | tee -a "$FINAL_REPORT/test_log.txt"
echo "" | tee -a "$FINAL_REPORT/test_log.txt"

# Quick test with minimal parameters
echo "======================================" | tee -a "$FINAL_REPORT/test_log.txt"
echo "Running Quick Test" | tee -a "$FINAL_REPORT/test_log.txt"
echo "======================================" | tee -a "$FINAL_REPORT/test_log.txt"

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "$DATA_PATH" \
    --sites Mexico AP Pemba \
    --models xgboost insilico \
    --n-bootstrap 5 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 10 \
    --checkpoint-interval 5 \
    --output-dir "$FINAL_REPORT" \
    --track-component-times \
    --random-seed 42 \
    2>&1 | tee -a "$FINAL_REPORT/test_log.txt"

echo "" | tee -a "$FINAL_REPORT/test_log.txt"
echo "Quick test completed at $(date)" | tee -a "$FINAL_REPORT/test_log.txt"
echo "Results saved to: $FINAL_REPORT" | tee -a "$FINAL_REPORT/test_log.txt"

# Check if results were generated
if [ -f "$FINAL_REPORT/va34_comparison_results.csv" ]; then
    echo ""
    echo "✓ Test completed successfully!"
    echo "✓ Results file generated: $FINAL_REPORT/va34_comparison_results.csv"
    
    # Show quick summary
    echo ""
    echo "Quick Summary:"
    poetry run python -c "
import pandas as pd
df = pd.read_csv('$FINAL_REPORT/va34_comparison_results.csv')
print(f'Total experiments: {len(df)}')
print(f'Models tested: {df[\"model\"].unique()}')
print(f'Sites tested: {df[\"site\"].unique()}')
print(f'\\nAverage CSMF accuracy by model:')
print(df.groupby('model')['csmf_accuracy'].mean())
"
else
    echo ""
    echo "✗ No results file generated - check the log for errors"
fi