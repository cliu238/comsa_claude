#!/bin/bash
# Simple test to verify the setup works

echo "Testing XGBoost investigation setup..."

# Test 1: Check if poetry works
echo "1. Testing poetry..."
poetry --version

# Test 2: Check if the comparison script exists and is importable
echo "2. Testing script import..."
poetry run python -c "import sys; sys.path.append('.'); from model_comparison.scripts.run_distributed_comparison import main; print('✓ Script imported successfully')"

# Test 3: Check if data file is accessible
echo "3. Testing data access..."
poetry run python -c "
import pandas as pd
df = pd.read_csv('va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv')
print(f'✓ Data loaded: {len(df)} rows, {len(df.columns)} columns')
"

# Test 4: Test a minimal run with just 1 bootstrap
echo "4. Running minimal test..."
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path "va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv" \
    --sites Mexico \
    --models xgboost \
    --n-bootstrap 1 \
    --n-workers 1 \
    --memory-per-worker 2GB \
    --output-dir "results/xgboost_investigation/test_run" \
    --random-seed 42

echo "Test complete!"