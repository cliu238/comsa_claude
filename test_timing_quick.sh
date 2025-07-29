#!/bin/bash
# Quick test script for component timing tracking

echo "Running quick timing test with 2 sites, 2 models, minimal bootstrap..."

poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP \
    --models logistic_regression xgboost \
    --n-workers 4 \
    --training-sizes 1.0 \
    --n-bootstrap 5 \
    --enable-tuning \
    --tuning-trials 3 \
    --tuning-cv-folds 3 \
    --track-component-times \
    --output-dir results/timing_test_$(date +%Y%m%d_%H%M%S) \
    --no-plots

echo "Test complete! Check the output CSV for timing columns:"
echo "- tuning_time_seconds"
echo "- training_time_seconds" 
echo "- inference_time_seconds"
echo "- execution_time_seconds"