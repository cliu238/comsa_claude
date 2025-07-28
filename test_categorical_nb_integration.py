#!/usr/bin/env python
"""Quick test script to verify CategoricalNB integration in distributed comparison."""

import subprocess
import sys
from pathlib import Path

def test_basic_run():
    """Test basic run without hyperparameter tuning."""
    print("Testing basic CategoricalNB integration...")
    
    cmd = [
        sys.executable,
        "model_comparison/scripts/run_distributed_comparison.py",
        "--data-path", "va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        "--sites", "AP",
        "--models", "categorical_nb",
        "--training-sizes", "0.5", "1.0",
        "--n-bootstrap", "10",
        "--n-workers", "2",
        "--output-dir", "results/test_categorical_nb_basic",
        "--no-ray-dashboard",
        "--no-prefect-dashboard"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")
    
    return result.returncode == 0

def test_with_tuning():
    """Test with hyperparameter tuning enabled."""
    print("\nTesting CategoricalNB with hyperparameter tuning...")
    
    cmd = [
        sys.executable,
        "model_comparison/scripts/run_distributed_comparison.py",
        "--data-path", "va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        "--sites", "AP",
        "--models", "categorical_nb", "xgboost",
        "--training-sizes", "1.0",
        "--n-bootstrap", "10",
        "--enable-tuning",
        "--tuning-trials", "5",
        "--tuning-algorithm", "random",
        "--n-workers", "2",
        "--output-dir", "results/test_categorical_nb_tuning",
        "--no-ray-dashboard",
        "--no-prefect-dashboard"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")
    
    return result.returncode == 0

if __name__ == "__main__":
    # Check if data file exists
    data_path = Path("va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure PHMRC VA data is available")
        sys.exit(1)
    
    # Run tests
    basic_success = test_basic_run()
    tuning_success = test_with_tuning()
    
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"Basic run: {'PASSED' if basic_success else 'FAILED'}")
    print(f"Tuning run: {'PASSED' if tuning_success else 'FAILED'}")
    print("="*50)
    
    sys.exit(0 if basic_success and tuning_success else 1)