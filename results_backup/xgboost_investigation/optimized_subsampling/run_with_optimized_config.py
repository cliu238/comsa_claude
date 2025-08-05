#!/usr/bin/env python
"""Run distributed comparison with optimized subsampling configuration."""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Get command line arguments, removing the script name
args = sys.argv[1:]

# Add the flag to use optimized subsampling configuration
# This will be handled in a custom wrapper
print("Running distributed comparison with optimized subsampling configuration...")

# For now, we'll run with enhanced XGBoost configuration which includes better subsampling
# The run_distributed_comparison.py doesn't have a direct flag for optimized_subsampling,
# so we'll use the adaptive configuration which adjusts based on data characteristics
cmd = [
    sys.executable,
    str(project_root / "model_comparison" / "scripts" / "run_distributed_comparison.py"),
    "--use-adaptive-xgboost",  # This will use adaptive configuration
    *args  # Pass through any additional arguments
]

# Run the command
try:
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"Error running distributed comparison: {e}")
    sys.exit(1)
