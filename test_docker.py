#!/usr/bin/env python
"""Test InSilicoVA Docker execution directly."""

import sys
sys.path.insert(0, 'va-data')

import pandas as pd
import numpy as np
import logging
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.model_config import InSilicoVAConfig
from baseline.config.data_config import DataConfig
from baseline.data.data_loader_preprocessor import VADataProcessor

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load some real data
print("Loading real PHMRC data...")
config = DataConfig(
    data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
    output_dir="test_output/",
    openva_encoding=True,
    stratify_by_site=True
)

processor = VADataProcessor(config)
data = processor.load_and_process()

# Take a small sample
print(f"Data shape: {data.shape}")
sample = data.sample(n=200, random_state=42)

# Prepare X and y
label_col = 'va34'
# Exclude ALL label columns, not just va34
label_cols = ['va34', 'va5', 'cod5', 'cod', 'va12', 'va15', 'va20', 'va55']
metadata_cols = ['site', 'subpop']
feature_cols = [col for col in sample.columns if col not in label_cols + metadata_cols]
X = sample[feature_cols]
y = sample[label_col]

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Unique causes: {y.nunique()}")

# Test with InSilicoVA
print("\nTesting InSilicoVA with Docker...")
model_config = InSilicoVAConfig(
    docker_image="insilicova-arm64:latest",
    docker_platform="linux/arm64",
    auto_tune=False,
    n_sim=100,
    verbose=True,
    output_dir="test_output/",
    debug_mode=True  # Enable debug mode for more detailed error output
)

model = InSilicoVAModel(model_config)

try:
    print("Fitting model...")
    # Check data format before fitting
    print(f"Sample of X values (first 3 columns, first 5 rows):")
    print(X.iloc[:5, :3])
    print(f"Unique values in first column: {X.iloc[:, 0].unique()[:10]}")
    
    model.fit(X, y)
    print("✓ Model fitted successfully!")
    
    # Test prediction
    X_test = X.iloc[:10]
    print(f"\nTesting prediction on {len(X_test)} samples...")
    predictions = model.predict(X_test)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()