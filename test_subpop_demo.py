#!/usr/bin/env python
"""
Demonstration of sub.pop parameter effect in InSilicoVA
Shows the actual model outputs with and without subpopulation labels
"""

import sys
sys.path.insert(0, 'va-data')

import pandas as pd
import numpy as np
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.model_config import InSilicoVAConfig
from baseline.data.data_loader_preprocessor import VADataProcessor
from baseline.config.data_config import DataConfig

print("=" * 60)
print("SUB.POP PARAMETER DEMONSTRATION")
print("=" * 60)

# Load real data
config = DataConfig(
    data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
    output_dir="test_output/",
    openva_encoding=True,
    stratify_by_site=True
)

processor = VADataProcessor(config)
data = processor.load_and_process()

# Take small samples from two sites
ap_data = data[data['site'] == 'AP'].sample(n=100, random_state=42)
mexico_data = data[data['site'] == 'Mexico'].sample(n=100, random_state=42)

# Prepare training data (mix of both sites)
train_data = pd.concat([
    ap_data.iloc[:50],      # 50 from AP
    mexico_data.iloc[:50]   # 50 from Mexico
], ignore_index=True)

# Prepare test data
test_data = pd.concat([
    ap_data.iloc[50:60],    # 10 from AP for testing
    mexico_data.iloc[50:60] # 10 from Mexico for testing
], ignore_index=True)

# Setup features and labels
label_cols = ['va34', 'va5', 'cod5', 'cod', 'va12', 'va15', 'va20', 'va55']
metadata_cols = ['site', 'subpop']
feature_cols = [col for col in train_data.columns if col not in label_cols + metadata_cols]

# Model configuration
model_config = InSilicoVAConfig(
    docker_image="insilicova-arm64:latest",
    docker_platform="linux/arm64",
    auto_tune=False,
    n_sim=1000,  # Reduced for demo
    verbose=True,
    output_dir="test_output/",
    debug_mode=True
)

print("\n1. RUNNING WITHOUT SUB.POP (Baseline)")
print("-" * 40)

X_train_base = train_data[feature_cols]
y_train = train_data['va34']
X_test = test_data[feature_cols]
y_test = test_data['va34']

model_baseline = InSilicoVAModel(model_config)
try:
    model_baseline.fit(X_train_base, y_train)
    pred_baseline = model_baseline.predict(X_test)
    print(f"✓ Baseline predictions shape: {pred_baseline.shape}")
    print(f"  Sample predictions (first 5): {pred_baseline[:5]}")
    print(f"  Unique predicted causes: {len(np.unique(pred_baseline))}")
except Exception as e:
    print(f"✗ Baseline failed: {e}")
    pred_baseline = None

print("\n2. RUNNING WITH SUB.POP (Treatment)")
print("-" * 40)

# Add subpopulation labels
subpop_labels = []
for idx, row in train_data.iterrows():
    if row['site'] == 'Mexico':
        subpop_labels.append('1')  # Source population
    else:
        subpop_labels.append('2')  # Target population

print(f"Subpop distribution: '1' (Mexico): {subpop_labels.count('1')}, '2' (AP): {subpop_labels.count('2')}")

model_treatment = InSilicoVAModel(model_config)
try:
    model_treatment.fit(X_train_base, y_train, subpop_labels=pd.Series(subpop_labels))
    pred_treatment = model_treatment.predict(X_test)
    print(f"✓ Treatment predictions shape: {pred_treatment.shape}")
    print(f"  Sample predictions (first 5): {pred_treatment[:5]}")
    print(f"  Unique predicted causes: {len(np.unique(pred_treatment))}")
except Exception as e:
    print(f"✗ Treatment failed: {e}")
    pred_treatment = None

print("\n3. COMPARISON OF RESULTS")
print("-" * 40)

if pred_baseline is not None and pred_treatment is not None:
    # Check if predictions are identical
    identical = np.array_equal(pred_baseline, pred_treatment)
    print(f"Predictions identical? {identical}")
    
    if not identical:
        # Count differences
        differences = (pred_baseline != pred_treatment).sum()
        print(f"Number of different predictions: {differences}/{len(pred_baseline)}")
        
        # Show examples of differences
        diff_indices = np.where(pred_baseline != pred_treatment)[0][:3]
        for idx in diff_indices:
            print(f"  Test sample {idx}: Baseline={pred_baseline[idx]}, Treatment={pred_treatment[idx]}")
    
    # Calculate accuracies
    from sklearn.metrics import accuracy_score
    acc_baseline = accuracy_score(y_test.astype(str), pred_baseline.astype(str))
    acc_treatment = accuracy_score(y_test.astype(str), pred_treatment.astype(str))
    
    print(f"\nAccuracy comparison:")
    print(f"  Baseline (no sub.pop):    {acc_baseline:.3f}")
    print(f"  Treatment (with sub.pop): {acc_treatment:.3f}")
    print(f"  Improvement:              {acc_treatment - acc_baseline:.3f}")
else:
    print("Could not complete comparison due to errors")

print("\n" + "=" * 60)
print("CONCLUSION:")
if pred_baseline is not None and pred_treatment is not None:
    if identical:
        print("Sub.pop parameter has NO EFFECT on individual predictions")
        print("The model produces IDENTICAL outputs with or without sub.pop")
    else:
        print(f"Sub.pop parameter changed {differences} predictions")
        print(f"Accuracy change: {(acc_treatment - acc_baseline)*100:.1f}%")
print("=" * 60)