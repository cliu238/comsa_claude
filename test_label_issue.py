#!/usr/bin/env python
"""Test script to verify if the label column issue is critical."""

import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "va-data"))

from baseline.data.data_loader_preprocessor import VADataProcessor
from baseline.config.data_config import DataConfig

# Test the actual flow
def test_label_column_flow():
    """Test if the label column handling causes issues."""
    
    # Step 1: Load data as in prefect_flows.py
    data_config = DataConfig(
        data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        output_dir="test_output",
        openva_encoding=False,
        stratify_by_site=False,
        label_column="va34"  # Using va34 as label type
    )
    
    processor = VADataProcessor(data_config)
    data = processor.load_and_process()
    
    print(f"Step 1 - Data loaded: {data.shape}")
    print(f"Columns present: {list(data.columns[:10])}...")
    print(f"Has 'cause' column: {'cause' in data.columns}")
    print(f"Has 'va34' column: {'va34' in data.columns}")
    
    # Step 2: Filter to sites (as in prefect_flows.py line 572-574)
    sites = ["Mexico", "AP"]
    data_filtered = data[data["site"].isin(sites)]
    print(f"\nStep 2 - After site filtering: {data_filtered.shape}")
    
    # Step 3: Create cause column (as in prefect_flows.py line 578-580)
    if "va34" in data_filtered.columns and "cause" not in data_filtered.columns:
        data_filtered["cause"] = data_filtered["va34"].astype(str)
        print(f"\nStep 3 - Created 'cause' column from 'va34'")
    
    print(f"Has 'cause' column now: {'cause' in data_filtered.columns}")
    
    # Step 4: This is what gets passed to prepare_data_for_site
    print(f"\nStep 4 - Data that would be passed to Ray workers:")
    print(f"Shape: {data_filtered.shape}")
    print(f"Has 'cause': {'cause' in data_filtered.columns}")
    
    # Step 5: Simulate what happens in prepare_data_for_site
    site = "Mexico"
    site_data = data_filtered[data_filtered["site"] == site]
    
    try:
        # This is line 768 in ray_tasks.py
        y = site_data["cause"]
        print(f"\nStep 5 - Successfully extracted y labels for site {site}: {len(y)} samples")
        print(f"Unique causes: {y.nunique()}")
    except KeyError as e:
        print(f"\nStep 5 - ERROR: {e}")
        print("This would cause the experiment to fail!")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = test_label_column_flow()
        if success:
            print("\n✅ Test passed - 'cause' column is properly available")
        else:
            print("\n❌ Test failed - Critical issue detected!")
    except Exception as e:
        print(f"\n❌ Test crashed with error: {e}")
        import traceback
        traceback.print_exc()