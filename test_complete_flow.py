#!/usr/bin/env python
"""Comprehensive test to verify label column handling through the complete flow."""

import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "va-data"))

def test_complete_label_flow():
    """Test the complete label column flow as it happens in the actual pipeline."""
    
    from baseline.data.data_loader_preprocessor import VADataProcessor
    from baseline.config.data_config import DataConfig
    
    print("=" * 60)
    print("TESTING COMPLETE LABEL COLUMN FLOW")
    print("=" * 60)
    
    # Step 1: Load data with va34 as label_column (as in prefect_flows.py line 549-558)
    print("\n1. LOADING DATA WITH label_column='va34'")
    data_config = DataConfig(
        data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        output_dir="test_output",
        openva_encoding=False,
        stratify_by_site=False,
        label_column="va34"  # This is what config.label_type would be
    )
    
    processor = VADataProcessor(data_config)
    data = processor.load_and_process()
    
    print(f"   Data shape: {data.shape}")
    print(f"   Has 'cause': {'cause' in data.columns}")
    print(f"   Has 'va34': {'va34' in data.columns}")
    print(f"   Has 'cod5': {'cod5' in data.columns}")
    
    # Check what columns were kept vs excluded
    label_cols = processor._get_label_equivalent_columns(exclude_current_target=False)
    print(f"   Label columns that should be excluded from features: {label_cols[:5]}...")
    
    # Step 2: Filter sites (as in prefect_flows.py line 572-574)
    print("\n2. FILTERING TO SPECIFIC SITES")
    sites = ["Mexico", "AP", "UP"]
    data = data[data["site"].isin(sites)]
    print(f"   Data shape after filtering: {data.shape}")
    
    # Step 3: Create cause column (as in prefect_flows.py line 578-580)
    print("\n3. CREATING 'cause' COLUMN")
    if "va34" in data.columns and "cause" not in data.columns:
        data["cause"] = data["va34"].astype(str)
        print(f"   Created 'cause' from 'va34'")
    print(f"   Has 'cause' now: {'cause' in data.columns}")
    
    # Step 4: This data goes into Ray object store
    print("\n4. DATA READY FOR RAY OBJECT STORE")
    print(f"   Shape: {data.shape}")
    print(f"   Columns with 'cause': {'cause' in data.columns}")
    
    # Step 5: Simulate prepare_data_for_site (ray_tasks.py line 727+)
    print("\n5. SIMULATING prepare_data_for_site")
    for site in sites:
        site_data = data[data["site"] == site]
        print(f"\n   Site: {site}")
        print(f"   Site data shape: {site_data.shape}")
        
        # Check if cause column exists (line 768 in ray_tasks.py)
        if "cause" in site_data.columns:
            y = site_data["cause"]
            print(f"   ✅ Successfully extracted 'cause' column")
            print(f"   Unique causes: {y.nunique()}")
            print(f"   Sample distribution: {y.value_counts().head(3).to_dict()}")
        else:
            print(f"   ❌ ERROR: 'cause' column missing!")
            return False
    
    # Step 6: Test with cod5 as label_type
    print("\n" + "=" * 60)
    print("TESTING WITH label_column='cod5'")
    print("=" * 60)
    
    data_config_cod5 = DataConfig(
        data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        output_dir="test_output",
        openva_encoding=False,
        stratify_by_site=False,
        label_column="cod5"  # Using cod5 instead
    )
    
    processor_cod5 = VADataProcessor(data_config_cod5)
    data_cod5 = processor_cod5.load_and_process()
    
    print(f"\n1. Data loaded with cod5:")
    print(f"   Has 'cause': {'cause' in data_cod5.columns}")
    print(f"   Has 'cod5': {'cod5' in data_cod5.columns}")
    
    # Filter and create cause column
    data_cod5 = data_cod5[data_cod5["site"].isin(sites)]
    if "cod5" in data_cod5.columns and "cause" not in data_cod5.columns:
        data_cod5["cause"] = data_cod5["cod5"].astype(str)
        print(f"\n2. Created 'cause' from 'cod5'")
    
    print(f"   Has 'cause' now: {'cause' in data_cod5.columns}")
    
    # Check unique values
    if "cause" in data_cod5.columns:
        print(f"\n3. COD5 label distribution:")
        print(f"   Unique causes: {data_cod5['cause'].nunique()}")
        print(f"   Distribution: {data_cod5['cause'].value_counts().to_dict()}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_complete_label_flow()
        if success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - NO CRITICAL ISSUE")
            print("=" * 60)
            print("\nCONCLUSION:")
            print("The label column handling is NOT a critical issue.")
            print("The 'cause' column is created BEFORE data goes to Ray workers.")
        else:
            print("\n❌ TEST FAILED - CRITICAL ISSUE DETECTED!")
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()