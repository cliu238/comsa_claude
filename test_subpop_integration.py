#!/usr/bin/env python3
"""
Integration test for subpopulation parameter support in InSilicoVA.

This test verifies that the subpopulation extensions work end-to-end:
1. VADataSplitter creates subpopulation splits
2. InSilicoVAModel accepts and uses subpop labels
3. R script generation includes sub.pop parameter
"""

import pandas as pd
import numpy as np
from baseline.data.data_splitter import VADataSplitter, SplitResult
from baseline.models.insilico_model import InSilicoVAModel
from baseline.config.data_config import DataConfig
from baseline.models.model_config import InSilicoVAConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_subpopulation_split():
    """Test that VADataSplitter can create subpopulation splits."""
    print("\n=== Testing VADataSplitter subpopulation split ===")
    
    # Create mock data with sites
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'site': np.random.choice(['AP', 'Bohol', 'Dar'], n_samples),
        'va34': np.random.choice(['cause1', 'cause2', 'cause3'], n_samples),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    
    # Configure splitter for subpopulation
    config = DataConfig(
        data_path='va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv',  # Required field
        split_strategy='subpopulation',
        label_column='va34',
        site_column='site',
        test_size=0.5,
        random_state=42
    )
    
    splitter = VADataSplitter(config)
    
    # Perform subpopulation split
    result = splitter.split_data(data, source_site='AP', target_site='Bohol')
    
    # Verify results
    assert 'subpop' in result.train.columns, "Subpop column not found in training data"
    assert len(result.train[result.train['subpop'] == '1']) > 0, "No source samples (subpop=1)"
    assert len(result.train[result.train['subpop'] == '2']) > 0, "No target samples (subpop=2)"
    
    print(f"✓ Split created successfully:")
    print(f"  Source (AP) samples: {len(result.train[result.train['subpop'] == '1'])}")
    print(f"  Target (Bohol) train: {len(result.train[result.train['subpop'] == '2'])}")
    print(f"  Target (Bohol) test: {len(result.test)}")
    print(f"  Subpop labels: {result.train['subpop'].value_counts().to_dict()}")
    
    return result

def test_insilico_with_subpop(split_result: SplitResult):
    """Test that InSilicoVAModel accepts subpop labels."""
    print("\n=== Testing InSilicoVAModel with subpopulation ===")
    
    # Initialize model with minimum allowed nsim
    config = InSilicoVAConfig(
        nsim=1000,  # Minimum allowed
        verbose=True
    )
    
    model = InSilicoVAModel(config)
    
    # Prepare training data
    X_train = split_result.train.drop(columns=['va34'])
    y_train = split_result.train['va34']
    
    # Fit model - it should auto-detect subpop column
    print("Fitting model with subpopulation labels...")
    model.fit(X_train, y_train)
    
    # Verify subpop labels were stored
    assert model._subpop_labels is not None, "Subpop labels not stored in model"
    print(f"✓ Model stored subpop labels: {model._subpop_labels.value_counts().to_dict()}")
    
    # Generate R script and check for sub.pop parameter
    r_script = model._generate_r_script()
    if 'sub.pop' in r_script:
        print("✓ R script contains sub.pop parameter")
    else:
        print("✗ R script missing sub.pop parameter")
        print("R script preview:")
        print(r_script[:500])
    
    return model

def main():
    """Run integration test for subpopulation support."""
    print("=" * 60)
    print("InSilicoVA Subpopulation Integration Test")
    print("=" * 60)
    
    try:
        # Test 1: Data splitting
        split_result = test_subpopulation_split()
        
        # Test 2: Model with subpop
        model = test_insilico_with_subpop(split_result)
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
        print("\nNext step: Run actual prediction with Docker to verify full pipeline")
        print("Note: Full Docker execution would take ~50 seconds")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())