#!/usr/bin/env python3
"""Example usage script for the VA data splitting module.

This script demonstrates how to use the VADataSplitter class with different
split strategies for future data analysis workflows.

Usage:
    python baseline/example_data_splitting.py

Expected runtime: <1 minute
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.config.data_config import DataConfig
from baseline.data.data_splitter import VADataSplitter

def create_sample_data():
    """Create sample VA data for demonstration."""
    np.random.seed(42)
    
    # Create sample data with realistic VA structure
    n_samples = 1000
    sites = ['Pemba', 'Matlab', 'Nairobi', 'Bohol', 'Vadu']
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Common VA causes
    
    data = {
        'site': np.random.choice(sites, n_samples),
        'va34': np.random.choice(labels, n_samples),
        'feature1': np.random.randint(0, 2, n_samples),
        'feature2': np.random.randint(0, 2, n_samples),
        'feature3': np.random.randint(0, 2, n_samples),
        'feature4': np.random.randint(0, 2, n_samples),
        'feature5': np.random.randint(0, 2, n_samples),
    }
    
    return pd.DataFrame(data)

def main():
    """Demonstrate different data splitting strategies."""
    print("=" * 60)
    print("VA Data Splitting Example")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data for demonstration
    print("\n1. Creating sample VA data...")
    data = create_sample_data()
    
    print(f"Sample data shape: {data.shape}")
    print(f"Sites available: {sorted(data['site'].unique())}")
    print(f"Labels available: {sorted(data['va34'].unique())}")
    print(f"Site distribution: {data['site'].value_counts().to_dict()}")
    
    # Example 1: Standard Train/Test Split
    print("\n2. Example 1: Standard Train/Test Split")
    print("-" * 40)
    
    # Create a temporary CSV file for DataConfig validation
    temp_csv = "temp_sample_data.csv"
    data.to_csv(temp_csv, index=False)
    
    config1 = DataConfig(
        data_path=temp_csv,
        split_strategy="train_test",
        test_size=0.2,  # 20% test, 80% train
        random_state=42
    )
    
    splitter1 = VADataSplitter(config1)
    splits1 = splitter1.split_data(data)
    stats1 = splitter1.get_split_statistics(splits1)
    
    print(f"Train set: {stats1['train']['n_samples']} samples")
    print(f"Test set: {stats1['test']['n_samples']} samples")
    print(f"Features: {stats1['train']['n_features']}")
    print(f"Classes: {stats1['train']['n_classes']}")
    
    # Example 2: Cross-Site Split
    print("\n3. Example 2: Cross-Site Split")
    print("-" * 40)
    
    # Get available sites for demonstration
    available_sites = sorted(data['site'].unique())
    train_sites = available_sites[:3]  # First 3 sites for training
    test_sites = available_sites[3:]   # Remaining sites for testing
    
    print(f"Using sites for training: {train_sites}")
    print(f"Using sites for testing: {test_sites}")
    
    config2 = DataConfig(
        data_path=temp_csv,
        split_strategy="cross_site",
        train_sites=train_sites,
        test_sites=test_sites,
        random_state=42
    )
    
    splitter2 = VADataSplitter(config2)
    splits2 = splitter2.split_data(data)
    stats2 = splitter2.get_split_statistics(splits2)
    
    print(f"Train set: {stats2['train']['n_samples']} samples from {stats2['train']['n_sites']} sites")
    print(f"Test set: {stats2['test']['n_samples']} samples from {stats2['test']['n_sites']} sites")
    
    # Example 3: Stratified Site Split
    print("\n4. Example 3: Stratified Site Split")
    print("-" * 40)
    
    config3 = DataConfig(
        data_path=temp_csv,
        split_strategy="stratified_site",
        test_size=0.25,  # 25% test from each site
        random_state=42
    )
    
    splitter3 = VADataSplitter(config3)
    splits3 = splitter3.split_data(data)
    stats3 = splitter3.get_split_statistics(splits3)
    
    print(f"Train set: {stats3['train']['n_samples']} samples from {stats3['train']['n_sites']} sites")
    print(f"Test set: {stats3['test']['n_samples']} samples from {stats3['test']['n_sites']} sites")
    
    # Show site distribution for stratified split
    print("\nSite distribution in stratified split:")
    for split_name, split_data in splits3.items():
        site_dist = split_data['site'].value_counts().sort_index()
        print(f"  {split_name.capitalize()}: {site_dist.to_dict()}")
    
    # Example 4: Reproducibility Test
    print("\n5. Example 4: Reproducibility Test")
    print("-" * 40)
    
    # Run same split twice with same random state
    config4a = DataConfig(
        data_path=temp_csv,
        split_strategy="train_test",
        test_size=0.3,
        random_state=123
    )
    
    config4b = DataConfig(
        data_path=temp_csv,
        split_strategy="train_test",
        test_size=0.3,
        random_state=123  # Same random state
    )
    
    splitter4a = VADataSplitter(config4a)
    splitter4b = VADataSplitter(config4b)
    
    splits4a = splitter4a.split_data(data)
    splits4b = splitter4b.split_data(data)
    
    # Check if splits are identical
    are_identical = (
        splits4a['train'].equals(splits4b['train']) and
        splits4a['test'].equals(splits4b['test'])
    )
    
    print(f"Same random state produces identical splits: {are_identical}")
    
    # Example 5: Configuration Validation
    print("\n6. Example 5: Configuration Validation")
    print("-" * 40)
    
    try:
        # This should fail - test_size out of range
        DataConfig(
            data_path=temp_csv,
            test_size=1.5  # Invalid - > 1.0
        )
        print("ERROR: Should have failed validation!")
    except ValueError as e:
        print(f"✓ Configuration validation works: {e}")
    
    try:
        # This should fail - invalid split strategy
        DataConfig(
            data_path=temp_csv,
            split_strategy="invalid_strategy"
        )
        print("ERROR: Should have failed validation!")
    except ValueError as e:
        print(f"✓ Split strategy validation works: {e}")
    
    # Clean up temp file
    import os
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    # Summary of key features
    print("\nKey Features Demonstrated:")
    print("• Standard train/test split with stratification")
    print("• Cross-site validation for generalization testing")
    print("• Stratified site split maintaining distributions")
    print("• Reproducible splits with random state control")
    print("• Configuration validation and error handling")
    print("• Detailed statistics and logging")
    
    print("\nNext Steps for Analysis:")
    print("• Use train/test splits for model evaluation")
    print("• Apply cross-site splits for domain adaptation")
    print("• Leverage stratified splits for balanced analysis")
    print("• Combine with existing VA processing pipeline")


if __name__ == "__main__":
    main()