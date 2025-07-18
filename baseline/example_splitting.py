#!/usr/bin/env python
"""Example usage of the VA data splitting functionality.

This script demonstrates how to use the VADataSplitter to split VA data
using different strategies: train_test, cross_site, and stratified_site.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import baseline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Also add va-data to the path
va_data_path = Path(__file__).parent.parent / "va-data"
if va_data_path.exists():
    sys.path.insert(0, str(va_data_path))

from baseline.config.data_config import DataConfig  # noqa: E402
from baseline.data.data_loader_preprocessor import VADataProcessor  # noqa: E402
from baseline.data.data_splitter import VADataSplitter  # noqa: E402


def main() -> int:
    """Run example VA data splitting demonstrations."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Define data file path
    data_file = "data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"

    # Check if data file exists
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please ensure PHMRC data files are in data/raw/PHMRC/")
        logger.info("For demonstration, creating sample data...")
        
        # Create sample data for demonstration
        sample_data = create_sample_data()
        return demonstrate_splitting_with_sample_data(sample_data)

    try:
        # First, load and preprocess the data
        logger.info("=" * 60)
        logger.info("Step 1: Loading and preprocessing data")
        logger.info("=" * 60)
        
        # Load data with numeric encoding for splitting
        config_load = DataConfig(
            data_path=data_file,
            output_dir="results/baseline/",
            openva_encoding=False,  # Numeric encoding
            stratify_by_site=False,
        )
        config_load.setup_logging("INFO")
        
        processor = VADataProcessor(config_load)
        data = processor.load_and_process()
        
        logger.info(f"Loaded {len(data)} records with {data.shape[1]} features")
        logger.info(f"Sites available: {sorted(data['site'].unique())}")
        logger.info(f"Classes available: {sorted(data['va34'].unique())}")
        
        # Now demonstrate different splitting strategies
        return demonstrate_splitting_strategies(data)
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.exception("Full traceback:")
        return 1


def create_sample_data():
    """Create sample data for demonstration purposes."""
    import pandas as pd
    
    # Create sample VA data with multiple sites and classes
    sample_data = pd.DataFrame({
        "site": (["Site_A"] * 50 + ["Site_B"] * 50 + ["Site_C"] * 50 + ["Site_D"] * 50),
        "va34": ([1] * 20 + [2] * 15 + [3] * 10 + [4] * 5) * 4,  # Imbalanced classes
        "cod5": ([1] * 20 + [2] * 15 + [3] * 10 + [4] * 5) * 4,
        "symptom1": ([1, 0] * 100),
        "symptom2": ([0, 1] * 100),
        "symptom3": ([1, 1, 0] * 67)[:200],  # Ensure exact length
        "symptom4": ([0, 1, 1, 0] * 50),
        "symptom5": ([1] * 100 + [0] * 100)
    })
    
    return sample_data


def demonstrate_splitting_with_sample_data(data):
    """Demonstrate splitting with sample data."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Using sample data for demonstration")
    logger.info("=" * 60)
    
    logger.info(f"Sample data shape: {data.shape}")
    logger.info(f"Sites: {sorted(data['site'].unique())}")
    logger.info(f"Classes: {sorted(data['va34'].unique())}")
    
    return demonstrate_splitting_strategies(data)


def demonstrate_splitting_strategies(data):
    """Demonstrate different splitting strategies."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create a temporary CSV file for configuration validation
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("dummy,data\n1,2\n")
        temp_file.close()
        
        # Example 1: Simple train/test split
        logger.info("\n" + "=" * 60)
        logger.info("Example 1: Simple train/test split with stratification")
        logger.info("=" * 60)
        
        config_train_test = DataConfig(
            data_path=temp_file.name,  # Use temporary file for validation
            output_dir="results/baseline/",
            split_strategy="train_test",
            test_size=0.3,
            random_state=42,
            handle_small_classes="warn"
        )
        
        splitter_train_test = VADataSplitter(config_train_test)
        result_train_test = splitter_train_test.split_data(data)
        
        logger.info(f"Train/test split results:")
        logger.info(f"  Training samples: {len(result_train_test.train)}")
        logger.info(f"  Test samples: {len(result_train_test.test)}")
        logger.info(f"  Actual test ratio: {result_train_test.metadata['actual_test_ratio']:.3f}")
        logger.info(f"  Stratified: {result_train_test.metadata.get('stratified', 'N/A')}")
        
        # Show class distributions
        logger.info("\n  Training class distribution:")
        for cls, count in result_train_test.metadata['train_class_distribution'].items():
            logger.info(f"    Class {cls}: {count} samples")
        
        logger.info("\n  Test class distribution:")
        for cls, count in result_train_test.metadata['test_class_distribution'].items():
            logger.info(f"    Class {cls}: {count} samples")
        
        # Example 2: Cross-site split
        logger.info("\n" + "=" * 60)
        logger.info("Example 2: Cross-site split (train on some sites, test on others)")
        logger.info("=" * 60)
        
        config_cross_site = DataConfig(
            data_path=temp_file.name,
            output_dir="results/baseline/",
            split_strategy="cross_site",
            test_size=0.3,
            random_state=42,
            train_sites=None,  # Auto-select
            test_sites=None,   # Auto-select
            handle_small_classes="warn"
        )
        
        splitter_cross_site = VADataSplitter(config_cross_site)
        result_cross_site = splitter_cross_site.split_data(data)
        
        logger.info(f"Cross-site split results:")
        logger.info(f"  Training samples: {len(result_cross_site.train)}")
        logger.info(f"  Test samples: {len(result_cross_site.test)}")
        logger.info(f"  Training sites: {result_cross_site.metadata.get('train_sites', [])}")
        logger.info(f"  Test sites: {result_cross_site.metadata.get('test_sites', [])}")
        
        # Example 3: Stratified site split
        logger.info("\n" + "=" * 60)
        logger.info("Example 3: Stratified site split (maintain distribution within sites)")
        logger.info("=" * 60)
        
        config_stratified_site = DataConfig(
            data_path=temp_file.name,
            output_dir="results/baseline/",
            split_strategy="stratified_site",
            test_size=0.3,
            random_state=42,
            handle_small_classes="warn"
        )
        
        splitter_stratified_site = VADataSplitter(config_stratified_site)
        result_stratified_site = splitter_stratified_site.split_data(data)
        
        logger.info(f"Stratified site split results:")
        logger.info(f"  Training samples: {len(result_stratified_site.train)}")
        logger.info(f"  Test samples: {len(result_stratified_site.test)}")
        logger.info(f"  Sites processed: {result_stratified_site.metadata.get('sites_processed', 0)}")
        
        # Show site distributions
        train_sites = result_stratified_site.train['site'].value_counts()
        test_sites = result_stratified_site.test['site'].value_counts()
        
        logger.info("\n  Training site distribution:")
        for site, count in train_sites.items():
            logger.info(f"    {site}: {count} samples")
        
        logger.info("\n  Test site distribution:")
        for site, count in test_sites.items():
            logger.info(f"    {site}: {count} samples")
        
        # Example 4: Handling small classes
        logger.info("\n" + "=" * 60)
        logger.info("Example 4: Handling small classes (error mode)")
        logger.info("=" * 60)
        
        # Create data with single-instance class
        problematic_data = data.copy()
        problematic_data.loc[0, "va34"] = 999  # Single instance class
        
        config_error_mode = DataConfig(
            data_path=temp_file.name,
            output_dir="results/baseline/",
            split_strategy="train_test",
            test_size=0.3,
            random_state=42,
            handle_small_classes="error"
        )
        
        splitter_error_mode = VADataSplitter(config_error_mode)
        
        try:
            result_error = splitter_error_mode.split_data(problematic_data)
            logger.info("Unexpectedly succeeded with single-instance class")
        except ValueError as e:
            logger.info(f"Expected error caught: {str(e)}")
        
        # Example 5: Excluding small classes
        logger.info("\n" + "=" * 60)
        logger.info("Example 5: Excluding small classes")
        logger.info("=" * 60)
        
        config_exclude_mode = DataConfig(
            data_path=temp_file.name,
            output_dir="results/baseline/",
            split_strategy="train_test",
            test_size=0.3,
            random_state=42,
            handle_small_classes="exclude"
        )
        
        splitter_exclude_mode = VADataSplitter(config_exclude_mode)
        result_exclude = splitter_exclude_mode.split_data(problematic_data)
        
        logger.info(f"Exclude mode results:")
        logger.info(f"  Training samples: {len(result_exclude.train)}")
        logger.info(f"  Test samples: {len(result_exclude.test)}")
        logger.info(f"  Total samples: {len(result_exclude.train) + len(result_exclude.test)}")
        logger.info(f"  Original samples: {len(problematic_data)}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary of all splitting strategies")
        logger.info("=" * 60)
        
        strategies = [
            ("Train/Test", result_train_test),
            ("Cross-Site", result_cross_site),
            ("Stratified Site", result_stratified_site),
            ("Exclude Small Classes", result_exclude)
        ]
        
        for name, result in strategies:
            logger.info(f"{name:20} | Train: {len(result.train):4d} | Test: {len(result.test):4d} | "
                       f"Ratio: {result.metadata['actual_test_ratio']:.3f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Data splitting examples completed successfully!")
        logger.info("Check results/baseline/splits/ for saved split files")
        logger.info("=" * 60)
        
        # Clean up temporary file
        import os
        try:
            os.unlink(temp_file.name)
        except:
            pass  # Ignore cleanup errors
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during splitting demonstration: {str(e)}")
        logger.exception("Full traceback:")
        
        # Clean up temporary file
        import os
        try:
            os.unlink(temp_file.name)
        except:
            pass  # Ignore cleanup errors
        
        return 1


def demonstrate_configuration_validation():
    """Demonstrate configuration validation."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 60)
    logger.info("Configuration validation examples")
    logger.info("=" * 60)
    
    # Create temporary file for validation
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.write("dummy,data\n1,2\n")
    temp_file.close()
    
    try:
        # Example: Invalid test_size
        try:
            invalid_config = DataConfig(
                data_path=temp_file.name,
                test_size=0.8  # Invalid - too large
            )
            logger.info("Unexpectedly allowed invalid test_size")
        except ValueError as e:
            logger.info(f"Correctly caught invalid test_size: {str(e)}")
        
        # Example: Invalid split_strategy
        try:
            invalid_config = DataConfig(
                data_path=temp_file.name,
                split_strategy="invalid_strategy"
            )
            logger.info("Unexpectedly allowed invalid split_strategy")
        except ValueError as e:
            logger.info(f"Correctly caught invalid split_strategy: {str(e)}")
        
        # Example: Invalid handle_small_classes
        try:
            invalid_config = DataConfig(
                data_path=temp_file.name,
                handle_small_classes="invalid_handler"
            )
            logger.info("Unexpectedly allowed invalid handle_small_classes")
        except ValueError as e:
            logger.info(f"Correctly caught invalid handle_small_classes: {str(e)}")
    
    finally:
        # Clean up temporary file
        import os
        try:
            os.unlink(temp_file.name)
        except:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    sys.exit(main())