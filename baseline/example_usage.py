#!/usr/bin/env python
"""Example usage of the VA data processing pipeline.

This script demonstrates how to use the VADataProcessor to load and process
PHMRC data for both standard ML algorithms and InSilicoVA.
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


def main() -> int:
    """Run example VA data processing pipeline."""
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
        return 1

    try:
        # Example 1: Process data for standard ML algorithms (numeric encoding)
        logger.info("=" * 60)
        logger.info("Example 1: Processing data for ML algorithms (numeric)")
        logger.info("=" * 60)

        config_ml = DataConfig(
            data_path=data_file,
            output_dir="results/baseline/",
            openva_encoding=False,  # Numeric encoding for ML
            stratify_by_site=False,  # Process all sites together
        )
        config_ml.setup_logging("INFO")

        processor_ml = VADataProcessor(config_ml)
        df_ml = processor_ml.load_and_process()

        logger.info(f"Processed {len(df_ml)} records for ML")
        logger.info(f"Shape: {df_ml.shape}")
        logger.info(f"Columns: {list(df_ml.columns)[:10]}...")  # Show first 10 columns

        # Show sample of processed data
        logger.info("\nSample of processed data (first 5 rows, first 5 columns):")
        print(df_ml.iloc[:5, :5])

        # Example 2: Process data for InSilicoVA (OpenVA encoding)
        logger.info("\n" + "=" * 60)
        logger.info("Example 2: Processing data for InSilicoVA (OpenVA encoding)")
        logger.info("=" * 60)

        config_insilico = DataConfig(
            data_path=data_file,
            output_dir="results/baseline/",
            openva_encoding=True,  # OpenVA encoding for InSilicoVA
            stratify_by_site=False,
        )

        processor_insilico = VADataProcessor(config_insilico)
        df_insilico = processor_insilico.load_and_process()

        logger.info(f"Processed {len(df_insilico)} records for InSilicoVA")
        logger.info(f"Shape: {df_insilico.shape}")

        # Show sample of encoded data
        logger.info("\nSample of OpenVA encoded data (first 5 rows, columns 3-7):")
        print(df_insilico.iloc[:5, 3:8])

        # Example 3: Process data with site stratification (simplified)
        # Note: Site stratification with xform transformation requires 
        # processing the entire dataset, so we'll skip this example for now
        logger.info("\n" + "=" * 60)
        logger.info("Example 3: Site stratification (skipped - see note)")
        logger.info("=" * 60)
        logger.info("Site stratification requires custom handling of xform transformations.")

        # Show class distribution
        logger.info("\n" + "=" * 60)
        logger.info("Target (va34) distribution in ML processed data:")
        logger.info("=" * 60)
        if "va34" in df_ml.columns:
            class_dist = df_ml["va34"].value_counts().head(10)
            print(class_dist)

        logger.info("\n" + "=" * 60)
        logger.info("Processing complete! Check results/baseline/processed_data/")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
