"""VA data loading and preprocessing module.

This module implements the core data processing pipeline for Verbal Autopsy data,
supporting both standard ML preprocessing and OpenVA encoding for InSilicoVA.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

# CRITICAL: Import order matters - phmrc_plugins must be imported before PHMRCData
import va_data.phmrc_plugins  # noqa: F401 # Ensures transform registration
from va_data.phmrc_data import PHMRCData

from baseline.config.data_config import DataConfig

logger = logging.getLogger(__name__)


class VADataProcessor:
    """Processor for VA data with support for multiple encoding formats.

    This class handles loading, validation, and preprocessing of PHMRC data,
    providing outputs suitable for both standard ML algorithms and InSilicoVA.
    """

    def __init__(self, config: DataConfig):
        """Initialize the VA data processor.

        Args:
            config: Configuration object with processing settings
        """
        self.config = config
        self._setup_output_dirs()
        logger.info(f"Initialized VADataProcessor with config: {config}")

    def _setup_output_dirs(self) -> None:
        """Create necessary output directories."""
        output_path = Path(self.config.output_dir)
        processed_data_path = output_path / "processed_data"
        processed_data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directories created at: {processed_data_path}")

    def load_and_process(self) -> pd.DataFrame:
        """Load and process PHMRC data according to configuration.

        Returns:
            Processed DataFrame ready for ML models or InSilicoVA

        Raises:
            ValueError: If data validation or processing fails
        """
        # Extract dataset name from file path
        dataset_name = self._extract_dataset_name()

        # Stage 1: Load and validate data
        logger.info(f"Loading PHMRC data from {self.config.data_path}")
        try:
            va_data = PHMRCData(self.config.data_path)
            df = va_data.validate(nullable=False, drop=self.config.drop_columns)
            logger.info(f"Validated {len(df)} records with {len(df.columns)} columns")

            # Log validation statistics
            self._log_data_statistics(df, "post-validation")

        except Exception as e:
            logger.error(f"Failed to load/validate data: {str(e)}")
            raise ValueError(f"Data validation failed: {str(e)}")

        # Stage 2: Apply OpenVA transformation
        logger.info("Applying OpenVA transformation")
        df = va_data.xform("openva")
        logger.info(f"OpenVA transformation complete: {len(df)} records")

        # Stage 3: Apply encoding based on configuration
        if self.config.openva_encoding:
            logger.info("Applying OpenVA encoding for InSilicoVA")
            df = self._apply_openva_encoding(df)
            encoding_type = "openva"
        else:
            logger.info("Converting categorical to numeric for ML models")
            df = self._convert_categorical_to_numeric(df)
            encoding_type = "numeric"

        # Log final statistics
        self._log_data_statistics(df, f"post-{encoding_type}-encoding")

        # Stage 4: Save processed data
        output_path = self._save_results(df, dataset_name, encoding_type)
        logger.info(f"Processed data saved to: {output_path}")

        return df

    def _extract_dataset_name(self) -> str:
        """Extract dataset name from file path.

        Returns:
            Dataset name ('adult', 'child', or 'neonate')
        """
        file_name = Path(self.config.data_path).stem.lower()
        if "adult" in file_name:
            return "adult"
        elif "child" in file_name:
            return "child"
        elif "neonate" in file_name:
            return "neonate"
        else:
            logger.warning(
                f"Could not determine dataset type from filename: {file_name}"
            )
            return "unknown"

    def _get_label_equivalent_columns(self, exclude_current_target: bool = True) -> list[str]:
        """Get all columns that contain cause-of-death information.
        
        Args:
            exclude_current_target: Whether to exclude the current target column from the list
        
        Returns:
            List of column names that must be excluded from features to prevent data leakage.
        """
        # All label-equivalent columns
        all_label_columns = [
            "site",  # Site information (kept for stratification)
            "module",  # Module type
            "gs_code34", "gs_text34", "va34",  # 34-cause classification
            "gs_code46", "gs_text46", "va46",  # 46-cause classification  
            "gs_code55", "gs_text55", "va55",  # 55-cause classification
            "gs_comorbid1", "gs_comorbid2",  # Comorbidity information
            "gs_level",  # Gold standard level
            "cod5",  # 5-cause grouping
            "newid"  # ID column
        ]
        
        # If requested, keep the current target column for splitting/processing
        if exclude_current_target and hasattr(self, 'config') and self.config.label_column:
            # Remove the current target column from the exclusion list
            target_col = self.config.label_column
            if target_col in all_label_columns:
                all_label_columns = [col for col in all_label_columns if col != target_col]
        
        return all_label_columns

    def _apply_openva_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply OpenVA encoding for InSilicoVA compatibility.

        Args:
            df: DataFrame with OpenVA transformed data

        Returns:
            DataFrame with OpenVA encoding applied
        """
        # GOTCHA: Use exact mapping for InSilicoVA
        repldict = {1: "Y", 0: "", 2: "."}

        # Apply encoding to all feature columns (exclude ALL label-equivalent columns)
        # CRITICAL: Exclude ALL columns that contain cause-of-death information to prevent data leakage
        # Keep the current target column for splitting/processing, but don't encode it
        label_equivalent_columns = self._get_label_equivalent_columns(exclude_current_target=False)
        cols = df.columns.difference(label_equivalent_columns)
        df = df.astype({c: object for c in cols})

        # Handle pandas warning about silent downcasting
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace({c: repldict for c in cols})

        logger.info(f"OpenVA encoding applied to {len(cols)} feature columns")

        # Verify encoding worked correctly
        sample_col = df[cols[0]].dropna()
        unique_vals = sample_col.unique()
        logger.debug(f"Sample unique values after encoding: {unique_vals[:5]}")

        return df

    def _convert_categorical_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical data to numeric format for ML models.

        Args:
            df: DataFrame with categorical data

        Returns:
            DataFrame with numeric features
        """
        data = df.copy()

        # Get feature columns (exclude ALL label-equivalent columns)
        # Keep the current target column for splitting/processing
        label_equivalent_columns = self._get_label_equivalent_columns(exclude_current_target=True)
        feature_cols = [
            col for col in data.columns if col not in label_equivalent_columns
        ]
        converted_count = 0

        for col in feature_cols:
            if data[col].dtype == "object":
                # Convert to string to ensure consistent handling
                data[col] = data[col].astype(str)

                # Common Yes/No mappings
                with pd.option_context("future.no_silent_downcasting", True):
                    data[col] = data[col].replace(
                        {
                            "Yes": 1,
                            "No": 0,
                            "Y": 1,
                            "N": 0,
                            "yes": 1,
                            "no": 0,
                            "y": 1,
                            "n": 0,
                            "1": 1,
                            "0": 0,
                            "True": 1,
                            "False": 0,
                            "true": 1,
                            "false": 0,
                        }
                    )

                    # Handle missing values and empty strings
                    data[col] = data[col].replace(
                        {
                            "": 0,
                            "nan": 0,
                            "NaN": 0,
                            "None": 0,
                            "NA": 0,
                            "na": 0,
                            ".": 0,
                            "null": 0,
                        }
                    )

                # Convert to numeric, coercing any remaining strings to NaN
                data[col] = pd.to_numeric(data[col], errors="coerce")

                # Fill remaining NaN values with 0 (conservative approach)
                data[col] = data[col].fillna(0)

                # Ensure integer type for binary features
                if data[col].nunique() <= 10:  # Likely categorical
                    data[col] = data[col].astype(int)

                converted_count += 1

        logger.info(f"Converted {converted_count} categorical columns to numeric")

        # Handle missing values in target columns
        if "va34" in data.columns:
            data = data.dropna(subset=["va34"])
            logger.info(
                f"Removed rows with missing va34 values. Final record count: {len(data)}"
            )

        return data

    def _log_data_statistics(self, df: pd.DataFrame, stage: str) -> None:
        """Log statistics about the dataset.

        Args:
            df: DataFrame to analyze
            stage: Description of the processing stage
        """
        stats = {
            "stage": stage,
            "n_samples": len(df),
            "n_features": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        if "va34" in df.columns:
            stats["n_classes"] = int(df["va34"].nunique())
            # Handle case where va34 might be encoded as strings
            class_dist = df["va34"].value_counts().to_dict()
            try:
                stats["class_distribution"] = {int(k): int(v) for k, v in class_dist.items()}
            except ValueError:
                # If conversion fails (e.g., encoded as "Y", "", "."), keep as strings
                stats["class_distribution"] = {str(k): int(v) for k, v in class_dist.items()}

        logger.info(
            f"Data statistics at {stage}: {json.dumps(stats, indent=2, default=str)}"
        )

    def _save_results(
        self, df: pd.DataFrame, dataset_name: str, encoding_type: str
    ) -> Path:
        """Save processed data and metadata.

        Args:
            df: Processed DataFrame
            dataset_name: Name of the dataset
            encoding_type: Type of encoding applied

        Returns:
            Path to the saved data file
        """
        # Generate output path
        output_path = self.config.get_output_path(dataset_name, encoding_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "encoding_type": encoding_type,
            "processing_timestamp": datetime.now().isoformat(),
            "config": self.config.model_dump(),
            "data_shape": df.shape,
            "columns": list(df.columns),
            "va34_distribution": (
                {int(k): int(v) for k, v in df["va34"].value_counts().to_dict().items()} 
                if "va34" in df.columns else None
            ),
        }

        metadata_path = output_path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")

        return output_path

    def process_with_site_stratification(self) -> Dict[str, pd.DataFrame]:
        """Process data with site-based stratification.

        Returns:
            Dictionary mapping site names to processed DataFrames
        """
        if not self.config.stratify_by_site:
            logger.warning("Site stratification requested but not enabled in config")
            return {"all": self.load_and_process()}

        # Load initial data to get sites
        logger.info("Loading data for site stratification")
        va_data = PHMRCData(self.config.data_path)
        df = va_data.validate(nullable=False, drop=self.config.drop_columns)

        if "site" not in df.columns:
            logger.warning(
                "No 'site' column found in data, processing without stratification"
            )
            return {"all": self.load_and_process()}

        sites = df["site"].unique()
        logger.info(f"Found {len(sites)} sites: {sites}")

        results = {}
        for site in sites:
            logger.info(f"Processing site: {site}")
            site_df = df[df["site"] == site].copy()

            # Apply transformations - note: xform works on the whole dataset, not subsets
            # For site stratification, we need to process differently
            # Skip xform for now as it doesn't support subset processing

            if self.config.openva_encoding:
                site_df = self._apply_openva_encoding(site_df)
            else:
                site_df = self._convert_categorical_to_numeric(site_df)

            results[site] = site_df
            logger.info(f"Processed {len(site_df)} records for site {site}")

        return results
