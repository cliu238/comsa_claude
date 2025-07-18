"""VA data splitting module for train/test and site-based splits.

This module implements the core data splitting functionality for Verbal Autopsy data,
supporting multiple splitting strategies with proper handling of imbalanced classes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from baseline.config.data_config import DataConfig
from baseline.utils.class_validator import ClassValidator
from baseline.utils.split_validator import SplitValidator

logger = logging.getLogger(__name__)


class SplitResult(BaseModel):
    """Result of a data splitting operation.
    
    Attributes:
        train: Training data DataFrame
        test: Test data DataFrame
        metadata: Dictionary containing split metadata
    """
    
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, Any]
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class VADataSplitter:
    """Data splitter for VA datasets with site-based and stratified splitting.
    
    This class provides multiple splitting strategies for VA data:
    - train_test: Simple train/test split with optional stratification
    - cross_site: Train on some sites, test on others
    - stratified_site: Stratified sampling within sites
    """
    
    def __init__(self, config: DataConfig):
        """Initialize the VA data splitter.
        
        Args:
            config: Configuration object with splitting settings
        """
        self.config = config
        self.class_validator = ClassValidator(config.min_samples_per_class)
        self.split_validator = SplitValidator(config)
        self.logger = logging.getLogger(__name__)
        
        # Setup output directories
        self._setup_output_dirs()
        
        self.logger.info(f"Initialized VADataSplitter with {config.split_strategy} strategy")
    
    def _setup_output_dirs(self) -> None:
        """Create necessary output directories."""
        output_path = Path(self.config.output_dir)
        splits_path = output_path / "splits"
        splits_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Split output directory created: {splits_path}")
    
    def split_data(self, data: pd.DataFrame) -> SplitResult:
        """Split data according to the configured strategy.
        
        Args:
            data: Input DataFrame to split
            
        Returns:
            SplitResult containing train/test data and metadata
            
        Raises:
            ValueError: If data is not suitable for splitting
        """
        # Validate input data
        self.split_validator.validate_data_for_splitting(data)
        
        # Log initial data statistics
        self._log_split_statistics(data, "input")
        
        # Dispatch to appropriate splitting method
        if self.config.split_strategy == "train_test":
            result = self._train_test_split(data)
        elif self.config.split_strategy == "cross_site":
            result = self._cross_site_split(data)
        elif self.config.split_strategy == "stratified_site":
            result = self._stratified_site_split(data)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.split_strategy}")
        
        # Log final split statistics
        self._log_split_statistics(result.train, "train")
        self._log_split_statistics(result.test, "test")
        
        # Save split results
        self._save_split_results(result)
        
        return result
    
    def _train_test_split(self, data: pd.DataFrame) -> SplitResult:
        """Perform simple train/test split with optional stratification.
        
        Args:
            data: Input DataFrame
            
        Returns:
            SplitResult with train/test split
        """
        self.logger.info("Performing train/test split")
        
        # Validate class distribution
        y = data[self.config.label_column]
        validation_result = self.class_validator.validate_class_distribution(y)
        
        # Handle validation results
        if not validation_result.is_valid:
            if self.config.handle_small_classes == "error":
                raise ValueError(f"Class validation failed: {validation_result.errors}")
            elif self.config.handle_small_classes == "warn":
                for warning in validation_result.warnings:
                    self.logger.warning(warning)
                for error in validation_result.errors:
                    self.logger.warning(f"Continuing despite error: {error}")
            elif self.config.handle_small_classes == "exclude":
                # Filter out problematic classes
                y_filtered = self.class_validator.get_stratifiable_classes(y)
                data = data[data[self.config.label_column].isin(y_filtered)]
                y = data[self.config.label_column]
                self.logger.info(f"Excluded small classes, {len(data)} samples remaining")
        
        # Attempt stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                data.drop(columns=[self.config.label_column]),
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
            stratified = True
            self.logger.info("Stratified split successful")
            
        except ValueError as e:
            if "least populated class" in str(e):
                self.logger.warning("Falling back to non-stratified split due to small classes")
                X_train, X_test, y_train, y_test = train_test_split(
                    data.drop(columns=[self.config.label_column]),
                    y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=None
                )
                stratified = False
            else:
                raise
        
        # Reconstruct DataFrames
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Generate metadata
        metadata = self._generate_split_metadata(
            train_data, test_data, 
            extra_info={"stratified": stratified}
        )
        
        return SplitResult(train=train_data, test=test_data, metadata=metadata)
    
    def _cross_site_split(self, data: pd.DataFrame) -> SplitResult:
        """Perform cross-site split (train on some sites, test on others).
        
        Args:
            data: Input DataFrame
            
        Returns:
            SplitResult with cross-site split
        """
        self.logger.info("Performing cross-site split")
        
        # Get available sites
        available_sites = self.split_validator.get_available_sites(data)
        
        # Determine train/test sites
        if self.config.train_sites and self.config.test_sites:
            train_sites = self.config.train_sites
            test_sites = self.config.test_sites
        else:
            train_sites, test_sites = self.split_validator.suggest_site_split(data)
        
        # Split data by sites
        train_data = data[data[self.config.site_column].isin(train_sites)].copy()
        test_data = data[data[self.config.site_column].isin(test_sites)].copy()
        
        # Validate split results
        if len(train_data) == 0:
            raise ValueError(f"No training data found for sites: {train_sites}")
        if len(test_data) == 0:
            raise ValueError(f"No test data found for sites: {test_sites}")
        
        # Generate metadata
        metadata = self._generate_split_metadata(
            train_data, test_data,
            extra_info={
                "train_sites": train_sites,
                "test_sites": test_sites,
                "total_sites": len(available_sites)
            }
        )
        
        return SplitResult(train=train_data, test=test_data, metadata=metadata)
    
    def _stratified_site_split(self, data: pd.DataFrame) -> SplitResult:
        """Perform stratified split maintaining class distribution within sites.
        
        Args:
            data: Input DataFrame
            
        Returns:
            SplitResult with stratified site split
        """
        self.logger.info("Performing stratified site split")
        
        # Get available sites
        available_sites = self.split_validator.get_available_sites(data)
        
        train_parts = []
        test_parts = []
        
        # Split each site individually
        for site in available_sites:
            site_data = data[data[self.config.site_column] == site].copy()
            
            if len(site_data) < 2:
                self.logger.warning(f"Site {site} has only {len(site_data)} samples, "
                                   f"adding to training set")
                train_parts.append(site_data)
                continue
            
            # Validate site classes
            site_y = site_data[self.config.label_column]
            
            try:
                # Attempt stratified split for this site
                site_X_train, site_X_test, site_y_train, site_y_test = train_test_split(
                    site_data.drop(columns=[self.config.label_column]),
                    site_y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=site_y
                )
                
                # Reconstruct site DataFrames
                site_train = pd.concat([site_X_train, site_y_train], axis=1)
                site_test = pd.concat([site_X_test, site_y_test], axis=1)
                
                train_parts.append(site_train)
                test_parts.append(site_test)
                
            except ValueError as e:
                if "least populated class" in str(e):
                    self.logger.warning(f"Site {site} has single-instance classes, "
                                       f"using random split")
                    # Fall back to random split for this site
                    site_X_train, site_X_test, site_y_train, site_y_test = train_test_split(
                        site_data.drop(columns=[self.config.label_column]),
                        site_y,
                        test_size=self.config.test_size,
                        random_state=self.config.random_state,
                        stratify=None
                    )
                    
                    site_train = pd.concat([site_X_train, site_y_train], axis=1)
                    site_test = pd.concat([site_X_test, site_y_test], axis=1)
                    
                    train_parts.append(site_train)
                    test_parts.append(site_test)
                else:
                    raise
        
        # Combine all site splits
        train_data = pd.concat(train_parts, ignore_index=True)
        test_data = pd.concat(test_parts, ignore_index=True)
        
        # Generate metadata
        metadata = self._generate_split_metadata(
            train_data, test_data,
            extra_info={
                "sites_processed": len(available_sites),
                "sites_with_stratification": len([p for p in train_parts if len(p) > 0])
            }
        )
        
        return SplitResult(train=train_data, test=test_data, metadata=metadata)
    
    def _generate_split_metadata(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                                extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate metadata for split results.
        
        Args:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            extra_info: Additional metadata to include
            
        Returns:
            Dictionary containing split metadata
        """
        metadata = {
            "split_strategy": self.config.split_strategy,
            "test_size": self.config.test_size,
            "random_state": self.config.random_state,
            "split_timestamp": datetime.now().isoformat(),
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "total_samples": len(train_data) + len(test_data),
            "actual_test_ratio": len(test_data) / (len(train_data) + len(test_data)),
            "train_class_distribution": self._get_class_distribution(train_data),
            "test_class_distribution": self._get_class_distribution(test_data),
            "config": self.config.model_dump()
        }
        
        if extra_info:
            metadata.update(extra_info)
        
        return metadata
    
    def _get_class_distribution(self, data: pd.DataFrame) -> Dict[str, int]:
        """Get class distribution from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary mapping class labels to counts
        """
        if self.config.label_column not in data.columns:
            return {}
        
        class_counts = data[self.config.label_column].value_counts()
        return {str(k): int(v) for k, v in class_counts.items()}
    
    def _log_split_statistics(self, data: pd.DataFrame, stage: str) -> None:
        """Log statistics about the split data.
        
        Args:
            data: DataFrame to analyze
            stage: Description of the split stage
        """
        stats = {
            "stage": stage,
            "n_samples": len(data),
            "n_features": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        if self.config.label_column in data.columns:
            stats["n_classes"] = int(data[self.config.label_column].nunique())
            stats["class_distribution"] = self._get_class_distribution(data)
        
        if self.config.site_column in data.columns:
            stats["n_sites"] = int(data[self.config.site_column].nunique())
        
        self.logger.info(f"Split statistics at {stage}: {json.dumps(stats, indent=2)}")
    
    def _save_split_results(self, result: SplitResult) -> None:
        """Save split results to files.
        
        Args:
            result: Split result to save
        """
        # Generate output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / "splits" / f"{self.config.split_strategy}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train data
        train_path = output_dir / "train.csv"
        result.train.to_csv(train_path, index=False)
        self.logger.info(f"Saved training data: {train_path}")
        
        # Save test data
        test_path = output_dir / "test.csv"
        result.test.to_csv(test_path, index=False)
        self.logger.info(f"Saved test data: {test_path}")
        
        # Save metadata
        metadata_path = output_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(result.metadata, f, indent=2, default=str)
        self.logger.info(f"Saved split metadata: {metadata_path}")
        
        # Save summary
        summary_path = output_dir / "split_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Data Split Summary\n")
            f.write("==================\n\n")
            f.write(f"Strategy: {self.config.split_strategy}\n")
            f.write(f"Timestamp: {result.metadata['split_timestamp']}\n")
            f.write(f"Train samples: {result.metadata['train_samples']}\n")
            f.write(f"Test samples: {result.metadata['test_samples']}\n")
            f.write(f"Test ratio: {result.metadata['actual_test_ratio']:.3f}\n")
            f.write("\nTrain class distribution:\n")
            for cls, count in result.metadata['train_class_distribution'].items():
                f.write(f"  {cls}: {count}\n")
            f.write("\nTest class distribution:\n")
            for cls, count in result.metadata['test_class_distribution'].items():
                f.write(f"  {cls}: {count}\n")
        
        self.logger.info(f"Split results saved to: {output_dir}")