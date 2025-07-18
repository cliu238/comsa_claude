"""Split configuration validation utilities for VA data splitting.

This module provides utilities for validating split configurations and
ensuring that the data is suitable for the requested splitting strategy.
"""

import logging
from typing import List, Set

import pandas as pd

from baseline.config.data_config import DataConfig

logger = logging.getLogger(__name__)


class SplitValidator:
    """Utility for validating split configurations and data suitability.
    
    This class provides methods to validate split configurations and ensure
    that the data contains the necessary columns and sites for the requested
    splitting strategy.
    """
    
    def __init__(self, config: DataConfig):
        """Initialize the split validator.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_data_for_splitting(self, data: pd.DataFrame) -> None:
        """Validate that data is suitable for the configured splitting strategy.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If data is not suitable for splitting
        """
        # Check for required columns
        self._validate_required_columns(data)
        
        # Check for sufficient data
        self._validate_sufficient_data(data)
        
        # Strategy-specific validation
        if self.config.split_strategy in ["cross_site", "stratified_site"]:
            self._validate_site_data(data)
        
        self.logger.info(f"Data validation passed for {self.config.split_strategy} strategy")
    
    def _validate_required_columns(self, data: pd.DataFrame) -> None:
        """Validate that required columns are present.
        
        Args:
            data: Input DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [self.config.label_column]
        
        # Add site column if needed for site-based strategies
        if self.config.split_strategy in ["cross_site", "stratified_site"]:
            required_columns.append(self.config.site_column)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.debug(f"Required columns validated: {required_columns}")
    
    def _validate_sufficient_data(self, data: pd.DataFrame) -> None:
        """Validate that there is sufficient data for splitting.
        
        Args:
            data: Input DataFrame
            
        Raises:
            ValueError: If insufficient data for splitting
        """
        min_samples = max(10, int(1 / self.config.test_size) + 1)
        
        if len(data) < min_samples:
            raise ValueError(
                f"Insufficient data for splitting: {len(data)} samples, "
                f"need at least {min_samples} for test_size={self.config.test_size}"
            )
        
        self.logger.debug(f"Sufficient data validated: {len(data)} samples")
    
    def _validate_site_data(self, data: pd.DataFrame) -> None:
        """Validate site-specific data for site-based splitting strategies.
        
        Args:
            data: Input DataFrame
            
        Raises:
            ValueError: If site data is insufficient
        """
        if self.config.site_column not in data.columns:
            raise ValueError(f"Site column '{self.config.site_column}' not found in data")
        
        # Check for sites with missing values
        site_missing = data[self.config.site_column].isnull().sum()
        if site_missing > 0:
            self.logger.warning(f"Found {site_missing} rows with missing site information")
        
        # Get unique sites
        available_sites = set(data[self.config.site_column].dropna().unique())
        
        if len(available_sites) < 2:
            raise ValueError(
                f"Need at least 2 sites for site-based splitting, "
                f"found {len(available_sites)}: {available_sites}"
            )
        
        # Validate specified sites for cross_site strategy
        if self.config.split_strategy == "cross_site":
            self._validate_cross_site_configuration(available_sites)
        
        # Check site sample sizes
        self._validate_site_sample_sizes(data, available_sites)
        
        self.logger.info(f"Site validation passed: {len(available_sites)} sites available")
    
    def _validate_cross_site_configuration(self, available_sites: Set[str]) -> None:
        """Validate cross-site splitting configuration.
        
        Args:
            available_sites: Set of available site names
            
        Raises:
            ValueError: If cross-site configuration is invalid
        """
        # Check if specific sites are configured
        if self.config.train_sites or self.config.test_sites:
            # Validate train sites
            if self.config.train_sites:
                missing_train = set(self.config.train_sites) - available_sites
                if missing_train:
                    raise ValueError(f"Train sites not found in data: {missing_train}")
            
            # Validate test sites
            if self.config.test_sites:
                missing_test = set(self.config.test_sites) - available_sites
                if missing_test:
                    raise ValueError(f"Test sites not found in data: {missing_test}")
            
            # Check for overlap
            if (self.config.train_sites and self.config.test_sites and
                set(self.config.train_sites) & set(self.config.test_sites)):
                overlap = set(self.config.train_sites) & set(self.config.test_sites)
                raise ValueError(f"Train and test sites overlap: {overlap}")
        
        self.logger.debug("Cross-site configuration validated")
    
    def _validate_site_sample_sizes(self, data: pd.DataFrame, available_sites: Set[str]) -> None:
        """Validate that sites have sufficient sample sizes.
        
        Args:
            data: Input DataFrame
            available_sites: Set of available site names
        """
        min_samples_per_site = 5  # Minimum samples per site
        
        site_counts = data[self.config.site_column].value_counts()
        small_sites = site_counts[site_counts < min_samples_per_site]
        
        if len(small_sites) > 0:
            small_site_info = {str(k): int(v) for k, v in small_sites.items()}
            self.logger.warning(
                f"Sites with fewer than {min_samples_per_site} samples: {small_site_info}"
            )
        
        self.logger.debug(f"Site sample sizes validated: {len(site_counts)} sites")
    
    def get_available_sites(self, data: pd.DataFrame) -> List[str]:
        """Get list of available sites in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of available site names
        """
        if self.config.site_column not in data.columns:
            return []
        
        sites = data[self.config.site_column].dropna().unique().tolist()
        sites.sort()  # Sort for consistent ordering
        
        return [str(site) for site in sites]  # Ensure string type
    
    def suggest_site_split(self, data: pd.DataFrame) -> tuple[List[str], List[str]]:
        """Suggest train/test site split based on available data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (train_sites, test_sites)
        """
        available_sites = self.get_available_sites(data)
        
        if len(available_sites) < 2:
            return [], []
        
        # Calculate number of test sites based on test_size
        n_test_sites = max(1, int(len(available_sites) * self.config.test_size))
        n_train_sites = len(available_sites) - n_test_sites
        
        # For simplicity, take the first n sites for training
        train_sites = available_sites[:n_train_sites]
        test_sites = available_sites[n_train_sites:]
        
        self.logger.info(f"Suggested split: {len(train_sites)} train sites, "
                        f"{len(test_sites)} test sites")
        
        return train_sites, test_sites