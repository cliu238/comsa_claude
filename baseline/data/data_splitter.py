"""VA data splitting module for site-based train/test splits.

This module provides clean, simple data splitting functionality for VA data,
supporting multiple split strategies with minimal complexity.
"""

import logging
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from baseline.config.data_config import DataConfig

logger = logging.getLogger(__name__)


class VADataSplitter:
    """Simple data splitter for VA data with site-based splitting strategies.
    
    This class provides three core splitting strategies:
    1. train_test: Standard train/test split across all sites
    2. cross_site: Train on specified sites, test on other sites
    3. stratified_site: Stratified split maintaining label distribution per site
    """

    def __init__(self, config: DataConfig):
        """Initialize the VA data splitter.

        Args:
            config: Configuration object with split settings
        """
        self.config = config
        logger.info(f"Initialized VADataSplitter with strategy: {config.split_strategy}")

    def split_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data according to the configured strategy.

        Args:
            data: Input DataFrame with site and label columns

        Returns:
            Dictionary with 'train' and 'test' DataFrames

        Raises:
            ValueError: If split strategy is invalid or data is insufficient
        """
        logger.info(f"Splitting data with strategy: {self.config.split_strategy}")
        logger.info(f"Input data shape: {data.shape}")
        
        # Validate required columns exist
        self._validate_columns(data)
        
        # Log data distribution
        self._log_data_distribution(data)
        
        # Route to appropriate split method
        if self.config.split_strategy == "train_test":
            return self._train_test_split(data)
        elif self.config.split_strategy == "cross_site":
            return self._cross_site_split(data)
        elif self.config.split_strategy == "stratified_site":
            return self._stratified_site_split(data)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.split_strategy}")

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Validate that required columns exist in the data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [self.config.site_column, self.config.label_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _log_data_distribution(self, data: pd.DataFrame) -> None:
        """Log data distribution by site and label."""
        site_counts = data[self.config.site_column].value_counts()
        label_counts = data[self.config.label_column].value_counts()
        
        logger.info(f"Site distribution: {site_counts.to_dict()}")
        logger.info(f"Label distribution: {label_counts.to_dict()}")

    def _train_test_split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Standard train/test split across all sites.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        logger.info("Performing standard train/test split")
        
        # Prepare features and target
        X = data.drop(columns=[self.config.label_column])
        y = data[self.config.label_column]
        
        # Perform stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
            logger.info("Used stratified split")
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
        
        # Combine features and target
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        return {"train": train_data, "test": test_data}

    def _cross_site_split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cross-site split using specified train and test sites.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with 'train' and 'test' DataFrames

        Raises:
            ValueError: If train/test sites are not specified or invalid
        """
        logger.info("Performing cross-site split")
        
        # Get available sites
        available_sites = set(data[self.config.site_column].unique())
        logger.info(f"Available sites: {available_sites}")
        
        # Determine train and test sites
        if self.config.train_sites and self.config.test_sites:
            train_sites = set(self.config.train_sites)
            test_sites = set(self.config.test_sites)
        elif self.config.train_sites:
            train_sites = set(self.config.train_sites)
            test_sites = available_sites - train_sites
        elif self.config.test_sites:
            test_sites = set(self.config.test_sites)
            train_sites = available_sites - test_sites
        else:
            raise ValueError("Must specify train_sites or test_sites for cross_site strategy")
        
        # Validate sites exist
        invalid_train = train_sites - available_sites
        invalid_test = test_sites - available_sites
        
        if invalid_train:
            raise ValueError(f"Invalid train sites: {invalid_train}")
        if invalid_test:
            raise ValueError(f"Invalid test sites: {invalid_test}")
        
        # Split data by sites
        train_data = data[data[self.config.site_column].isin(train_sites)].copy()
        test_data = data[data[self.config.site_column].isin(test_sites)].copy()
        
        if len(train_data) == 0:
            raise ValueError("No training data found for specified sites")
        if len(test_data) == 0:
            raise ValueError("No test data found for specified sites")
        
        logger.info(f"Train sites: {train_sites}, Test sites: {test_sites}")
        logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        return {"train": train_data, "test": test_data}

    def _stratified_site_split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Stratified split maintaining label distribution within each site.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        logger.info("Performing stratified site split")
        
        train_data_list = []
        test_data_list = []
        
        sites = data[self.config.site_column].unique()
        
        for site in sites:
            site_data = data[data[self.config.site_column] == site].copy()
            
            if len(site_data) < 2:
                logger.warning(f"Site {site} has insufficient data ({len(site_data)} samples), adding to test set")
                test_data_list.append(site_data)
                continue
            
            # Prepare features and target for this site
            X_site = site_data.drop(columns=[self.config.label_column])
            y_site = site_data[self.config.label_column]
            
            # Try stratified split for this site
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_site, y_site,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y_site
                )
                
                # Combine features and target
                site_train = pd.concat([X_train, y_train], axis=1)
                site_test = pd.concat([X_test, y_test], axis=1)
                
                train_data_list.append(site_train)
                test_data_list.append(site_test)
                
                logger.info(f"Site {site}: Train {len(site_train)}, Test {len(site_test)}")
                
            except ValueError as e:
                logger.warning(f"Site {site} stratified split failed: {e}. Using random split")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_site, y_site,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )
                
                # Combine features and target
                site_train = pd.concat([X_train, y_train], axis=1)
                site_test = pd.concat([X_test, y_test], axis=1)
                
                train_data_list.append(site_train)
                test_data_list.append(site_test)
                
                logger.info(f"Site {site}: Train {len(site_train)}, Test {len(site_test)}")
        
        # Combine all site splits
        train_data = pd.concat(train_data_list, ignore_index=True)
        test_data = pd.concat(test_data_list, ignore_index=True)
        
        logger.info(f"Total: Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        return {"train": train_data, "test": test_data}

    def get_split_statistics(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, int]]:
        """Get statistics for the data splits.

        Args:
            splits: Dictionary containing train/test DataFrames

        Returns:
            Dictionary with statistics for each split
        """
        stats = {}
        
        for split_name, split_data in splits.items():
            stats[split_name] = {
                "n_samples": len(split_data),
                "n_features": len(split_data.columns) - 1,  # Exclude target
                "n_sites": len(split_data[self.config.site_column].unique()) if self.config.site_column in split_data.columns else 0,
                "n_classes": len(split_data[self.config.label_column].unique()),
            }
        
        return stats