"""Adaptive XGBoost configuration based on data characteristics.

This module provides XGBoost configurations that adapt to the characteristics
of the training data, such as site size and class distribution.
"""

import logging
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig


logger = logging.getLogger(__name__)


class XGBoostAdaptiveConfig(XGBoostEnhancedConfig):
    """Adaptive XGBoost configuration that adjusts based on data characteristics.
    
    This configuration automatically adjusts subsampling and regularization
    parameters based on:
    - Training data size
    - Number of unique classes
    - Class imbalance ratio
    - Site characteristics (if available)
    """
    
    @classmethod
    def from_data_characteristics(
        cls,
        n_samples: int,
        n_classes: int,
        class_imbalance_ratio: float,
        site_name: Optional[str] = None,
        **kwargs
    ) -> "XGBoostAdaptiveConfig":
        """Create adaptive configuration based on data characteristics.
        
        Args:
            n_samples: Number of training samples
            n_classes: Number of unique classes
            class_imbalance_ratio: Ratio of largest to smallest class
            site_name: Name of the site (for site-specific adjustments)
            **kwargs: Additional parameters to override
            
        Returns:
            Adaptive XGBoost configuration
        """
        # Base parameters
        params = {
            "n_estimators": 500,
            "objective": "multi:softprob",
            "n_jobs": -1,
        }
        
        # Adapt tree depth based on data size and complexity
        if n_samples < 1000:
            # Small dataset - very shallow trees
            params["max_depth"] = 3
            params["min_child_weight"] = max(50, n_samples // 20)
        elif n_samples < 5000:
            # Medium dataset
            params["max_depth"] = 4
            params["min_child_weight"] = 30
        else:
            # Large dataset - can afford slightly deeper trees
            params["max_depth"] = 5
            params["min_child_weight"] = 20
            
        # Adapt learning rate based on dataset size
        if n_samples < 1000:
            params["learning_rate"] = 0.01
            params["n_estimators"] = 1000
        elif n_samples < 5000:
            params["learning_rate"] = 0.05
        else:
            params["learning_rate"] = 0.1
            
        # Adapt subsampling based on dataset size and imbalance
        if n_samples < 1000:
            # Small data - less aggressive subsampling
            params["subsample"] = 0.8
            params["colsample_bytree"] = 0.7
            params["colsample_bylevel"] = 0.7
            params["colsample_bynode"] = 0.7
        elif class_imbalance_ratio > 10:
            # High imbalance - moderate subsampling
            params["subsample"] = 0.6
            params["colsample_bytree"] = 0.5
            params["colsample_bylevel"] = 0.5
            params["colsample_bynode"] = 0.5
        else:
            # Normal case - standard enhanced subsampling
            params["subsample"] = 0.6
            params["colsample_bytree"] = 0.5
            params["colsample_bylevel"] = 0.5
            params["colsample_bynode"] = 0.5
            
        # Adapt regularization based on number of classes and samples
        samples_per_class = n_samples / n_classes
        if samples_per_class < 50:
            # Few samples per class - strong regularization
            params["reg_alpha"] = 50.0
            params["reg_lambda"] = 100.0
            params["gamma"] = 5.0
        elif samples_per_class < 200:
            # Moderate samples per class
            params["reg_alpha"] = 20.0
            params["reg_lambda"] = 50.0
            params["gamma"] = 2.0
        else:
            # Many samples per class
            params["reg_alpha"] = 10.0
            params["reg_lambda"] = 20.0
            params["gamma"] = 1.0
            
        # Site-specific adjustments (based on known challenging sites)
        if site_name:
            site_adjustments = cls._get_site_specific_adjustments(site_name)
            params.update(site_adjustments)
            
        # Apply any user overrides
        params.update(kwargs)
        
        logger.info(
            f"Created adaptive config for {n_samples} samples, {n_classes} classes, "
            f"imbalance ratio {class_imbalance_ratio:.2f}"
        )
        
        return cls(**params)
    
    @staticmethod
    def _get_site_specific_adjustments(site_name: str) -> Dict[str, Any]:
        """Get site-specific parameter adjustments.
        
        Based on empirical observations of site characteristics.
        
        Args:
            site_name: Name of the site
            
        Returns:
            Dictionary of parameter adjustments
        """
        # Known challenging sites that need extra regularization
        challenging_sites = {
            "Pemba": {
                # Pemba has unique symptom patterns - needs very conservative params
                "max_depth": 3,
                "subsample": 0.4,
                "colsample_bytree": 0.3,
                "reg_alpha": 100.0,
                "reg_lambda": 200.0,
            },
            "Bohol": {
                # Bohol also shows poor transfer - moderate adjustments
                "max_depth": 3,
                "subsample": 0.5,
                "reg_alpha": 50.0,
            },
        }
        
        # High-performing sites that can use less conservative params
        robust_sites = {
            "Mexico": {
                # Mexico generalizes well - can use less regularization
                "max_depth": 5,
                "subsample": 0.7,
                "reg_alpha": 5.0,
                "reg_lambda": 10.0,
            },
            "AP": {
                # AP/UP sites perform well
                "subsample": 0.7,
                "reg_alpha": 10.0,
            },
            "UP": {
                "subsample": 0.7,
                "reg_alpha": 10.0,
            },
        }
        
        if site_name in challenging_sites:
            logger.info(f"Applying challenging site adjustments for {site_name}")
            return challenging_sites[site_name]
        elif site_name in robust_sites:
            logger.info(f"Applying robust site adjustments for {site_name}")
            return robust_sites[site_name]
        else:
            return {}
    
    @classmethod
    def analyze_data_characteristics(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        site_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze data characteristics to inform configuration.
        
        Args:
            X: Feature data
            y: Label data
            site_col: Column name containing site information
            
        Returns:
            Dictionary of data characteristics
        """
        # Basic statistics
        n_samples = len(y)
        n_classes = y.nunique()
        
        # Class distribution
        class_counts = y.value_counts()
        class_imbalance_ratio = class_counts.max() / class_counts.min()
        
        # Feature statistics
        n_features = X.shape[1]
        
        # Site information if available
        site_name = None
        if site_col and site_col in X.columns:
            site_counts = X[site_col].value_counts()
            if len(site_counts) == 1:
                site_name = site_counts.index[0]
        
        characteristics = {
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_features": n_features,
            "class_imbalance_ratio": class_imbalance_ratio,
            "site_name": site_name,
            "samples_per_class_mean": n_samples / n_classes,
            "samples_per_class_min": class_counts.min(),
            "samples_per_class_max": class_counts.max(),
        }
        
        logger.info(f"Data characteristics: {characteristics}")
        
        return characteristics