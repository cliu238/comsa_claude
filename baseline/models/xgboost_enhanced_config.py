"""Enhanced XGBoost configuration for better generalization.

This module provides an enhanced XGBoost configuration with stronger
regularization and conservative defaults to improve out-of-domain performance.
"""

from typing import Optional

from baseline.models.xgboost_config import XGBoostConfig


class XGBoostEnhancedConfig(XGBoostConfig):
    """Enhanced XGBoost configuration with optimized subsampling for VA data.
    
    This configuration is designed to improve both in-domain and out-domain
    performance by using moderate subsampling that creates diversity without
    losing signal. Key improvements:
    1. Optimized tree depth (max_depth=5) with better subsampling
    2. Moderate learning rate (0.06) balanced with subsampling noise
    3. Reduced L1/L2 regularization due to subsampling effects
    4. Optimized multi-level subsampling (0.7/0.5/0.55/0.65)
    5. Adjusted minimum child weight (15) for sparse data
    6. Lower gamma (0.5) with improved ensemble diversity
    """
    
    def __init__(
        self,
        # Tree complexity control
        max_depth: int = 5,  # Slightly deeper with improved subsampling
        min_child_weight: int = 15,  # Reduced from 20 due to subsampling
        gamma: float = 0.5,  # Reduced from 1.0 with better subsampling
        
        # Learning parameters
        learning_rate: float = 0.06,  # Slightly higher with subsampling noise
        n_estimators: int = 600,  # More trees for better ensemble
        
        # Sampling parameters - OPTIMIZED FOR VA DATA
        subsample: float = 0.7,  # Increased from 0.6 for better diversity
        colsample_bytree: float = 0.5,  # Kept moderate for ~100-110 features per tree
        colsample_bylevel: float = 0.55,  # Slightly increased for level diversity
        colsample_bynode: float = 0.65,  # Increased from 0.5 for node quality
        
        # Regularization - REDUCED DUE TO SUBSAMPLING
        reg_alpha: float = 5.0,  # Reduced from 10.0
        reg_lambda: float = 15.0,  # Reduced from 20.0
        
        # Other parameters from parent
        objective: str = "multi:softprob",
        n_jobs: int = -1,
        **kwargs
    ):
        """Initialize enhanced XGBoost configuration.
        
        Args:
            max_depth: Maximum tree depth (default: 5, balanced with subsampling)
            min_child_weight: Minimum child weight (default: 15, optimized for sparse data)
            gamma: Minimum loss reduction for split (default: 0.5, moderate pruning)
            learning_rate: Learning rate (default: 0.06, balanced with ensemble size)
            n_estimators: Number of trees (default: 600, larger ensemble)
            subsample: Sample ratio of training instances (default: 0.7, optimized)
            colsample_bytree: Sample ratio of columns per tree (default: 0.5, ~100 features)
            colsample_bylevel: Sample ratio of columns per level (default: 0.55, diversity)
            colsample_bynode: Sample ratio of columns per node (default: 0.65, quality)
            reg_alpha: L1 regularization (default: 5.0, reduced with subsampling)
            reg_lambda: L2 regularization (default: 15.0, reduced with subsampling)
            objective: XGBoost objective function
            n_jobs: Number of parallel threads
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            **kwargs
        )
        
        # Store additional parameters not in base config
        self._enhanced_params = {
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
        }
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for XGBoost.
        
        Returns:
            Dictionary of XGBoost parameters
        """
        params = super().to_dict()
        
        # Add enhanced parameters
        params.update(self._enhanced_params)
        
        return params
    
    @classmethod
    def conservative(cls) -> "XGBoostEnhancedConfig":
        """Create a very conservative configuration for extreme overfitting cases.
        
        Returns:
            XGBoostEnhancedConfig with very conservative parameters
        """
        return cls(
            max_depth=3,
            min_child_weight=50,
            gamma=5.0,
            learning_rate=0.01,
            n_estimators=1000,
            subsample=0.4,
            colsample_bytree=0.3,
            colsample_bylevel=0.3,
            colsample_bynode=0.3,
            reg_alpha=50.0,
            reg_lambda=100.0,
        )
    
    @classmethod
    def optimized_subsampling(cls) -> "XGBoostEnhancedConfig":
        """Create configuration with optimized subsampling for VA data.
        
        This configuration uses moderate subsampling to improve both
        in-domain and out-domain performance by creating ensemble
        diversity without losing too much signal.
        
        Returns:
            XGBoostEnhancedConfig with optimized subsampling parameters
        """
        return cls(
            max_depth=5,
            min_child_weight=15,
            gamma=0.5,
            learning_rate=0.06,
            n_estimators=600,
            subsample=0.7,
            colsample_bytree=0.5,
            colsample_bylevel=0.55,
            colsample_bynode=0.65,
            reg_alpha=5.0,
            reg_lambda=15.0,
        )