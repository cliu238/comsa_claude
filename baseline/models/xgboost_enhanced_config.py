"""Enhanced XGBoost configuration for better generalization.

This module provides an enhanced XGBoost configuration with stronger
regularization and conservative defaults to improve out-of-domain performance.
"""

from typing import Optional

from baseline.models.xgboost_config import XGBoostConfig


class XGBoostEnhancedConfig(XGBoostConfig):
    """Enhanced XGBoost configuration with stronger regularization.
    
    This configuration is designed to reduce overfitting and improve
    generalization to out-of-domain data by:
    1. Using shallower trees (max_depth=4)
    2. Smaller learning rate (0.05)
    3. Stronger L1/L2 regularization
    4. More aggressive subsampling
    5. Higher minimum child weight
    6. Non-zero gamma for pruning
    """
    
    def __init__(
        self,
        # Tree complexity control
        max_depth: int = 4,
        min_child_weight: int = 20,
        gamma: float = 1.0,
        
        # Learning parameters
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        
        # Sampling parameters
        subsample: float = 0.6,
        colsample_bytree: float = 0.5,
        colsample_bylevel: float = 0.5,
        colsample_bynode: float = 0.5,
        
        # Regularization
        reg_alpha: float = 10.0,
        reg_lambda: float = 20.0,
        
        # Other parameters from parent
        objective: str = "multi:softprob",
        n_jobs: int = -1,
        **kwargs
    ):
        """Initialize enhanced XGBoost configuration.
        
        Args:
            max_depth: Maximum tree depth (default: 4, more conservative)
            min_child_weight: Minimum child weight (default: 20, higher for stability)
            gamma: Minimum loss reduction for split (default: 1.0, enables pruning)
            learning_rate: Learning rate (default: 0.05, lower for stability)
            n_estimators: Number of trees (default: 500, more trees with lower rate)
            subsample: Sample ratio of training instances (default: 0.6)
            colsample_bytree: Sample ratio of columns per tree (default: 0.5)
            colsample_bylevel: Sample ratio of columns per level (default: 0.5)
            colsample_bynode: Sample ratio of columns per node (default: 0.5)
            reg_alpha: L1 regularization (default: 10.0, stronger)
            reg_lambda: L2 regularization (default: 20.0, stronger)
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