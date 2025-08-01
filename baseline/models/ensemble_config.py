"""Configuration for ensemble models including DuckVotingClassifier.

This module provides configuration management for ensemble models,
including the DuckVotingClassifier and its weighted variant.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple, Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble model parameters.
    
    This configuration class manages all settings for ensemble models,
    including voting strategy, base estimators, and weighting options.
    """
    
    # Core ensemble parameters
    voting: Literal["hard", "soft"] = Field(
        default="soft",
        description="Voting strategy: 'hard' for majority vote, 'soft' for probability averaging"
    )
    
    # Base estimator configuration
    estimators: List[Tuple[str, str]] = Field(
        default=[
            ("xgboost", "xgboost"),
            ("random_forest", "random_forest"),
            ("categorical_nb", "categorical_nb"),
        ],
        description="List of (name, model_type) tuples for base estimators"
    )
    
    # Weighting configuration
    weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for each estimator. If None, equal weights are used"
    )
    
    weight_optimization: Literal["none", "manual", "cv", "performance"] = Field(
        default="none",
        description="Weight optimization strategy"
    )
    
    # Diversity constraints
    min_diversity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum required diversity between base estimators (0-1)"
    )
    
    diversity_metric: Literal["disagreement", "correlation", "kappa"] = Field(
        default="disagreement",
        description="Metric to measure diversity between estimators"
    )
    
    # Ensemble size constraints
    min_estimators: int = Field(
        default=3,
        ge=2,
        description="Minimum number of estimators in ensemble"
    )
    
    max_estimators: int = Field(
        default=7,
        ge=2,
        description="Maximum number of estimators in ensemble"
    )
    
    # Performance thresholds
    min_base_performance: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum required CSMF accuracy for base estimators"
    )
    
    # Ensemble selection strategy
    selection_strategy: Literal["greedy", "exhaustive", "genetic"] = Field(
        default="greedy",
        description="Strategy for selecting base estimators"
    )
    
    # Training configuration
    fit_base_estimators: bool = Field(
        default=True,
        description="Whether to fit base estimators during ensemble fit"
    )
    
    use_pretrained_estimators: bool = Field(
        default=False,
        description="Whether to use pre-trained base estimators"
    )
    
    # Parallelization
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs for fitting base estimators"
    )
    
    # Random state
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    # Validation settings
    validate_estimator_compatibility: bool = Field(
        default=True,
        description="Validate that all estimators have compatible interfaces"
    )
    
    @field_validator("estimators")
    @classmethod
    def validate_estimators(cls, v: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Validate estimator configuration.
        
        Args:
            v: List of estimator tuples
            
        Returns:
            Validated estimator list
            
        Raises:
            ValueError: If estimator configuration is invalid
        """
        if not v:
            raise ValueError("At least one estimator must be specified")
            
        valid_model_types = [
            "xgboost", "random_forest", "logistic_regression", 
            "categorical_nb", "insilico"
        ]
        
        names = [name for name, _ in v]
        if len(names) != len(set(names)):
            raise ValueError("Estimator names must be unique")
            
        for name, model_type in v:
            if model_type not in valid_model_types:
                raise ValueError(
                    f"Model type '{model_type}' not in {valid_model_types}"
                )
                
        return v
    
    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Optional[List[float]], info) -> Optional[List[float]]:
        """Validate weight configuration.
        
        Args:
            v: Weight list or None
            info: Validation info containing other fields
            
        Returns:
            Validated weights
            
        Raises:
            ValueError: If weights are invalid
        """
        if v is None:
            return v
            
        estimators = info.data.get("estimators", [])
        if len(v) != len(estimators):
            raise ValueError(
                f"Number of weights ({len(v)}) must match number of estimators ({len(estimators)})"
            )
            
        if not all(w >= 0 for w in v):
            raise ValueError("All weights must be non-negative")
            
        if sum(v) == 0:
            raise ValueError("At least one weight must be positive")
            
        return v
    
    @field_validator("max_estimators")
    @classmethod
    def validate_max_estimators(cls, v: int, info) -> int:
        """Validate max estimators is greater than min.
        
        Args:
            v: Max estimators value
            info: Validation info
            
        Returns:
            Validated max estimators
            
        Raises:
            ValueError: If max < min
        """
        min_estimators = info.data.get("min_estimators", 2)
        if v < min_estimators:
            raise ValueError(
                f"max_estimators ({v}) must be >= min_estimators ({min_estimators})"
            )
        return v
    
    def get_normalized_weights(self) -> Optional[List[float]]:
        """Get normalized weights that sum to 1.
        
        Returns:
            Normalized weights or None if no weights specified
        """
        if self.weights is None:
            return None
            
        total = sum(self.weights)
        if total == 0:
            return None
            
        return [w / total for w in self.weights]
    
    def get_estimator_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for each base estimator.
        
        Returns:
            Dictionary mapping estimator names to their configurations
        """
        # This will be extended to support custom configs per estimator
        configs = {}
        for name, model_type in self.estimators:
            configs[name] = {
                "model_type": model_type,
                "random_seed": self.random_seed,
            }
        return configs