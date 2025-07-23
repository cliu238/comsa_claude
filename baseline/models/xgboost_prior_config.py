"""Configuration for prior-enhanced XGBoost model."""

from typing import Literal, Optional

from pydantic import Field

from .xgboost_config import XGBoostConfig


class XGBoostPriorConfig(XGBoostConfig):
    """Configuration for XGBoost with medical prior integration.
    
    Extends the base XGBoost configuration with parameters for
    incorporating medical knowledge from InSilicoVA priors.
    """
    
    # Prior integration settings
    use_medical_priors: bool = Field(
        default=True,
        description="Whether to use medical priors"
    )
    
    prior_method: Literal["custom_objective", "feature_engineering", "both"] = Field(
        default="both",
        description="Method for incorporating priors"
    )
    
    lambda_prior: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Weight for prior term in custom objective"
    )
    
    prior_data_path: Optional[str] = Field(
        default=None,
        description="Path to InSilicoVA prior data files"
    )
    
    feature_prior_weight: float = Field(
        default=1.0,
        gt=0,
        description="Weight multiplier for prior-based features"
    )
    
    # Prior feature settings
    include_likelihood_features: bool = Field(
        default=True,
        description="Include symptom likelihood features"
    )
    
    include_log_odds_features: bool = Field(
        default=True,
        description="Include log odds features"
    )
    
    include_rank_features: bool = Field(
        default=True,
        description="Include cause rank features"
    )
    
    include_cause_prior_features: bool = Field(
        default=True,
        description="Include population cause prior features"
    )
    
    include_plausibility_features: bool = Field(
        default=True,
        description="Include medical plausibility features"
    )
    
    # Adaptive lambda settings
    lambda_schedule: Literal["constant", "linear_decay", "exponential_decay"] = Field(
        default="constant",
        description="Schedule for lambda_prior during training"
    )
    
    lambda_min: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="Minimum lambda value for decay schedules"
    )
    
    # Validation settings
    enforce_plausibility: bool = Field(
        default=False,
        description="Whether to hard-enforce plausibility constraints"
    )
    
    plausibility_threshold: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Threshold for plausibility enforcement"
    )