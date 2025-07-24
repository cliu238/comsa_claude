"""Configuration for Random Forest model using Pydantic."""

from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest model following InSilicoVA pattern.

    This configuration class defines all parameters needed for the Random Forest
    model, including hyperparameters, performance settings, and class balancing
    options.
    """

    # Model parameters
    n_estimators: int = Field(
        default=100, ge=1, le=5000, description="Number of trees in the forest"
    )
    max_depth: Optional[int] = Field(
        default=None, ge=1, description="Maximum depth of trees"
    )
    min_samples_split: int = Field(
        default=2, ge=2, description="Minimum samples required to split a node"
    )
    min_samples_leaf: int = Field(
        default=1, ge=1, description="Minimum samples required at a leaf node"
    )
    max_features: Union[str, int, float, None] = Field(
        default="sqrt", description="Number of features to consider for splits"
    )
    max_leaf_nodes: Optional[int] = Field(
        default=None, ge=2, description="Maximum number of leaf nodes"
    )

    # Ensemble parameters
    bootstrap: bool = Field(
        default=True, description="Whether to use bootstrap samples"
    )
    oob_score: bool = Field(
        default=False, description="Whether to use out-of-bag samples for scoring"
    )

    # Class imbalance handling
    class_weight: Union[str, Dict[int, float], None] = Field(
        default="balanced", description="Class weights for imbalanced data"
    )

    # Performance
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    verbose: int = Field(default=0, ge=0, description="Verbosity level")

    # Tree parameters
    criterion: str = Field(
        default="gini", description="Function to measure split quality"
    )
    min_weight_fraction_leaf: float = Field(
        default=0.0, ge=0.0, le=0.5, description="Minimum weighted fraction in leaf"
    )
    max_samples: Optional[Union[int, float]] = Field(
        default=None, description="Number of samples to draw for training each tree"
    )

    @field_validator("max_features")
    def validate_max_features(cls, v: Union[str, int, float, None]) -> Union[str, int, float, None]:
        """Validate max_features parameter."""
        if v is None:
            return v
        if isinstance(v, str):
            valid_strings = ["sqrt", "log2"]
            if v not in valid_strings:
                raise ValueError(f"max_features string must be one of {valid_strings}")
        elif isinstance(v, int):
            if v < 1:
                raise ValueError("max_features int must be >= 1")
        elif isinstance(v, float):
            if not 0.0 < v <= 1.0:
                raise ValueError("max_features float must be in (0.0, 1.0]")
        return v

    @field_validator("criterion")
    def validate_criterion(cls, v: str) -> str:
        """Validate criterion parameter."""
        valid_criteria = ["gini", "entropy", "log_loss"]
        if v not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}")
        return v

    @field_validator("class_weight")
    def validate_class_weight(cls, v: Union[str, Dict[int, float], None]) -> Union[str, Dict[int, float], None]:
        """Validate class_weight parameter."""
        if v is None:
            return v
        if isinstance(v, str):
            valid_strings = ["balanced", "balanced_subsample"]
            if v not in valid_strings:
                raise ValueError(f"class_weight string must be one of {valid_strings}")
        elif isinstance(v, dict):
            # Ensure all values are positive
            for class_label, weight in v.items():
                if weight <= 0:
                    raise ValueError(f"Class weight for {class_label} must be positive")
        return v

    @field_validator("max_samples")
    def validate_max_samples(cls, v: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        """Validate max_samples parameter."""
        if v is None:
            return v
        if isinstance(v, int):
            if v < 1:
                raise ValueError("max_samples int must be >= 1")
        elif isinstance(v, float):
            if not 0.0 < v <= 1.0:
                raise ValueError("max_samples float must be in (0.0, 1.0]")
        return v

    model_config = {"validate_assignment": True, "extra": "forbid"}