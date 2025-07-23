"""Configuration for XGBoost model using Pydantic."""

from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class XGBoostConfig(BaseModel):
    """Configuration for XGBoost model following InSilicoVA pattern.

    This configuration class defines all parameters needed for the XGBoost
    model, including hyperparameters, performance settings, and regularization
    options.
    """

    # Model parameters
    n_estimators: int = Field(
        default=100, ge=1, description="Number of boosting rounds"
    )
    max_depth: int = Field(default=6, ge=1, le=20, description="Maximum tree depth")
    learning_rate: float = Field(default=0.3, gt=0, le=1, description="Learning rate")
    subsample: float = Field(default=1.0, gt=0, le=1, description="Subsample ratio")
    colsample_bytree: float = Field(
        default=1.0, gt=0, le=1, description="Column subsample ratio"
    )

    # Multi-class specific
    objective: str = Field(default="multi:softprob", description="Objective function")
    num_class: Optional[int] = Field(
        default=None, description="Number of classes (auto-detected)"
    )

    # Class imbalance handling
    scale_pos_weight: Optional[Dict[int, float]] = Field(
        default=None, description="Class weights"
    )

    # Performance
    tree_method: str = Field(default="hist", description="Tree construction algorithm")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")
    n_jobs: int = Field(default=-1, description="Number of parallel threads")

    # Regularization
    reg_alpha: float = Field(default=0, ge=0, description="L1 regularization")
    reg_lambda: float = Field(default=1, ge=0, description="L2 regularization")

    # Missing values
    missing: float = Field(
        default=float("nan"), description="Value to treat as missing"
    )

    # Early stopping
    early_stopping_rounds: Optional[int] = Field(
        default=10, description="Early stopping rounds"
    )

    @field_validator("tree_method")
    def validate_tree_method(cls, v: str) -> str:
        """Validate tree_method parameter."""
        valid_methods = ["auto", "exact", "approx", "hist", "gpu_hist"]
        if v not in valid_methods:
            raise ValueError(f"tree_method must be one of {valid_methods}")
        return v

    @field_validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device parameter."""
        if not v.startswith(("cpu", "cuda", "gpu")):
            raise ValueError("device must be 'cpu' or start with 'cuda' or 'gpu'")
        return v

    @field_validator("objective")
    def validate_objective(cls, v: str) -> str:
        """Validate objective parameter for multi-class classification."""
        valid_objectives = ["multi:softmax", "multi:softprob"]
        if v not in valid_objectives:
            raise ValueError(
                f"objective must be one of {valid_objectives} for multi-class classification"
            )
        return v

    model_config = {"validate_assignment": True, "extra": "forbid"}
