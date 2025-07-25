"""Configuration for VA34 site comparison experiments."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class HyperparameterSearchConfig(BaseModel):
    """Configuration for hyperparameter search."""
    
    enabled: bool = Field(default=False, description="Enable hyperparameter tuning")
    method: str = Field(
        default="optuna",
        pattern="^(grid|random|optuna|ray_tune)$",
        description="Tuning method to use"
    )
    n_trials: int = Field(default=50, ge=1, description="Number of trials")
    timeout_seconds: Optional[float] = Field(
        default=1800,  # 30 minutes
        ge=0,
        description="Maximum time for tuning per model"
    )
    metric: str = Field(
        default="csmf_accuracy",
        pattern="^(csmf_accuracy|cod_accuracy)$",
        description="Metric to optimize"
    )
    cv_folds: int = Field(default=5, ge=2, description="Cross-validation folds")
    cache_dir: str = Field(
        default="cache/tuned_params",
        description="Directory to cache tuned parameters"
    )
    
    model_config = {"validate_assignment": True}


class ExperimentConfig(BaseModel):
    """Configuration for VA34 site comparison experiment."""

    # Data configuration
    data_path: str = Field(..., description="Path to VA data")
    label_type: str = Field(default="va34", description="Label system to use")

    # Site configuration
    sites: List[str] = Field(..., description="List of sites to include")
    test_sites: Optional[List[str]] = Field(
        default=None, description="Specific test sites"
    )

    # Training configuration
    training_sizes: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], description="Fractions of training data to use"
    )

    # Model configuration
    models: List[str] = Field(
        default=["insilico", "xgboost"], description="Models to compare"
    )

    # Experiment settings
    n_bootstrap: int = Field(default=100, description="Bootstrap iterations")
    random_seed: int = Field(default=42, description="Random seed")
    n_jobs: int = Field(default=-1, description="Parallel jobs")

    # Output configuration
    output_dir: str = Field(default="results/va34_comparison")
    save_predictions: bool = Field(default=True)
    generate_plots: bool = Field(default=True)
    
    # Hyperparameter tuning configuration
    hyperparameter_search: Optional[HyperparameterSearchConfig] = Field(
        default=None,
        description="Hyperparameter tuning configuration"
    )

    @field_validator("training_sizes")
    def validate_training_sizes(cls, v: List[float]) -> List[float]:
        """Validate training sizes are between 0 and 1."""
        for size in v:
            if not 0 < size <= 1:
                raise ValueError(f"Training size {size} must be between 0 and 1")
        return v

    @field_validator("models")
    def validate_models(cls, v: List[str]) -> List[str]:
        """Validate model names."""
        valid_models = ["insilico", "xgboost", "random_forest", "logistic_regression"]
        for model in v:
            if model not in valid_models:
                raise ValueError(f"Model {model} not in {valid_models}")
        return v

    @field_validator("label_type")
    def validate_label_type(cls, v: str) -> str:
        """Validate label type."""
        valid_types = ["va34", "cod5"]
        if v not in valid_types:
            raise ValueError(f"Label type {v} not in {valid_types}")
        return v

    model_config = {"validate_assignment": True}
