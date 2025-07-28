"""Configuration for VA34 site comparison experiments."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""
    
    enabled: bool = Field(default=False, description="Enable hyperparameter tuning")
    n_trials: int = Field(default=100, description="Number of tuning trials")
    search_algorithm: str = Field(
        default="bayesian", 
        description="Search algorithm: 'grid', 'random', or 'bayesian'"
    )
    max_concurrent_trials: Optional[int] = Field(
        default=None, 
        description="Max parallel trials (None for automatic)"
    )
    cv_folds: int = Field(default=5, description="Cross-validation folds for tuning")
    tuning_metric: str = Field(
        default="csmf_accuracy", 
        description="Metric to optimize during tuning"
    )
    n_cpus_per_trial: float = Field(
        default=1.0,
        description="Number of CPUs allocated per tuning trial"
    )
    
    @field_validator("search_algorithm")
    def validate_search_algorithm(cls, v: str) -> str:
        """Validate search algorithm."""
        valid_algorithms = ["grid", "random", "bayesian"]
        if v not in valid_algorithms:
            raise ValueError(f"Search algorithm {v} not in {valid_algorithms}")
        return v
    
    @field_validator("tuning_metric")
    def validate_tuning_metric(cls, v: str) -> str:
        """Validate tuning metric."""
        valid_metrics = ["csmf_accuracy", "cod_accuracy"]
        if v not in valid_metrics:
            raise ValueError(f"Tuning metric {v} not in {valid_metrics}")
        return v
    
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
    
    # Tuning configuration
    tuning: TuningConfig = Field(
        default_factory=TuningConfig, 
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
        valid_models = ["insilico", "xgboost", "random_forest", "logistic_regression", "categorical_nb"]
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
