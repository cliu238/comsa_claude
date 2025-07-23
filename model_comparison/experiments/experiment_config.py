"""Configuration for VA34 site comparison experiments."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


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
        default=[0.25, 0.5, 0.75, 1.0], description="Fractions of training data to use"
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
    output_dir: str = Field(default="model_comparison/results/va34_comparison")
    save_predictions: bool = Field(default=True)
    generate_plots: bool = Field(default=True)

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
        valid_models = ["insilico", "xgboost"]
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
