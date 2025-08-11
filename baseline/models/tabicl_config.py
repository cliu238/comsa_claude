"""Configuration for TabICL (Tabular In-Context Learning) model."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TabICLConfig(BaseModel):
    """Configuration for TabICL model.
    
    TabICL is a foundation model for tabular data that uses in-context learning.
    It requires minimal hyperparameter tuning and performs well on medium-sized datasets.
    """
    
    # Core TabICL parameters
    n_estimators: int = Field(
        default=48,
        ge=1,
        le=64,
        description="Number of ensemble members for diversity (48 optimal for 34 classes)"
    )
    
    softmax_temperature: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Controls prediction confidence (lower = more confident, 0.5 for 34 classes)"
    )
    
    outlier_threshold: float = Field(
        default=4.0,
        ge=2.0,
        le=10.0,
        description="Z-score threshold for outlier clipping"
    )
    
    norm_methods: List[Literal["none", "power", "quantile"]] = Field(
        default=["quantile", "power"],
        description="Normalization strategies to ensemble (quantile+power optimal for binary features)"
    )
    
    feat_shuffle_method: Literal["latin", "random", "none"] = Field(
        default="latin",
        description="Feature permutation approach for ensemble diversity (latin for high-dim)"
    )
    
    class_shift: bool = Field(
        default=True,
        description="Apply cyclic label shifts for ensemble diversity"
    )
    
    average_logits: bool = Field(
        default=True,
        description="Average logits (True) or probabilities (False) in ensemble"
    )
    
    use_hierarchical: bool = Field(
        default=True,
        description="Use hierarchical approach for multi-class problems"
    )
    
    # Memory and performance settings
    batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Batch size for ensemble processing (4 optimal for 34 classes)"
    )
    
    use_amp: bool = Field(
        default=True,
        description="Use automatic mixed precision for faster inference"
    )
    
    offload_to_cpu: bool = Field(
        default=True,
        description="Offload model to CPU between predictions to save GPU memory (recommended for 34 classes)"
    )
    
    # Model checkpoint
    checkpoint_version: str = Field(
        default="tabicl-classifier-v1.1-0506.ckpt",
        description="TabICL checkpoint version to use"
    )
    
    # Device settings
    device: Optional[Literal["cpu", "cuda", "mps", "auto"]] = Field(
        default="auto",
        description="Device to use for inference (auto detects best available)"
    )
    
    # VA-specific settings
    max_classes_warning: int = Field(
        default=50,
        description="Warn if number of classes exceeds this (TabICL handles up to 50 with hierarchical mode)"
    )
    
    max_features_warning: int = Field(
        default=100,
        description="Warn if number of features exceeds this (TabICL optimal for <=100)"
    )
    
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    @field_validator("norm_methods")
    @classmethod
    def validate_norm_methods(cls, v: List[str]) -> List[str]:
        """Validate normalization methods."""
        if not v:
            raise ValueError("At least one normalization method must be specified")
        if len(v) > 3:
            raise ValueError("Maximum 3 normalization methods allowed")
        return v
    
    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        """Validate number of estimators."""
        if v > 48:
            # Warn about memory usage for large ensembles
            import warnings
            warnings.warn(
                f"Using {v} estimators may require significant memory. "
                "Consider using 48 or fewer for optimal performance with 34 classes.",
                UserWarning
            )
        return v
    
    def model_dump(self, **kwargs) -> dict:
        """Export configuration as dictionary."""
        return super().model_dump(**kwargs)