"""Configuration for Categorical Naive Bayes model using Pydantic."""

from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class CategoricalNBConfig(BaseModel):
    """Configuration for CategoricalNB model following established patterns.

    This configuration class defines all parameters needed for the CategoricalNB
    model, including smoothing parameters, prior specification, and performance settings.
    """

    # Core hyperparameters for CategoricalNB
    alpha: float = Field(
        default=1.0, gt=0, description="Additive (Laplace/Lidstone) smoothing parameter"
    )
    fit_prior: bool = Field(
        default=True, description="Whether to learn class prior probabilities"
    )
    class_prior: Optional[np.ndarray] = Field(
        default=None, description="Prior probabilities of the classes"
    )
    force_alpha: bool = Field(
        default=False,
        description="Force alpha to be used exactly as specified"
    )

    # Performance parameters (following established pattern)
    random_state: int = Field(
        default=42, description="Random seed for reproducibility"
    )

    # Note: alpha validation is handled by Pydantic Field(gt=0)

    @field_validator("class_prior", mode="before")
    @classmethod
    def validate_class_prior(
        cls, v: Optional[Union[np.ndarray, list]]
    ) -> Optional[np.ndarray]:
        """Validate class_prior parameter.

        Args:
            v: Class prior values to validate

        Returns:
            Validated class_prior as numpy array or None

        Raises:
            ValueError: If class_prior values are invalid
        """
        if v is None:
            return v

        # Convert to numpy array if list
        if isinstance(v, list):
            v = np.array(v)

        if not isinstance(v, np.ndarray):
            raise ValueError("class_prior must be a numpy array, list, or None")

        # Check that all values are non-negative
        if np.any(v < 0):
            raise ValueError("All class_prior values must be non-negative")

        # Check that values sum to 1 (allowing for small floating point errors)
        if not np.isclose(v.sum(), 1.0, rtol=1e-5):
            raise ValueError("class_prior values must sum to 1.0")

        return v

    @field_validator("random_state")
    @classmethod
    def validate_random_state(cls, v: int) -> int:
        """Validate random_state parameter.

        Args:
            v: Random state value to validate

        Returns:
            Validated random state

        Raises:
            ValueError: If random_state is negative
        """
        if v < 0:
            raise ValueError("random_state must be non-negative")
        return v

    model_config = ConfigDict(
        validate_assignment=True, 
        extra="forbid", 
        arbitrary_types_allowed=True
    )

