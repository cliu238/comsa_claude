"""Base model class for all ML models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseModel(ABC):
    """Base class for all ML models in the pipeline."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the model.

        Args:
            model_params: Dictionary of model parameters
        """
        self.model_params = model_params or {}
        self.model: Optional[BaseEstimator] = None
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the specific model implementation."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to the data.

        Args:
            X: Training features
            y: Training labels
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Array of probability predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("Model does not support probability predictions")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.

        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.get_params()

    def set_params(self, **params: Any) -> None:
        """Set the model parameters.

        Args:
            **params: Model parameters to set
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.set_params(**params)
        self.model_params.update(params)
