"""Explainable Boosting Machine (EBM) model implementation."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator

from ml_pipeline.models.base import BaseModel


class EBMModel(BaseModel):
    """Explainable Boosting Machine model wrapper."""

    def _initialize_model(self) -> None:
        """Initialize the EBM model with provided parameters."""
        self.model = ExplainableBoostingClassifier(**self.model_params)

    def __init__(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_site: Optional[str] = None,
    ):
        """Initialize EBM model.

        Args:
            hyperparameters: Dictionary of hyperparameters for the model
            training_site: Name of the site the model was trained on
        """
        super().__init__(model_params=hyperparameters)
        self.training_site = training_site

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBMModel":
        """Fit the EBM model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        # Convert y to numpy array of type int to avoid interpret bug with pandas nullable integer
        y = np.asarray(y, dtype=int)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of prediction probabilities
        """
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from the model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # EBM provides global feature importances through term_importances
        importances = self.model.term_importances()
        return dict(zip(self.model.feature_names_in_, importances))

    def explain_global(self) -> Dict[str, Any]:
        """Get global explanations for the model.

        Returns:
            Dictionary containing global explanations
        """
        return self.model.explain_global()

    def explain_local(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get local explanations for specific predictions.

        Args:
            X: Features to explain

        Returns:
            Dictionary containing local explanations
        """
        return self.model.explain_local(X)
