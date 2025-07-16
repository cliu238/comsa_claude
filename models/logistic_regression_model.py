"""Logistic Regression model implementation."""

from typing import Any, Dict, Optional

from sklearn.linear_model import LogisticRegression

from ml_pipeline.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""

    def _initialize_model(self) -> None:
        """Initialize the Logistic Regression model with default parameters."""
        default_params = {
            "penalty": "l1",
            "dual": False,
            "tol": 0.0001,
            "C": 10**-0.33,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "random_state": 42,
            "solver": "saga",
            "max_iter": 1000,
            "verbose": 0,
            "warm_start": False,
            "n_jobs": None,
            "l1_ratio": None,
        }
        params = {**default_params, **self.model_params}
        self.model = LogisticRegression(**params)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        if not hasattr(self.model, "coef_"):
            raise ValueError("Model has not been fitted yet")
        return dict(zip(self.model.feature_names_in_, abs(self.model.coef_[0])))
