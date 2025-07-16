"""Random Forest model implementation."""

from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def _initialize_model(self) -> None:
        """Initialize the Random Forest model with default parameters."""
        default_params = {
            "n_estimators": 422,
            "criterion": "gini",
            "max_depth": 100,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "min_weight_fraction_leaf": 0.0019633798330106066,
            "max_features": None,
            "max_leaf_nodes": 192,
            "min_impurity_decrease": 0.00015466176672592047,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": None,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0001953722735757501,
            "max_samples": None,
            "monotonic_cst": None,
        }
        params = {**default_params, **self.model_params}
        self.model = RandomForestClassifier(**params)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model has not been fitted yet")
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
