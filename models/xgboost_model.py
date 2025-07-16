"""XGBoost model implementation."""

from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from ml_pipeline.models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def _initialize_model(self) -> None:
        """Initialize the XGBoost model with default parameters."""
        default_params = {
            "objective": "multi:softprob",
            "n_estimators": 250,
            "learning_rate": 0.06962550818742679,
            "max_depth": 6,
            "subsample": 0.6034989467874442,
            "colsample_bytree": 0.8708817658975764,
            "min_child_weight": 4,
            "reg_alpha": 0.1,  # L1正则化（Lasso）
            "reg_lambda": 1.0,  # L2正则化（Ridge）
            "gamma": 0.1,  # 最小分裂损失
            "random_state": None,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        }
        params = {**default_params, **self.model_params}
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        """Fit the model to the data.

        Args:
            X: Features to train on
            y: Target values
        """

        if self.model is None:
            raise ValueError("Model not initialized")

        # Store feature names
        self.feature_names_in_ = X.columns.tolist()

        # Get unique classes and sort them
        unique_classes = sorted(np.unique(y))

        # Create a mapping from original labels to consecutive integers
        self.class_mapping = {label: idx for idx, label in enumerate(unique_classes)}

        # Map labels to consecutive integers
        y_encoded = np.array([self.class_mapping[label] for label in y])

        # Update XGBoost parameters to match number of classes
        self.model.set_params(num_class=len(unique_classes))

        # Fit model with encoded labels
        self.model.fit(X, y_encoded)

    def predict(self, X):
        """Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions in original label space
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        # Get predictions in encoded space
        y_pred_encoded = self.model.predict(X)

        # Create reverse mapping
        reverse_mapping = {idx: label for label, idx in self.class_mapping.items()}

        # Convert back to original label space and ensure integers
        y_pred = np.array([reverse_mapping[pred] for pred in y_pred_encoded], dtype=int)

        return y_pred

    def predict_proba(self, X):
        """Get probability predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of probability predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        # Get predictions
        y_pred_proba = self.model.predict_proba(X)

        return y_pred_proba

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
