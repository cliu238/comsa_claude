"""Logistic Regression model implementation for VA cause-of-death prediction."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from baseline.models.logistic_regression_config import LogisticRegressionConfig

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    """Logistic Regression model for VA cause-of-death prediction.

    This model follows the sklearn interface pattern similar to the InSilicoVA model,
    providing methods for fitting, prediction, and evaluation of cause-of-death
    predictions from verbal autopsy data. It supports multiple regularization
    options (L1, L2, ElasticNet) and provides coefficient-based feature importance.

    Attributes:
        config: LogisticRegressionConfig object containing model parameters
        model_: Trained LogisticRegression object (after fitting)
        label_encoder_: LabelEncoder for encoding/decoding cause labels
        feature_names_: List of feature names from training data
        classes_: Array of unique class labels
        _is_fitted: Boolean indicating if model has been trained
    """

    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        """Initialize Logistic Regression model with configuration.

        Args:
            config: LogisticRegressionConfig object. If None, uses default configuration.
        """
        self.config = config or LogisticRegressionConfig()
        self.model_: Optional[LogisticRegression] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self._is_fitted = False
        self._single_class = False
        self._single_class_label = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.
        """
        params = {"config": self.config}
        if deep and self.config is not None:
            # Add config parameters with prefix
            config_params = self.config.model_dump()
            for key, value in config_params.items():
                params[f"config__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "LogisticRegressionModel":
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Self: Estimator instance.
        """
        if "config" in params:
            self.config = params.pop("config")

        # Handle nested parameters like config__C
        config_params = {}
        for key, value in list(params.items()):
            if key.startswith("config__"):
                config_key = key.replace("config__", "")
                config_params[config_key] = value
                params.pop(key)

        if config_params:
            # Update config with new parameters
            current_config = self.config.model_dump()
            current_config.update(config_params)
            self.config = LogisticRegressionConfig(**current_config)

        # Call parent set_params if there are remaining params
        if params:
            super().set_params(**params)

        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionModel":
        """Fit Logistic Regression model following InSilicoVA pattern.

        Args:
            X: Training features as pandas DataFrame
            y: Training labels as pandas Series
            sample_weight: Optional sample weights for handling class imbalance

        Returns:
            Self: Fitted model instance

        Raises:
            TypeError: If X is not a DataFrame or y is not a Series
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")

        logger.info(f"Training Logistic Regression model with {len(X)} samples")

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        
        # Check for single class edge case
        if len(self.classes_) < 2:
            logger.warning(
                f"Only one class found: {self.classes_[0]}. "
                "Creating a dummy model that always predicts this class."
            )
            # Create a simple model that always predicts the single class
            self._single_class = True
            self._single_class_label = self.classes_[0]
            self._is_fitted = True
            return self

        # Handle class imbalance with sample weights if not provided
        if sample_weight is None and self.config.class_weight == "balanced":
            sample_weight = compute_sample_weight("balanced", y_encoded)
            logger.info("Using balanced sample weights for class imbalance")

        # Convert config to sklearn parameters
        sklearn_params = self._get_sklearn_params()

        # Create and fit model
        self.model_ = LogisticRegression(**sklearn_params)
        self.model_.fit(X.values, y_encoded, sample_weight=sample_weight)
        self._single_class = False

        self._is_fitted = True
        logger.info(
            f"Logistic Regression model trained successfully with {len(self.classes_)} classes"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cause of death.

        Args:
            X: Features as pandas DataFrame

        Returns:
            Array of predicted cause labels (decoded from numeric)

        Raises:
            ValueError: If model is not fitted
            TypeError: If X is not a DataFrame
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Ensure columns match training features
        if list(X.columns) != self.feature_names_:
            raise ValueError(
                f"Feature names mismatch. Expected {self.feature_names_}, "
                f"got {list(X.columns)}"
            )

        # Handle single class case
        if self._single_class:
            return np.array([self._single_class_label] * len(X))
        
        # Get predictions
        y_pred_encoded = self.model_.predict(X.values)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability distribution over causes.

        Args:
            X: Features as pandas DataFrame

        Returns:
            2D array of shape (n_samples, n_classes) with probability distributions

        Raises:
            ValueError: If model is not fitted
            TypeError: If X is not a DataFrame
        """
        self._check_is_fitted()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Ensure columns match training features
        if list(X.columns) != self.feature_names_:
            raise ValueError(
                f"Feature names mismatch. Expected {self.feature_names_}, "
                f"got {list(X.columns)}"
            )

        # Handle single class case
        if self._single_class:
            # Return all 1.0 for the single class
            return np.ones((len(X), 1))
        
        # Get probability predictions
        return self.model_.predict_proba(X.values)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from model coefficients.

        For logistic regression, feature importance is derived from the absolute
        values of the coefficients. For multiclass problems, we aggregate across
        all classes by taking the mean absolute coefficient.

        Returns:
            DataFrame with 'feature' and 'importance' columns, sorted by importance

        Raises:
            ValueError: If model is not fitted
        """
        self._check_is_fitted()
        
        # Handle single class case
        if self._single_class:
            # Return zero importance for all features
            return pd.DataFrame({
                "feature": self.feature_names_,
                "importance": np.zeros(len(self.feature_names_))
            })

        # Get coefficients
        coef = self.model_.coef_

        if coef.shape[0] == 1:
            # Binary classification case (though unlikely for VA data)
            importance = np.abs(coef[0])
        else:
            # Multiclass case - aggregate across classes
            # Use mean absolute coefficient across all classes
            importance = np.mean(np.abs(coef), axis=0)

        # Create DataFrame
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        })

        # Sort by importance
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5, stratified: bool = True
    ) -> Dict[str, Any]:
        """Perform cross-validation with stratification.

        Args:
            X: Features as pandas DataFrame
            y: Labels as pandas Series
            cv: Number of cross-validation folds
            stratified: Whether to use stratified K-fold

        Returns:
            Dictionary with mean scores:
                - csmf_accuracy: CSMF accuracy scores
                - cod_accuracy: Individual COD accuracy scores

        Raises:
            ValueError: If cv < 2
        """
        if cv < 2:
            raise ValueError("cv must be at least 2")

        if stratified:
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        scores: Dict[str, List[float]] = {
            "csmf_accuracy": [],
            "cod_accuracy": [],
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{cv}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone model for this fold
            model = LogisticRegressionModel(config=self.config)
            model.fit(X_train, y_train)

            # Calculate metrics
            y_pred = model.predict(X_val)

            # CSMF accuracy
            csmf_acc = self.calculate_csmf_accuracy(y_val, y_pred)
            scores["csmf_accuracy"].append(csmf_acc)

            # COD accuracy
            cod_acc = (y_val == y_pred).mean()
            scores["cod_accuracy"].append(cod_acc)

        # Return mean scores
        return {
            "csmf_accuracy_mean": np.mean(scores["csmf_accuracy"]),
            "csmf_accuracy_std": np.std(scores["csmf_accuracy"]),
            "cod_accuracy_mean": np.mean(scores["cod_accuracy"]),
            "cod_accuracy_std": np.std(scores["cod_accuracy"]),
            "csmf_accuracy_scores": scores["csmf_accuracy"],
            "cod_accuracy_scores": scores["cod_accuracy"],
        }

    def calculate_csmf_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate CSMF accuracy following InSilicoVA implementation.

        CSMF (Cause-Specific Mortality Fraction) accuracy measures how well
        the predicted distribution of causes matches the true distribution.

        Formula: 1 - sum(|pred_fraction - true_fraction|) / (2 * (1 - min(true_fraction)))

        Args:
            y_true: True cause labels
            y_pred: Predicted cause labels

        Returns:
            CSMF accuracy score between 0 and 1
        """
        # Get true and predicted fractions
        true_fractions = y_true.value_counts(normalize=True)
        pred_fractions = pd.Series(y_pred).value_counts(normalize=True)

        # Align categories
        all_categories = list(set(true_fractions.index) | set(pred_fractions.index))
        true_fractions = true_fractions.reindex(all_categories, fill_value=0)
        pred_fractions = pred_fractions.reindex(all_categories, fill_value=0)

        # Calculate CSMF accuracy
        diff = np.abs(true_fractions - pred_fractions).sum()
        min_frac = true_fractions.min()

        # Handle edge case where min_frac = 1 (single class)
        if min_frac == 1:
            return 1.0 if diff == 0 else 0.0

        csmf_accuracy = 1 - diff / (2 * (1 - min_frac))

        return max(0, csmf_accuracy)  # Ensure non-negative

    def _get_sklearn_params(self) -> Dict[str, Any]:
        """Convert config to sklearn LogisticRegression parameters.

        Returns:
            Dictionary of sklearn parameters
        """
        params = {
            "penalty": self.config.penalty,
            "C": self.config.C,
            "solver": self.config.solver,
            "max_iter": self.config.max_iter,
            "tol": self.config.tol,
            "multi_class": self.config.multi_class,
            "class_weight": self.config.class_weight,
            "fit_intercept": self.config.fit_intercept,
            "intercept_scaling": self.config.intercept_scaling,
            "n_jobs": self.config.n_jobs,
            "random_state": self.config.random_state,
            "verbose": self.config.verbose,
            "warm_start": self.config.warm_start,
        }

        # Add l1_ratio only if using elasticnet
        if self.config.penalty == "elasticnet":
            params["l1_ratio"] = self.config.l1_ratio

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        return params

    def _check_is_fitted(self) -> None:
        """Check if model is fitted.

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
        # For single class case, model_ can be None
        if not self._single_class and self.model_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )