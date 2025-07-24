"""Random Forest model implementation for VA cause-of-death prediction."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from baseline.models.random_forest_config import RandomForestConfig

logger = logging.getLogger(__name__)


class RandomForestModel(BaseEstimator, ClassifierMixin):
    """Random Forest model for VA cause-of-death prediction.

    This model follows the sklearn interface pattern similar to the InSilicoVA and
    XGBoost models, providing methods for fitting, prediction, and evaluation of
    cause-of-death predictions from verbal autopsy data.

    Attributes:
        config: RandomForestConfig object containing model parameters
        model_: Trained RandomForestClassifier object (after fitting)
        label_encoder_: LabelEncoder for encoding/decoding cause labels
        feature_names_: List of feature names from training data
        classes_: Array of unique class labels
        _is_fitted: Boolean indicating if model has been trained
    """

    def __init__(self, config: Optional[RandomForestConfig] = None):
        """Initialize Random Forest model with configuration.

        Args:
            config: RandomForestConfig object. If None, uses default configuration.
        """
        self.config = config or RandomForestConfig()
        self.model_: Optional[RandomForestClassifier] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self._is_fitted = False

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

    def set_params(self, **params: Any) -> "RandomForestModel":
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Self: Estimator instance.
        """
        if "config" in params:
            self.config = params.pop("config")

        # Handle nested parameters like config__n_estimators
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
            self.config = RandomForestConfig(**current_config)

        # Call parent set_params if there are remaining params
        if params:
            super().set_params(**params)

        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fit Random Forest model following InSilicoVA pattern.

        Args:
            X: Training features as pandas DataFrame
            y: Training labels as pandas Series

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

        logger.info(f"Training Random Forest model with {len(X)} samples")

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Create Random Forest classifier with config parameters
        self.model_ = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            bootstrap=self.config.bootstrap,
            oob_score=self.config.oob_score,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            class_weight=self.config.class_weight,
            criterion=self.config.criterion,
            min_weight_fraction_leaf=self.config.min_weight_fraction_leaf,
            max_samples=self.config.max_samples,
        )

        # Fit model
        self.model_.fit(X.values, y_encoded)

        self._is_fitted = True
        logger.info(f"Random Forest model trained with {self.config.n_estimators} trees")

        if self.config.oob_score and self.config.bootstrap:
            logger.info(f"Out-of-bag score: {self.model_.oob_score_:.4f}")

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
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(class_indices)

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

        # Get predictions
        proba = self.model_.predict_proba(X.values)

        return proba

    def get_feature_importance(
        self, importance_type: str = "mdi", X: Optional[pd.DataFrame] = None, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance metric.
                Options: 'mdi' (Mean Decrease in Impurity), 'permutation'
            X: Features for permutation importance (required if importance_type='permutation')
            y: Labels for permutation importance (required if importance_type='permutation')

        Returns:
            DataFrame with 'feature' and 'importance' columns, sorted by importance

        Raises:
            ValueError: If model is not fitted or importance_type is invalid
            ValueError: If X or y not provided for permutation importance
        """
        self._check_is_fitted()

        valid_types = ["mdi", "permutation"]
        if importance_type not in valid_types:
            raise ValueError(f"importance_type must be one of {valid_types}")

        if importance_type == "mdi":
            # Get built-in feature importances (Mean Decrease in Impurity)
            importance_values = self.model_.feature_importances_
            
            # Create DataFrame
            df = pd.DataFrame({
                "feature": self.feature_names_,
                "importance": importance_values
            })
        else:  # permutation
            if X is None or y is None:
                raise ValueError(
                    "X and y must be provided for permutation importance"
                )
            
            # Encode y if needed
            if isinstance(y, pd.Series):
                y_encoded = self.label_encoder_.transform(y)
            else:
                y_encoded = y
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model_, X.values, y_encoded, 
                n_repeats=10, random_state=42, n_jobs=self.config.n_jobs
            )
            
            # Create DataFrame
            df = pd.DataFrame({
                "feature": self.feature_names_,
                "importance": perm_importance.importances_mean,
                "importance_std": perm_importance.importances_std
            })

        # Sort by importance
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

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
            model = RandomForestModel(config=self.config)
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

    def _check_is_fitted(self) -> None:
        """Check if model is fitted.

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted or self.model_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )