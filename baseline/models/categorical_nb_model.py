"""CategoricalNB model implementation for VA cause-of-death prediction."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

from baseline.models.categorical_nb_config import CategoricalNBConfig

logger = logging.getLogger(__name__)


class CategoricalNBModel(BaseEstimator, ClassifierMixin):
    """CategoricalNB model for VA cause-of-death prediction.

    This model follows the sklearn interface pattern similar to LogisticRegressionModel,
    providing methods for fitting, prediction, and evaluation of cause-of-death
    predictions from verbal autopsy data. It uses Categorical Naive Bayes which
    handles categorical features natively and is robust to missing data.

    Attributes:
        config: CategoricalNBConfig object containing model parameters
        model_: Trained CategoricalNB object (after fitting)
        label_encoder_: LabelEncoder for encoding/decoding cause labels
        feature_names_: List of feature names from training data
        classes_: Array of unique class labels
        _is_fitted: Boolean indicating if model has been trained
        _single_class: Boolean indicating if only one class in training data
        _single_class_label: Label of single class (if applicable)
    """

    def __init__(self, config: Optional[CategoricalNBConfig] = None):
        """Initialize CategoricalNB model with configuration.

        Args:
            config: CategoricalNBConfig object. If None, uses default configuration.
        """
        self.config = config or CategoricalNBConfig()
        self.model_: Optional[CategoricalNB] = None
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

    def set_params(self, **params: Any) -> "CategoricalNBModel":
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Self: Estimator instance.
        """
        if "config" in params:
            self.config = params.pop("config")

        # Handle nested parameters like config__alpha
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
            self.config = CategoricalNBConfig(**current_config)

        # Call parent set_params if there are remaining params
        if params:
            super().set_params(**params)

        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CategoricalNBModel":
        """Fit CategoricalNB model following established pattern.

        Args:
            X: Training features as pandas DataFrame
            y: Training labels as pandas Series
            sample_weight: Optional sample weights (not used by CategoricalNB)

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

        logger.info(f"Training CategoricalNB model with {len(X)} samples")

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
            self._single_class = True
            self._single_class_label = self.classes_[0]
            self._is_fitted = True
            return self

        # Handle sample weight warning
        if sample_weight is not None:
            logger.warning(
                "CategoricalNB does not support sample_weight. "
                "Ignoring sample_weight parameter."
            )

        # Prepare categorical features
        X_categorical = self._prepare_categorical_features(X)

        # Convert config to sklearn parameters
        sklearn_params = self._get_sklearn_params()

        # Create and fit model
        self.model_ = CategoricalNB(**sklearn_params)
        self.model_.fit(X_categorical, y_encoded)
        self._single_class = False

        self._is_fitted = True
        logger.info(
            f"CategoricalNB model trained successfully with "
            f"{len(self.classes_)} classes"
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

        # Prepare categorical features
        X_categorical = self._prepare_categorical_features(X)

        # Get predictions
        y_pred_encoded = self.model_.predict(X_categorical)
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

        # Prepare categorical features
        X_categorical = self._prepare_categorical_features(X)

        # Get probability predictions
        return self.model_.predict_proba(X_categorical)

    def _prepare_categorical_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare categorical features for CategoricalNB.

        CategoricalNB expects integer-encoded categorical features without NaN values.
        This method handles VA-specific categorical encoding where Y/N/.'/DK/missing
        values need to be mapped to integer categories.

        Args:
            X: DataFrame with categorical data

        Returns:
            2D numpy array with integer-encoded categorical features
        """
        # VA-specific categorical mapping - more comprehensive than in PRP
        categorical_mapping = {
            # Yes variations -> 0
            'Y': 0, 'y': 0, 'Yes': 0, 'yes': 0, 'YES': 0,
            1: 0, '1': 0, 1.0: 0, True: 0, 'True': 0, 'true': 0,

            # No variations -> 1
            'N': 1, 'n': 1, 'No': 1, 'no': 1, 'NO': 1,
            0: 1, '0': 1, 0.0: 1, False: 1, 'False': 1, 'false': 1,

            # Missing/Unknown variations -> 2
            '.': 2, 'DK': 2, 'dk': 2, 'Dk': 2, 'dK': 2, 'DON\'T KNOW': 2,
            np.nan: 2, 'nan': 2, 'NaN': 2, 'NAN': 2,
            None: 2, 'None': 2, 'NONE': 2, '': 2, ' ': 2,
            'missing': 2, 'Missing': 2, 'MISSING': 2,
            'unknown': 2, 'Unknown': 2, 'UNKNOWN': 2,
            'NA': 2, 'na': 2, 'N/A': 2, 'n/a': 2,
        }

        # Initialize output array
        X_encoded = np.zeros((len(X), len(X.columns)), dtype=int)

        # Process each column
        for col_idx, col_name in enumerate(X.columns):
            col_data = X[col_name]

            # Apply mapping to each value in the column
            for row_idx, value in enumerate(col_data):
                if value in categorical_mapping:
                    X_encoded[row_idx, col_idx] = categorical_mapping[value]
                else:
                    # Handle unmapped values - convert to string and try basic mapping
                    str_value = str(value).strip().lower()
                    if str_value in ['y', 'yes', '1', 'true']:
                        X_encoded[row_idx, col_idx] = 0
                    elif str_value in ['n', 'no', '0', 'false']:
                        X_encoded[row_idx, col_idx] = 1
                    else:
                        # Default to missing category for unknown values
                        X_encoded[row_idx, col_idx] = 2
                        logger.debug(
                            f"Unknown value '{value}' in column '{col_name}' "
                            "mapped to missing category"
                        )

        logger.debug(f"Categorical encoding completed. Shape: {X_encoded.shape}")
        return X_encoded

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from log probability ratios.

        For CategoricalNB, feature importance is derived from the log probability
        ratios. For each feature, we compute the difference between the maximum
        and minimum log probability across classes, which indicates how much
        the feature discriminates between classes.

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

        # Get the log probabilities for each feature and class
        # feature_log_prob_ is a list of arrays, one per feature
        # Each array has shape (n_classes, n_categories_for_this_feature)
        feature_log_prob_list = self.model_.feature_log_prob_
        n_model_features = len(feature_log_prob_list)

        # Ensure we don't exceed the model's feature count
        n_features_to_process = min(len(self.feature_names_), n_model_features)
        
        # Calculate importance for each feature
        importances = []

        for feature_idx in range(n_features_to_process):
            # Get log probabilities for this feature across all classes and categories
            # Shape: (n_classes, n_categories_for_this_feature)
            feature_probs = feature_log_prob_list[feature_idx]

            # Calculate the range of log probabilities (max - min) across all values
            # This measures how much the feature discriminates between classes
            max_log_prob = np.max(feature_probs)
            min_log_prob = np.min(feature_probs)
            importance = max_log_prob - min_log_prob

            importances.append(importance)
        
        # Add zero importance for any additional features beyond model's capacity
        while len(importances) < len(self.feature_names_):
            importances.append(0.0)

        # Create DataFrame
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importances
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
            model = CategoricalNBModel(config=self.config)
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
        """Calculate CSMF accuracy following established implementation.

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
        """Convert config to sklearn CategoricalNB parameters.

        Returns:
            Dictionary of sklearn parameters
        """
        params = {
            "alpha": self.config.alpha,
            "fit_prior": self.config.fit_prior,
            "class_prior": self.config.class_prior,
            "force_alpha": self.config.force_alpha,
        }

        # Remove None values (except class_prior which can be None)
        params = {k: v for k, v in params.items() if v is not None or k == "class_prior"}

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

