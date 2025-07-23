"""XGBoost model implementation for VA cause-of-death prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from baseline.models.xgboost_config import XGBoostConfig

logger = logging.getLogger(__name__)


class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost model for VA cause-of-death prediction.

    This model follows the sklearn interface pattern similar to the InSilicoVA model,
    providing methods for fitting, prediction, and evaluation of cause-of-death
    predictions from verbal autopsy data.

    Attributes:
        config: XGBoostConfig object containing model parameters
        model_: Trained XGBoost Booster object (after fitting)
        label_encoder_: LabelEncoder for encoding/decoding cause labels
        feature_names_: List of feature names from training data
        classes_: Array of unique class labels
        _is_fitted: Boolean indicating if model has been trained
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        """Initialize XGBoost model with configuration.

        Args:
            config: XGBoostConfig object. If None, uses default configuration.
        """
        self.config = config or XGBoostConfig()
        self.model_: Optional[xgb.Booster] = None
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
        
    def set_params(self, **params: Any) -> "XGBoostModel":
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
            self.config = XGBoostConfig(**current_config)
            
        # Call parent set_params if there are remaining params
        if params:
            super().set_params(**params)
            
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "XGBoostModel":
        """Fit XGBoost model following InSilicoVA pattern.

        Args:
            X: Training features as pandas DataFrame
            y: Training labels as pandas Series
            sample_weight: Optional sample weights for handling class imbalance
            eval_set: Optional list of (X, y) tuples for validation during training

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

        logger.info(f"Training XGBoost model with {len(X)} samples")

        # Store feature names
        self.feature_names_ = X.columns.tolist()

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Update num_class if not set
        if self.config.num_class is None:
            self.config.num_class = len(self.classes_)

        # Handle class imbalance with sample weights if not provided
        if sample_weight is None and len(np.unique(y_encoded)) > 1:
            sample_weight = compute_sample_weight("balanced", y_encoded)
            logger.info("Using balanced sample weights for class imbalance")

        # Create DMatrix with missing value handling
        dtrain = xgb.DMatrix(
            X.values,
            label=y_encoded,
            feature_names=self.feature_names_,
            weight=sample_weight,
            missing=self.config.missing,
        )

        # Prepare parameters
        params = self._get_xgb_params()

        # Prepare eval list if provided
        evals = []
        if eval_set is not None:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                y_eval_encoded = self.label_encoder_.transform(y_eval)
                deval = xgb.DMatrix(
                    X_eval.values,
                    label=y_eval_encoded,
                    feature_names=self.feature_names_,
                    missing=self.config.missing,
                )
                evals.append((deval, f"eval_{i}"))
        else:
            evals = [(dtrain, "train")]

        # Train model
        callbacks = []
        if self.config.early_stopping_rounds and len(evals) > 1:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.config.early_stopping_rounds,
                    save_best=True,
                )
            )

        self.model_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False,
        )

        self._is_fitted = True
        logger.info(
            f"XGBoost model trained with {self.model_.num_boosted_rounds()} rounds"
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

        # Create DMatrix
        dtest = xgb.DMatrix(
            X.values,
            feature_names=self.feature_names_,
            missing=self.config.missing,
        )

        # Get predictions
        proba = self.model_.predict(dtest)

        # Ensure 2D array
        if proba.ndim == 1:
            proba = proba.reshape(-1, self.config.num_class)

        return proba

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance metric.
                Options: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'

        Returns:
            DataFrame with 'feature' and 'importance' columns, sorted by importance

        Raises:
            ValueError: If model is not fitted or importance_type is invalid
        """
        self._check_is_fitted()

        valid_types = ["gain", "weight", "cover", "total_gain", "total_cover"]
        if importance_type not in valid_types:
            raise ValueError(f"importance_type must be one of {valid_types}")

        importance = self.model_.get_score(importance_type=importance_type)

        # Create DataFrame
        if importance:
            df = pd.DataFrame(
                [{"feature": k, "importance": v} for k, v in importance.items()]
            )
            # Sort by importance
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            # No features used in model
            df = pd.DataFrame(columns=["feature", "importance"])

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
            model = XGBoostModel(config=self.config)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

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

    def _get_xgb_params(self) -> Dict[str, Any]:
        """Convert config to XGBoost parameters.

        Returns:
            Dictionary of XGBoost parameters
        """
        params = {
            "objective": self.config.objective,
            "num_class": self.config.num_class,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "tree_method": self.config.tree_method,
            "device": self.config.device,
            "nthread": self.config.n_jobs,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "eval_metric": "mlogloss",
            "seed": 42,  # For reproducibility
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Handle class weights for imbalanced data
        if self.config.scale_pos_weight is not None:
            # XGBoost doesn't directly support per-class weights in multi-class
            # We'll need to use sample weights instead
            logger.warning(
                "scale_pos_weight not directly supported for multi-class; "
                "use sample_weight in fit()"
            )

        return params

    def _check_is_fitted(self) -> None:
        """Check if model is fitted.

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted or self.model_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
