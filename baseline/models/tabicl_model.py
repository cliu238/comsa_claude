"""TabICL (Tabular In-Context Learning) model implementation for VA classification."""

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from baseline.models.tabicl_config import TabICLConfig

# Try to import TabICL
try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    TabICLClassifier = None


class TabICLModel(BaseEstimator, ClassifierMixin):
    """TabICL model wrapper for VA cause of death classification.
    
    TabICL is a foundation model that uses in-context learning for tabular data.
    It requires minimal hyperparameter tuning and excels on medium-sized datasets.
    """
    
    def __init__(self, config: Optional[TabICLConfig] = None):
        """Initialize TabICL model.
        
        Args:
            config: TabICL configuration object. If None, uses defaults.
        """
        self.config = config or TabICLConfig()
        self._tabicl_model = None
        self._label_encoder = LabelEncoder()
        self._feature_names = None
        self._n_features = None
        self._n_classes = None
        
        # Check TabICL availability
        self._tabicl_available = TABICL_AVAILABLE
        if not self._tabicl_available:
            warnings.warn(
                "TabICL not installed. Install with: pip install tabicl torch",
                UserWarning
                )
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _check_tabicl_available(self):
        """Check if TabICL is available."""
        if not self._tabicl_available:
            raise ImportError(
                "TabICL is not installed. Please install it with: "
                "pip install tabicl torch"
            )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabICLModel":
        """Fit the TabICL model.
        
        Args:
            X: Feature matrix as DataFrame
            y: Target labels as Series
            
        Returns:
            Self for method chaining
        """
        self._check_tabicl_available()
        
        # Store feature information
        self._feature_names = list(X.columns)
        self._n_features = len(self._feature_names)
        
        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)
        self._n_classes = len(self._label_encoder.classes_)
        
        # Validate VA constraints
        if self._n_classes > self.config.max_classes_warning:
            warnings.warn(
                f"TabICL is optimized for <=10 classes, found {self._n_classes}. "
                "Consider using XGBoost or Random Forest for better performance.",
                UserWarning
            )
        
        if self._n_features > self.config.max_features_warning:
            warnings.warn(
                f"TabICL is optimized for <=100 features, found {self._n_features}. "
                "Consider feature selection or dimensionality reduction.",
                UserWarning
            )
        
        # Initialize TabICL directly
        if TABICL_AVAILABLE:
            device = self._get_device()
            self._tabicl_model = TabICLClassifier(
                n_estimators=self.config.n_estimators,
                norm_methods=self.config.norm_methods,
                feat_shuffle_method=self.config.feat_shuffle_method,
                class_shift=self.config.class_shift,
                outlier_threshold=self.config.outlier_threshold,
                softmax_temperature=self.config.softmax_temperature,
                average_logits=self.config.average_logits,
                use_hierarchical=self.config.use_hierarchical,
                batch_size=self.config.batch_size,
                use_amp=self.config.use_amp,
                checkpoint_version=self.config.checkpoint_version,
                device=device,
                random_state=self.config.random_state
            )
            
            # Fit the model with memory fallback
            try:
                self._tabicl_model.fit(X.values, y_encoded)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(
                        "GPU out of memory. Falling back to CPU.",
                        UserWarning
                    )
                    self._tabicl_model.device = "cpu"
                    self._tabicl_model.fit(X.values, y_encoded)
                else:
                    raise
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cause of death labels.
        
        Args:
            X: Feature matrix as DataFrame
            
        Returns:
            Predicted labels as string array
        """
        check_is_fitted(self, "_tabicl_model")
        self._check_tabicl_available()
        
        # Validate features
        if list(X.columns) != self._feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self._feature_names}, "
                f"got {list(X.columns)}"
            )
        
        # Get predictions with error handling
        try:
            y_pred_encoded = self._tabicl_model.predict(X.values)
        except Exception as e:
            if "out of memory" in str(e).lower():
                # Try batch prediction
                y_pred_encoded = self._predict_in_batches(X.values)
            else:
                raise
        
        # Decode labels
        return self._label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cause of death probabilities.
        
        Args:
            X: Feature matrix as DataFrame
            
        Returns:
            Probability matrix of shape (n_samples, n_classes)
        """
        check_is_fitted(self, "_tabicl_model")
        self._check_tabicl_available()
        
        # Validate features
        if list(X.columns) != self._feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self._feature_names}, "
                f"got {list(X.columns)}"
            )
        
        # Get probabilities with error handling
        try:
            proba = self._tabicl_model.predict_proba(X.values)
        except Exception as e:
            if "out of memory" in str(e).lower():
                # Try batch prediction
                proba = self._predict_proba_in_batches(X.values)
            else:
                raise
        
        return proba
    
    def _predict_in_batches(self, X: np.ndarray) -> np.ndarray:
        """Predict in smaller batches to avoid memory issues."""
        n_samples = X.shape[0]
        batch_size = max(1, n_samples // 10)  # Use 10 batches
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            pred = self._tabicl_model.predict(batch)
            predictions.append(pred)
        
        return np.concatenate(predictions)
    
    def _predict_proba_in_batches(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities in smaller batches."""
        n_samples = X.shape[0]
        batch_size = max(1, n_samples // 10)
        probabilities = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            proba = self._tabicl_model.predict_proba(batch)
            probabilities.append(proba)
        
        return np.vstack(probabilities)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Args:
            deep: If True, return parameters for sub-objects
            
        Returns:
            Parameter dictionary
        """
        params = {"config": self.config}
        if deep and self.config is not None:
            config_params = self.config.model_dump()
            for key, value in config_params.items():
                params[f"config__{key}"] = value
        return params
    
    def set_params(self, **params: Any) -> "TabICLModel":
        """Set parameters for this estimator.
        
        Args:
            **params: Parameter dictionary
            
        Returns:
            Self for method chaining
        """
        config_params = {}
        other_params = {}
        
        for key, value in params.items():
            if key.startswith("config__"):
                config_key = key.replace("config__", "")
                config_params[config_key] = value
            else:
                other_params[key] = value
        
        # Update config if needed
        if config_params:
            if self.config is None:
                self.config = TabICLConfig(**config_params)
            else:
                for key, value in config_params.items():
                    setattr(self.config, key, value)
        
        # Handle other parameters
        for key, value in other_params.items():
            setattr(self, key, value)
        
        return self
    
    def calculate_csmf_accuracy(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate Cause-Specific Mortality Fraction (CSMF) accuracy.
        
        Args:
            y_true: True cause of death labels
            y_pred: Predicted cause of death labels
            
        Returns:
            CSMF accuracy score between 0 and 1
        """
        # Calculate true and predicted CSMFs
        true_counts = y_true.value_counts(normalize=True)
        pred_counts = pd.Series(y_pred).value_counts(normalize=True)
        
        # Align the indices
        all_causes = set(true_counts.index) | set(pred_counts.index)
        true_csmf = pd.Series(
            [true_counts.get(cause, 0) for cause in all_causes],
            index=list(all_causes)
        )
        pred_csmf = pd.Series(
            [pred_counts.get(cause, 0) for cause in all_causes],
            index=list(all_causes)
        )
        
        # Calculate CSMF accuracy
        min_csmf = np.minimum(true_csmf.values, pred_csmf.values).sum()
        csmf_accuracy = 2 * (min_csmf - 0.5)
        
        return max(0, min(1, csmf_accuracy))
    
    @property
    def classes_(self) -> np.ndarray:
        """Get the class labels."""
        check_is_fitted(self, "_label_encoder")
        return self._label_encoder.classes_
    
    @property
    def is_using_fallback(self) -> bool:
        """Check if the model is using fallback."""
        # Direct TabICL doesn't have fallback
        return False
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics about the model state."""
        diagnostics = {
            'model_type': 'TabICLModel',
            'is_fitted': self._tabicl_model is not None,
            'is_using_fallback': False,
            'tabicl_available': self._tabicl_available,
            'config': self.config.model_dump() if self.config else None,
            'n_features': self._n_features,
            'n_classes': self._n_classes,
            'feature_names': self._feature_names,
        }
        
        return diagnostics
    
    def __str__(self) -> str:
        """String representation."""
        return f"TabICLModel(n_estimators={self.config.n_estimators}, hierarchical={self.config.use_hierarchical})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"TabICLModel(config={self.config}, "
            f"n_features={self._n_features}, "
            f"n_classes={self._n_classes}, "
            f"hierarchical={self.config.use_hierarchical})"
        )