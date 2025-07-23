"""XGBoost model enhanced with medical priors from InSilicoVA.

This module implements an XGBoost classifier that incorporates medical
knowledge through prior probabilities, improving cross-site generalization.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .medical_priors import PriorCalculator, PriorConstraints, PriorLoader
from .xgboost_model import XGBoostModel
from .xgboost_prior_config import XGBoostPriorConfig

logger = logging.getLogger(__name__)


class XGBoostPriorEnhanced(XGBoostModel):
    """XGBoost classifier enhanced with medical prior knowledge.
    
    This model extends the base XGBoost implementation by incorporating
    medical priors from InSilicoVA, which helps improve generalization
    across different sites/populations.
    
    The priors can be integrated through:
    1. Custom objective function that includes prior probabilities
    2. Feature engineering based on symptom-cause associations
    3. Both approaches combined
    """
    
    def __init__(self, config: Optional[XGBoostPriorConfig] = None):
        """Initialize prior-enhanced XGBoost model.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        if config is None:
            config = XGBoostPriorConfig()
        
        # Initialize parent with base config
        super().__init__(config)
        
        # Store enhanced config
        self.config: XGBoostPriorConfig = config
        
        # Prior-related attributes
        self.priors = None
        self.prior_calculator = None
        self.prior_constraints = None
        self._original_features = None
        self._prior_feature_names = []
        self.custom_objective = None
        self._use_custom_objective = False
        
    def _load_medical_priors(self) -> None:
        """Load medical priors if not already loaded."""
        if self.priors is None:
            logger.info("Loading medical priors...")
            prior_path = Path(self.config.prior_data_path) if self.config.prior_data_path else None
            loader = PriorLoader(prior_path)
            self.priors = loader.load_priors()
            
            # Initialize calculator and constraints
            self.prior_calculator = PriorCalculator(self.priors)
            self.prior_constraints = PriorConstraints(
                self.priors, 
                lambda_prior=self.config.lambda_prior
            )
            
            logger.info(f"Loaded priors with {len(self.priors.symptom_names)} symptoms "
                       f"and {len(self.priors.cause_names)} causes")
    
    def _augment_with_prior_features(self, X: np.ndarray) -> np.ndarray:
        """Augment features with prior-based features.
        
        Args:
            X: Original features (n_samples, n_features)
            
        Returns:
            Augmented feature array
        """
        if not self.config.use_medical_priors or \
           self.config.prior_method not in ["feature_engineering", "both"]:
            return X
        
        # Calculate prior features (calculator expects numpy)
        X_numpy = X.values if isinstance(X, pd.DataFrame) else X
        prior_features = self.prior_calculator.calculate_prior_features(X_numpy)
        
        # Apply feature selection based on config
        selected_features = []
        feature_names = []
        n_causes = len(self.priors.cause_names)
        
        # Select features based on configuration
        idx = 0
        if self.config.include_likelihood_features:
            selected_features.append(prior_features[:, idx:idx+n_causes])
            feature_names.extend([f"prior_likelihood_{c}" for c in self.priors.cause_names])
            idx += n_causes
            
        if self.config.include_log_odds_features:
            selected_features.append(prior_features[:, idx:idx+n_causes])
            feature_names.extend([f"prior_log_odds_{c}" for c in self.priors.cause_names])
            idx += n_causes
            
        if self.config.include_rank_features:
            selected_features.append(prior_features[:, idx:idx+n_causes])
            feature_names.extend([f"prior_rank_{c}" for c in self.priors.cause_names])
            idx += n_causes
            
        if self.config.include_cause_prior_features:
            selected_features.append(prior_features[:, idx:idx+n_causes])
            feature_names.extend([f"cause_prior_{c}" for c in self.priors.cause_names])
            idx += n_causes
            
        if self.config.include_plausibility_features:
            selected_features.append(prior_features[:, idx:idx+n_causes])
            feature_names.extend([f"plausibility_{c}" for c in self.priors.cause_names])
        
        # Concatenate selected features
        if selected_features:
            prior_features_selected = np.hstack(selected_features)
            
            # Apply feature weight
            prior_features_selected *= self.config.feature_prior_weight
            
            # Store feature names for later use
            self._prior_feature_names = feature_names
            
            # Concatenate with original features
            augmented = np.hstack([X_numpy, prior_features_selected])
            
            # If original was DataFrame, return DataFrame
            if isinstance(X, pd.DataFrame):
                all_columns = list(X.columns) + self._prior_feature_names
                return pd.DataFrame(augmented, columns=all_columns, index=X.index)
            else:
                return augmented
        else:
            return X
    
    def _setup_custom_objective(self) -> None:
        """Set up custom objective function if enabled."""
        if not self.config.use_medical_priors or \
           self.config.prior_method not in ["custom_objective", "both"]:
            return
        
        # Create custom objective function
        self.custom_objective = self.prior_constraints.create_fobj()
        self._use_custom_objective = True
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostPriorEnhanced":
        """Fit the prior-enhanced XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            
        Returns:
            Self for method chaining
        """
        # Store original data type
        X_is_dataframe = isinstance(X, pd.DataFrame)
        y_is_series = isinstance(y, pd.Series)
        
        # Store original features for prior calculations (as numpy)
        X_numpy = X.values if X_is_dataframe else X
        y_numpy = y.values if y_is_series else y
        self._original_features = X_numpy.copy()
        
        # Load priors if using them
        if self.config.use_medical_priors:
            self._load_medical_priors()
            
            # Set features for prior constraints
            if self.config.prior_method in ["custom_objective", "both"]:
                self.prior_constraints.set_features(X_numpy)
            
            # Augment features if using feature engineering
            X = self._augment_with_prior_features(X)
            
            # Set up custom objective if needed
            self._setup_custom_objective()
        
        # Check if we need custom objective
        if self._use_custom_objective and self.custom_objective is not None:
            # Convert to numpy for custom objective
            X_numpy = X.values if isinstance(X, pd.DataFrame) else X
            y_numpy = y.values if isinstance(y, pd.Series) else y
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_numpy, label=y_numpy, weight=sample_weight)
            
            # Prepare parameters for custom objective
            params = {
                'num_class': len(np.unique(y_numpy)),
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'tree_method': self.config.tree_method,
                'device': self.config.device,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'seed': 42
            }
            
            # Train with custom objective
            self.booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.n_estimators,
                obj=self.custom_objective,
                verbose_eval=False
            )
            
            # Store label info for compatibility
            self.classes_ = np.unique(y_numpy)
            self.n_classes_ = len(self.classes_)
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y_numpy)
            
            # Mark as fitted
            self._is_fitted = True
        else:
            # Use parent's fit method for standard training
            super().fit(X, y, sample_weight)
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        # Check if fitted
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy
        X = X.values if isinstance(X, pd.DataFrame) else X
        
        # Augment features if using feature engineering
        if self.config.use_medical_priors and \
           self.config.prior_method in ["feature_engineering", "both"]:
            X = self._augment_with_prior_features(X)
        
        # Make predictions
        if hasattr(self, 'booster') and self.booster is not None:
            # Using custom objective, predict with booster
            dtest = xgb.DMatrix(X)
            predictions = self.booster.predict(dtest)
            
            # Ensure 2D shape
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, self.n_classes_)
                
            return predictions
        else:
            # Use parent's predict_proba
            return super().predict_proba(X)
    
    def get_prior_influence_report(self) -> Dict[str, float]:
        """Get report on how priors influenced the model.
        
        Returns:
            Dictionary with prior influence metrics
        """
        if not self.config.use_medical_priors or self.prior_constraints is None:
            return {"prior_influence": "Not using medical priors"}
        
        # Get some predictions to analyze
        if hasattr(self, '_original_features') and self._original_features is not None:
            # Use training data for analysis
            X_sample = self._original_features[:100]  # Sample for efficiency
            predictions = self.predict_proba(X_sample)
            
            # Calculate influence metrics
            influence = self.prior_constraints.evaluate_prior_influence(
                predictions, X_sample
            )
            
            # Add configuration info
            influence.update({
                "prior_method": self.config.prior_method,
                "n_prior_features": len(self._prior_feature_names),
                "feature_prior_weight": self.config.feature_prior_weight
            })
            
            return influence
        else:
            return {"prior_influence": "No data available for analysis"}
    
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """Get feature importance including prior features.
        
        Args:
            importance_type: Type of importance to return
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get base importance
        importance = super().get_feature_importance(importance_type)
        
        # If we added prior features, update the feature names
        if self._prior_feature_names:
            # The importance dict will have generic names, we need to map them
            n_original = len(importance) - len(self._prior_feature_names)
            
            # Create new importance dict with proper names
            new_importance = {}
            
            # Original features
            for i in range(n_original):
                key = f"f{i}"
                if key in importance:
                    new_importance[f"original_f{i}"] = importance[key]
            
            # Prior features
            for i, name in enumerate(self._prior_feature_names):
                key = f"f{n_original + i}"
                if key in importance:
                    new_importance[name] = importance[key]
            
            return new_importance
        else:
            return importance
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.config.use_medical_priors:
            return f"XGBoostPriorEnhanced(method={self.config.prior_method}, λ={self.config.lambda_prior})"
        else:
            return "XGBoostPriorEnhanced(priors=disabled)"