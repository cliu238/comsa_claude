"""Advanced XGBoost model with custom objectives and domain adaptation.

This module extends the base XGBoost model with advanced techniques for
improving out-of-domain generalization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.models.xgboost_custom_objectives import (
    CSMFOptimizedObjective,
    create_csmf_eval_metric,
)

logger = logging.getLogger(__name__)


class XGBoostAdvancedModel(XGBoostModel):
    """Advanced XGBoost model with custom objectives and domain adaptation.
    
    This model extends the base XGBoost implementation with:
    - Custom objective functions optimized for CSMF accuracy
    - Domain adaptation techniques for better generalization
    - Monotonic constraints based on medical knowledge
    - Advanced regularization strategies
    """
    
    def __init__(
        self,
        config: Optional[XGBoostEnhancedConfig] = None,
        use_custom_objective: bool = True,
        objective_type: str = "csmf_weighted",
        use_monotonic_constraints: bool = False,
        domain_adaptation: bool = False,
        **objective_kwargs,
    ):
        """Initialize advanced XGBoost model.
        
        Args:
            config: XGBoost configuration
            use_custom_objective: Whether to use custom objective function
            objective_type: Type of custom objective ("csmf_weighted", "focal", "domain_adversarial")
            use_monotonic_constraints: Whether to apply monotonic constraints
            domain_adaptation: Whether to use domain adaptation techniques
            **objective_kwargs: Additional arguments for the objective function
        """
        super().__init__(config)
        
        self.use_custom_objective = use_custom_objective
        self.objective_type = objective_type
        self.use_monotonic_constraints = use_monotonic_constraints
        self.domain_adaptation = domain_adaptation
        self.objective_kwargs = objective_kwargs
        
        # Initialize custom objective if needed
        if self.use_custom_objective:
            self.custom_objective = CSMFOptimizedObjective(
                objective_type=objective_type,
                **objective_kwargs
            )
        
        # Store domain information if using domain adaptation
        self.domain_encoder_: Optional[LabelEncoder] = None
        self.domain_labels_: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        domain_labels: Optional[pd.Series] = None,
        monotonic_constraints: Optional[Dict[str, int]] = None,
    ) -> "XGBoostAdvancedModel":
        """Fit advanced XGBoost model with custom objectives.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            eval_set: Optional validation sets
            domain_labels: Optional domain/site labels for domain adaptation
            monotonic_constraints: Dict mapping feature names to constraint direction
                                 (-1: decreasing, 0: no constraint, 1: increasing)
        
        Returns:
            Self: Fitted model
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        
        logger.info(f"Training advanced XGBoost model with {len(X)} samples")
        
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        
        # Update num_class if not set
        if self.config.num_class is None:
            self.config.num_class = len(self.classes_)
        
        # Handle domain labels for domain adaptation
        if self.domain_adaptation and domain_labels is not None:
            self.domain_encoder_ = LabelEncoder()
            self.domain_labels_ = self.domain_encoder_.fit_transform(domain_labels)
            
            # Add domain labels to objective kwargs
            self.objective_kwargs['domain_labels'] = self.domain_labels_
        
        # Apply monotonic constraints if specified
        constraint_str = None
        if self.use_monotonic_constraints and monotonic_constraints:
            constraint_str = self._create_constraint_string(monotonic_constraints)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(
            X.values,
            label=y_encoded,
            feature_names=self.feature_names_,
            weight=sample_weight,
            missing=self.config.missing,
        )
        
        # Prepare parameters
        params = self._get_xgb_params()
        
        # Use custom objective if specified
        obj = None
        if self.use_custom_objective:
            obj = self.custom_objective.get_xgb_objective()
            # Remove objective from params if using custom
            params.pop('objective', None)
        
        # Add monotonic constraints to params
        if constraint_str:
            params['monotone_constraints'] = constraint_str
        
        # Prepare evaluation sets
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
        
        # Add custom evaluation metric
        feval = create_csmf_eval_metric() if self.use_custom_objective else None
        
        # Callbacks
        callbacks = []
        if self.config.early_stopping_rounds and len(evals) > 1:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.config.early_stopping_rounds,
                    save_best=True,
                    metric_name="csmf_acc" if feval else "mlogloss",
                    maximize=True if feval else False,
                )
            )
        
        # Train model
        self.model_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            obj=obj,
            feval=feval,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False,
        )
        
        self._is_fitted = True
        logger.info(
            f"Advanced XGBoost model trained with {self.model_.num_boosted_rounds()} rounds"
        )
        
        return self
    
    def _create_constraint_string(self, constraints: Dict[str, int]) -> str:
        """Create monotonic constraint string for XGBoost.
        
        Args:
            constraints: Dict mapping feature names to constraint direction
            
        Returns:
            Constraint string for XGBoost parameters
        """
        # Create constraint string based on feature order
        constraint_list = []
        
        for feature in self.feature_names_:
            if feature in constraints:
                constraint_list.append(str(constraints[feature]))
            else:
                constraint_list.append("0")  # No constraint
        
        return "(" + ",".join(constraint_list) + ")"
    
    def create_medical_constraints(self) -> Dict[str, int]:
        """Create monotonic constraints based on medical knowledge.
        
        Returns:
            Dictionary of feature constraints for VA data
        """
        # Example constraints based on medical logic
        # These should be customized based on actual VA features
        constraints = {
            # Symptoms that increase death probability
            "fever_high": 1,  # Higher fever -> higher risk
            "breathing_difficulty": 1,  # More difficulty -> higher risk
            "consciousness_loss": 1,  # Loss of consciousness -> higher risk
            
            # Protective factors
            "healthcare_access": -1,  # Better access -> lower risk
            "vaccination_complete": -1,  # Vaccination -> lower risk
            
            # Age-related (if encoded as numeric)
            "age_years": 1,  # Older age -> higher risk for most causes
        }
        
        # Only return constraints for features that exist
        return {
            feat: direction
            for feat, direction in constraints.items()
            if feat in self.feature_names_
        }
    
    def predict_with_uncertainty(
        self, X: pd.DataFrame, n_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates using dropout at inference.
        
        Args:
            X: Features to predict
            n_iterations: Number of forward passes with dropout
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self._check_is_fitted()
        
        # Get base predictions
        base_preds = self.predict(X)
        
        # Get probability predictions for uncertainty
        all_probs = []
        
        for i in range(n_iterations):
            # Create DMatrix with subsample to simulate dropout
            subsample_rate = 0.8
            n_samples = len(X)
            sample_idx = np.random.choice(
                n_samples,
                size=int(n_samples * subsample_rate),
                replace=False
            )
            
            X_subsample = X.iloc[sample_idx]
            
            # Get predictions
            probs = self.predict_proba(X_subsample)
            
            # Create full prediction array
            full_probs = np.zeros((n_samples, probs.shape[1]))
            full_probs[sample_idx] = probs
            
            all_probs.append(full_probs)
        
        # Calculate uncertainty as entropy of average predictions
        avg_probs = np.mean(all_probs, axis=0)
        uncertainties = -np.sum(avg_probs * np.log(avg_probs + 1e-8), axis=1)
        
        return base_preds, uncertainties
    
    def fit_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_models: int = 5,
        subsample_data: float = 0.8,
        subsample_features: float = 0.8,
    ) -> List["XGBoostAdvancedModel"]:
        """Fit an ensemble of models for better generalization.
        
        Args:
            X: Training features
            y: Training labels
            n_models: Number of models in ensemble
            subsample_data: Fraction of data to use for each model
            subsample_features: Fraction of features to use for each model
            
        Returns:
            List of fitted models
        """
        models = []
        n_samples = len(X)
        n_features = len(X.columns)
        
        for i in range(n_models):
            logger.info(f"Training ensemble model {i+1}/{n_models}")
            
            # Subsample data
            sample_idx = np.random.choice(
                n_samples,
                size=int(n_samples * subsample_data),
                replace=False
            )
            
            # Subsample features
            feature_idx = np.random.choice(
                n_features,
                size=int(n_features * subsample_features),
                replace=False
            )
            selected_features = X.columns[feature_idx].tolist()
            
            # Create model with slightly different config
            config = XGBoostEnhancedConfig(
                **self.config.model_dump(),
                random_state=42 + i,  # Different seed for each model
            )
            
            model = XGBoostAdvancedModel(
                config=config,
                use_custom_objective=self.use_custom_objective,
                objective_type=self.objective_type,
                **self.objective_kwargs,
            )
            
            # Fit on subsampled data
            model.fit(
                X.iloc[sample_idx][selected_features],
                y.iloc[sample_idx]
            )
            
            models.append(model)
        
        return models
    
    def predict_ensemble(
        self, models: List["XGBoostAdvancedModel"], X: pd.DataFrame
    ) -> np.ndarray:
        """Make predictions using model ensemble.
        
        Args:
            models: List of fitted models
            X: Features to predict
            
        Returns:
            Ensemble predictions
        """
        # Collect predictions from all models
        all_probs = []
        
        for model in models:
            # Get features used by this model
            model_features = model.feature_names_
            X_subset = X[model_features]
            
            # Get probability predictions
            probs = model.predict_proba(X_subset)
            all_probs.append(probs)
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        
        # Get class predictions
        pred_indices = np.argmax(avg_probs, axis=1)
        
        # Use first model's label encoder
        return models[0].label_encoder_.inverse_transform(pred_indices)