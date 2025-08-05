"""Ensemble model implementations for VA cause-of-death prediction.

This module provides ensemble models including the DuckVotingClassifier
that combines multiple base estimators using voting strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from baseline.models.categorical_nb_config import CategoricalNBConfig
from baseline.models.categorical_nb_model import CategoricalNBModel
from baseline.models.ensemble_config import EnsembleConfig
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.logistic_regression_config import LogisticRegressionConfig
from baseline.models.logistic_regression_model import LogisticRegressionModel
from baseline.models.model_config import InSilicoVAConfig
from baseline.models.random_forest_config import RandomForestConfig
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class DuckVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Enhanced DuckVotingClassifier following sklearn patterns.
    
    This ensemble model combines multiple classifiers using voting strategies
    (hard or soft) and supports weighted voting, diversity constraints,
    and automatic estimator selection.
    
    Attributes:
        config: EnsembleConfig object containing ensemble parameters
        estimators_: List of fitted (name, estimator) tuples
        classes_: Array of unique class labels
        feature_names_: List of feature names from training data
        weights_: Normalized weights for each estimator
        diversity_scores_: Diversity scores between estimators
        _is_fitted: Boolean indicating if ensemble has been trained
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble with configuration.
        
        Args:
            config: EnsembleConfig object. If None, uses default configuration.
        """
        self.config = config or EnsembleConfig()
        self.estimators_: Optional[List[Tuple[str, BaseEstimator]]] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.weights_: Optional[np.ndarray] = None
        self.diversity_scores_: Optional[Dict[str, float]] = None
        self._is_fitted = False
        
    def _create_estimator(self, model_type: str) -> BaseEstimator:
        """Create an estimator instance based on model type.
        
        Creates models with configurations optimized for small datasets
        to avoid internal stratification errors.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model type is not supported
        """
        model_map = {
            "xgboost": (XGBoostModel, XGBoostConfig),
            "random_forest": (RandomForestModel, RandomForestConfig),
            "logistic_regression": (LogisticRegressionModel, LogisticRegressionConfig),
            "categorical_nb": (CategoricalNBModel, CategoricalNBConfig),
            "insilico": (InSilicoVAModel, InSilicoVAConfig),
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class, config_class = model_map[model_type]
        
        # Create config with settings to avoid internal stratification issues
        if model_type == "random_forest":
            # Disable out-of-bag scoring which can fail with small datasets
            config = config_class(oob_score=False, bootstrap=True)
        elif model_type == "xgboost":
            # Use smaller min_child_weight to handle small leaf nodes
            config = config_class(min_child_weight=1)
        elif model_type == "categorical_nb":
            # Use higher alpha for better smoothing with small datasets
            config = config_class(alpha=1.0)
        else:
            # Default config for other models
            config = config_class()
            
        return model_class(config=config)
        
    def _calculate_diversity(
        self, 
        predictions: List[np.ndarray], 
        metric: str = "disagreement"
    ) -> float:
        """Calculate diversity between estimator predictions.
        
        Args:
            predictions: List of prediction arrays from each estimator
            metric: Diversity metric to use
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        n_estimators = len(predictions)
        if n_estimators < 2:
            return 0.0
            
        if metric == "disagreement":
            # Calculate pairwise disagreement rate
            disagreements = []
            for i in range(n_estimators):
                for j in range(i + 1, n_estimators):
                    disagree_rate = np.mean(predictions[i] != predictions[j])
                    disagreements.append(disagree_rate)
            return np.mean(disagreements) if disagreements else 0.0
            
        elif metric == "correlation":
            # Calculate average correlation between predictions
            # Convert to numeric for correlation
            pred_numeric = [
                pd.Series(pred).astype('category').cat.codes 
                for pred in predictions
            ]
            correlations = []
            for i in range(n_estimators):
                for j in range(i + 1, n_estimators):
                    corr = np.corrcoef(pred_numeric[i], pred_numeric[j])[0, 1]
                    correlations.append(1 - abs(corr))  # Convert to diversity
            return np.mean(correlations) if correlations else 0.0
            
        else:
            raise ValueError(f"Unknown diversity metric: {metric}")
            
    def _optimize_weights(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        X_openva: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None
    ) -> np.ndarray:
        """Optimize estimator weights using specified strategy.
        
        Args:
            X: Training features (numeric format)
            y: Training labels
            X_openva: Training features in OpenVA format (optional)
            strategy: Weight optimization strategy (uses config if not provided)
            
        Returns:
            Optimized weights array
        """
        n_estimators = len(self.estimators_)
        strategy = strategy or self.config.weight_optimization
        
        if strategy == "none":
            # Equal weights
            return np.ones(n_estimators) / n_estimators
            
        elif strategy == "manual":
            # Use provided weights
            if self.config.weights is None:
                return np.ones(n_estimators) / n_estimators
            return np.array(self.config.get_normalized_weights())
            
        elif strategy == "performance":
            # Weight by individual performance on validation set
            from sklearn.model_selection import train_test_split, KFold
            
            # Check if we have enough samples for stratified split
            min_samples_needed = 2  # Minimum samples per class for stratification
            class_counts = y.value_counts()
            can_stratify = all(count >= min_samples_needed for count in class_counts.values)
            
            # For small datasets or when stratification isn't possible, use cross-validation
            total_samples = len(y)
            use_cv = total_samples < 100 or not can_stratify
            
            if use_cv:
                # Use cross-validation for small datasets
                logger.info(
                    f"Using {3}-fold cross-validation for weight optimization "
                    f"(total_samples={total_samples}, can_stratify={can_stratify})"
                )
                
                weights = []
                kf = KFold(n_splits=min(3, total_samples // 10), shuffle=True, random_state=self.config.random_seed)
                
                for name, estimator in self.estimators_:
                    scores = []
                    for train_idx, val_idx in kf.split(X):
                        # Split data
                        X_train_cv = X.iloc[train_idx]
                        X_val_cv = X.iloc[val_idx]
                        y_train_cv = y.iloc[train_idx]
                        y_val_cv = y.iloc[val_idx]
                        
                        # Handle OpenVA data if available
                        if self.estimator_data_types_.get(name) == "openva" and X_openva is not None:
                            X_train_cv_openva = X_openva.iloc[train_idx]
                            X_val_cv_openva = X_openva.iloc[val_idx]
                            
                            estimator_clone = clone(estimator)
                            estimator_clone.fit(X_train_cv_openva, y_train_cv)
                            y_pred = estimator_clone.predict(X_val_cv_openva)
                        else:
                            estimator_clone = clone(estimator)
                            estimator_clone.fit(X_train_cv, y_train_cv)
                            y_pred = estimator_clone.predict(X_val_cv)
                        
                        # Calculate CSMF accuracy
                        from model_comparison.metrics.comparison_metrics import calculate_csmf_accuracy
                        csmf_acc = calculate_csmf_accuracy(y_val_cv, y_pred)
                        scores.append(csmf_acc)
                    
                    # Average score across folds
                    avg_score = np.mean(scores)
                    weights.append(max(avg_score, 0.1))  # Minimum weight of 0.1
                    logger.info(f"{name} CV score: {avg_score:.3f}")
                
            else:
                # Use holdout validation for larger datasets
                try:
                    # Try stratified split first
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=self.config.random_seed, stratify=y
                    )
                    logger.info("Using stratified holdout validation for weight optimization")
                except ValueError as e:
                    # Fall back to random split if stratification fails
                    logger.warning(f"Stratified split failed: {e}. Using random split.")
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=self.config.random_seed
                    )
                
                X_train_openva, X_val_openva = None, None
                if X_openva is not None:
                    # Split OpenVA data using same indices
                    train_idx = X_train.index
                    val_idx = X_val.index
                    X_train_openva = X_openva.loc[train_idx]
                    X_val_openva = X_openva.loc[val_idx]
                
                weights = []
                for name, estimator in self.estimators_:
                    # Refit on train split with appropriate data format
                    estimator_clone = clone(estimator)
                    
                    if self.estimator_data_types_.get(name) == "openva" and X_train_openva is not None:
                        estimator_clone.fit(X_train_openva, y_train)
                        y_pred = estimator_clone.predict(X_val_openva)
                    else:
                        estimator_clone.fit(X_train, y_train)
                        y_pred = estimator_clone.predict(X_val)
                    
                    # Calculate CSMF accuracy on validation
                    from model_comparison.metrics.comparison_metrics import calculate_csmf_accuracy
                    csmf_acc = calculate_csmf_accuracy(y_val, y_pred)
                    weights.append(max(csmf_acc, 0.1))  # Minimum weight of 0.1
                    logger.info(f"{name} holdout score: {csmf_acc:.3f}")
                    
            weights = np.array(weights)
            return weights / weights.sum()
            
        else:
            logger.warning(f"Unknown weight optimization strategy: {strategy}")
            return np.ones(n_estimators) / n_estimators
            
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
            X_openva: Optional[pd.DataFrame] = None) -> "DuckVotingEnsemble":
        """Fit the ensemble model with support for mixed data formats.
        
        Supports models requiring different data formats. InSilicoVA requires
        OpenVA encoding ("Y"/"."/"") while other models use numeric encoding.
        
        Args:
            X: Training features (numeric encoding for ML models)
            y: Training labels
            X_openva: Training features in OpenVA format for InSilicoVA (optional)
            
        Returns:
            Fitted ensemble instance
        """
        # Store feature names
        self.feature_names_ = list(X.columns)
        self.estimator_data_types_ = {}  # Track which data format each estimator uses
        
        # Ensure y is a Series for consistency
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # Get unique classes
        self.classes_ = np.unique(y.values)
        
        # Check minimum samples before fitting
        min_samples_per_class = 2
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < min_samples_per_class]
        
        if len(rare_classes) > 0:
            logger.warning(
                f"Found {len(rare_classes)} classes with < {min_samples_per_class} samples. "
                f"Filtering out rare classes to avoid fitting errors."
            )
            # Filter out rare classes
            valid_mask = ~y.isin(rare_classes.index)
            X_filtered = X[valid_mask]
            y_filtered = y[valid_mask]
            if X_openva is not None:
                X_openva_filtered = X_openva[valid_mask]
            else:
                X_openva_filtered = None
                
            # Update classes after filtering
            self.classes_ = np.unique(y_filtered.values)
            logger.info(f"After filtering: {len(X_filtered)} samples, {len(self.classes_)} classes")
        else:
            X_filtered = X
            y_filtered = y
            X_openva_filtered = X_openva
        
        # Create and fit estimators
        self.estimators_ = []
        for name, model_type in self.config.estimators:
            logger.info(f"Fitting {name} ({model_type}) estimator")
            
            if self.config.use_pretrained_estimators:
                # Load pretrained model (not implemented yet)
                raise NotImplementedError("Pretrained estimators not yet supported")
            else:
                try:
                    # Create and fit new estimator with appropriate data format
                    estimator = self._create_estimator(model_type)
                    
                    if model_type == "insilico" and X_openva_filtered is not None:
                        # InSilicoVA needs OpenVA format
                        estimator.fit(X_openva_filtered, y_filtered)
                        self.estimator_data_types_[name] = "openva"
                        logger.info(f"  Using OpenVA format for {name}")
                    else:
                        # Other models use numeric format
                        estimator.fit(X_filtered, y_filtered)
                        self.estimator_data_types_[name] = "numeric"
                        
                    self.estimators_.append((name, estimator))
                    
                except Exception as e:
                    logger.error(f"Failed to fit {name}: {str(e)}")
                    # Skip this estimator but continue with others
                    if len(self.config.estimators) > 1:
                        logger.warning(f"Continuing without {name} estimator")
                    else:
                        # If this was the only estimator, re-raise
                        raise
                        
        # Ensure we have at least one fitted estimator
        if len(self.estimators_) == 0:
            raise RuntimeError("Failed to fit any estimators in the ensemble")
                
        # Calculate diversity if needed
        if self.config.min_diversity > 0:
            # Use appropriate data format for predictions
            predictions = []
            for name, est in self.estimators_:
                if self.estimator_data_types_.get(name) == "openva" and X_openva_filtered is not None:
                    pred = est.predict(X_openva_filtered)
                else:
                    pred = est.predict(X_filtered)
                predictions.append(pred)
                
            diversity = self._calculate_diversity(
                predictions, 
                metric=self.config.diversity_metric
            )
            logger.info(f"Ensemble diversity: {diversity:.3f}")
            
            if diversity < self.config.min_diversity:
                logger.warning(
                    f"Ensemble diversity ({diversity:.3f}) below minimum "
                    f"({self.config.min_diversity})"
                )
                
        # Optimize weights using filtered data
        self.weights_ = self._optimize_weights(
            X_filtered, y_filtered, X_openva_filtered, strategy=self.config.weight_optimization
        )
        logger.info(f"Estimator weights: {self.weights_}")
        
        self._is_fitted = True
        return self
        
    def predict_proba(self, X: pd.DataFrame, X_openva: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate class probability predictions with support for mixed data formats.
        
        Args:
            X: Features to predict (numeric format)
            X_openva: Features in OpenVA format for InSilicoVA (optional)
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        check_is_fitted(self, ["estimators_", "classes_"])
        
        if self.config.voting == "soft":
            # Collect probability predictions from each estimator
            all_probs = []
            
            for i, (name, estimator) in enumerate(self.estimators_):
                # Get probabilities using appropriate data format
                if hasattr(self, 'estimator_data_types_') and \
                   self.estimator_data_types_.get(name) == "openva" and X_openva is not None:
                    proba = estimator.predict_proba(X_openva)
                else:
                    proba = estimator.predict_proba(X)
                
                # Align classes if necessary
                if hasattr(estimator, 'classes_'):
                    # Create mapping from estimator classes to ensemble classes
                    class_mapping = np.zeros((len(estimator.classes_), len(self.classes_)))
                    for j, cls in enumerate(estimator.classes_):
                        if cls in self.classes_:
                            k = np.where(self.classes_ == cls)[0][0]
                            class_mapping[j, k] = 1.0
                    
                    # Transform probabilities to ensemble class order
                    aligned_proba = proba @ class_mapping
                else:
                    # Assume classes are already aligned
                    aligned_proba = proba
                    
                # Apply weight
                weighted_proba = aligned_proba * self.weights_[i]
                all_probs.append(weighted_proba)
                
            # Sum weighted probabilities
            ensemble_proba = np.sum(all_probs, axis=0)
            
            # Normalize to sum to 1
            ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
            
        else:  # hard voting
            # Get predictions from each estimator
            all_preds = []
            for name, estimator in self.estimators_:
                # Use appropriate data format
                if hasattr(self, 'estimator_data_types_') and \
                   self.estimator_data_types_.get(name) == "openva" and X_openva is not None:
                    preds = estimator.predict(X_openva)
                else:
                    preds = estimator.predict(X)
                all_preds.append(preds)
                
            # Convert to class probabilities based on weighted votes
            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            ensemble_proba = np.zeros((n_samples, n_classes))
            
            for i in range(n_samples):
                for j, (pred, weight) in enumerate(zip(all_preds, self.weights_)):
                    # Find class index
                    if pred[i] in self.classes_:
                        class_idx = np.where(self.classes_ == pred[i])[0][0]
                        ensemble_proba[i, class_idx] += weight
                        
            # Normalize
            ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
            
        return ensemble_proba
        
    def predict(self, X: pd.DataFrame, X_openva: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate class predictions with support for mixed data formats.
        
        Args:
            X: Features to predict (numeric format)
            X_openva: Features in OpenVA format for InSilicoVA (optional)
            
        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X, X_openva)
        return self.classes_[np.argmax(proba, axis=1)]
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Args:
            deep: If True, return parameters for sub-estimators
            
        Returns:
            Parameter dictionary
        """
        params = {"config": self.config}
        if deep and self.config is not None:
            config_params = self.config.model_dump()
            for key, value in config_params.items():
                params[f"config__{key}"] = value
        return params
        
    def set_params(self, **params: Any) -> "DuckVotingEnsemble":
        """Set parameters for this estimator.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        if "config" in params:
            self.config = params.pop("config")
            
        # Handle nested parameters
        config_params = {}
        for key, value in list(params.items()):
            if key.startswith("config__"):
                config_key = key.replace("config__", "")
                config_params[config_key] = value
                params.pop(key)
                
        if config_params:
            current_config = self.config.model_dump()
            current_config.update(config_params)
            self.config = EnsembleConfig(**current_config)
            
        return self