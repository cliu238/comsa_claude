"""Cross-domain hyperparameter tuning for better generalization.

This module implements hyperparameter tuning strategies that optimize for
out-of-domain performance rather than just in-domain accuracy.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.logistic_regression_model import LogisticRegressionModel
from baseline.models.categorical_nb_model import CategoricalNBModel


logger = logging.getLogger(__name__)


class CrossDomainCV:
    """Cross-validation that ensures train/val come from different sites.
    
    This implementation creates CV folds where validation data comes from
    a different site than training data to measure generalization.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """Initialize cross-domain CV.
        
        Args:
            n_splits: Number of CV splits (ignored if fewer sites available)
            shuffle: Whether to shuffle sites
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X: pd.DataFrame, y: pd.Series, sites: pd.Series):
        """Generate cross-domain train/validation splits.
        
        Args:
            X: Features
            y: Labels
            sites: Site labels for each sample
            
        Yields:
            Train and validation indices
        """
        unique_sites = sites.unique()
        n_sites = len(unique_sites)
        
        if n_sites < 2:
            raise ValueError("Need at least 2 sites for cross-domain CV")
        
        # Shuffle sites if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            unique_sites = rng.permutation(unique_sites)
        
        # Create leave-one-site-out splits
        for val_site in unique_sites[:min(self.n_splits, n_sites)]:
            train_mask = sites != val_site
            val_mask = sites == val_site
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


def create_model(model_name: str, params: Dict[str, Any]) -> Any:
    """Create a model instance with given parameters.
    
    Args:
        model_name: Name of the model
        params: Model parameters
        
    Returns:
        Model instance
    """
    # Remove config__ prefix from params
    clean_params = {}
    for key, value in params.items():
        if key.startswith("config__"):
            clean_params[key.replace("config__", "")] = value
    
    if model_name == "xgboost":
        from baseline.models.xgboost_config import XGBoostConfig
        config = XGBoostConfig(**clean_params)
        return XGBoostModel(config=config)
    elif model_name == "random_forest":
        from baseline.models.random_forest_config import RandomForestConfig
        config = RandomForestConfig(**clean_params)
        return RandomForestModel(config=config)
    elif model_name == "logistic_regression":
        from baseline.models.logistic_regression_config import LogisticRegressionConfig
        config = LogisticRegressionConfig(**clean_params)
        return LogisticRegressionModel(config=config)
    elif model_name == "categorical_nb":
        from baseline.models.categorical_nb_config import CategoricalNBConfig
        config = CategoricalNBConfig(**clean_params)
        return CategoricalNBModel(config=config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_cross_domain_performance(
    model_name: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    sites: pd.Series,
    metric: str = "csmf_accuracy",
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate model performance using cross-domain CV.
    
    Args:
        model_name: Name of the model
        params: Model parameters to evaluate
        X: Features
        y: Labels
        sites: Site labels
        metric: Metric to optimize ("csmf_accuracy" or "cod_accuracy")
        n_splits: Number of CV splits
        
    Returns:
        Dictionary with mean and std of metric
    """
    cv = CrossDomainCV(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, sites)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create and train model
        model = create_model(model_name, params)
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if metric == "csmf_accuracy":
                score = model.calculate_csmf_accuracy(y_val, y_pred)
            elif metric == "cod_accuracy":
                score = (y_val == y_pred).mean()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
            logger.info(f"Fold {fold}: {metric} = {score:.4f}")
            
        except Exception as e:
            logger.warning(f"Error in fold {fold}: {e}")
            scores.append(0.0)  # Penalize failed models
    
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "scores": scores,
    }


def multi_objective_score(
    in_domain_score: float,
    out_domain_score: float,
    alpha: float = 0.7,
) -> float:
    """Combine in-domain and out-domain scores.
    
    Args:
        in_domain_score: Performance on same-site data
        out_domain_score: Performance on different-site data
        alpha: Weight for in-domain score (1-alpha for out-domain)
        
    Returns:
        Combined score
    """
    return alpha * in_domain_score + (1 - alpha) * out_domain_score


def evaluate_multi_objective_performance(
    model_name: str,
    params: Dict[str, Any],
    train_data: Tuple[pd.DataFrame, pd.Series],
    train_sites: pd.Series,
    test_data_by_site: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    metric: str = "csmf_accuracy",
    alpha: float = 0.7,
) -> float:
    """Evaluate model with multi-objective scoring.
    
    Args:
        model_name: Name of the model
        params: Model parameters
        train_data: Training data (X, y)
        train_sites: Site labels for training data
        test_data_by_site: Test data organized by site
        metric: Metric to evaluate
        alpha: Weight for in-domain performance
        
    Returns:
        Multi-objective score
    """
    X_train, y_train = train_data
    
    # Train model once
    model = create_model(model_name, params)
    model.fit(X_train, y_train)
    
    # Evaluate on each test site
    in_domain_scores = []
    out_domain_scores = []
    
    train_site_set = set(train_sites.unique())
    
    for test_site, (X_test, y_test) in test_data_by_site.items():
        y_pred = model.predict(X_test)
        
        if metric == "csmf_accuracy":
            score = model.calculate_csmf_accuracy(y_test, y_pred)
        else:
            score = (y_test == y_pred).mean()
        
        # Classify as in-domain or out-domain
        if test_site in train_site_set:
            in_domain_scores.append(score)
        else:
            out_domain_scores.append(score)
    
    # Calculate multi-objective score
    in_domain_avg = np.mean(in_domain_scores) if in_domain_scores else 0.0
    out_domain_avg = np.mean(out_domain_scores) if out_domain_scores else 0.0
    
    return multi_objective_score(in_domain_avg, out_domain_avg, alpha)