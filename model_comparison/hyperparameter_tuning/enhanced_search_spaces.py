"""Enhanced hyperparameter search spaces for better generalization.

This module provides improved search spaces that prioritize regularization
and generalization over in-domain performance.
"""

from typing import Any, Dict

from ray import tune


def get_xgboost_enhanced_search_space() -> Dict[str, Any]:
    """Get enhanced XGBoost search space with stronger regularization.
    
    This search space is designed to combat overfitting in high-dimensional
    VA data by:
    1. Limiting tree depth more aggressively
    2. Using stronger L1/L2 regularization
    3. Adding gamma for pruning control
    4. Including min_child_weight to prevent overfitting to small samples
    5. More aggressive feature and sample subsampling
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # Tree complexity control - shallower trees
        'config__max_depth': tune.choice([2, 3, 4, 5, 6]),
        'config__min_child_weight': tune.choice([5, 10, 20, 50, 100]),
        'config__gamma': tune.loguniform(0.1, 10.0),  # Minimum loss reduction for split
        
        # Learning rate - favor smaller rates for better generalization
        'config__learning_rate': tune.loguniform(0.005, 0.1),
        'config__n_estimators': tune.choice([300, 500, 800, 1000]),
        
        # Aggressive subsampling to reduce overfitting
        'config__subsample': tune.uniform(0.4, 0.7),
        'config__colsample_bytree': tune.uniform(0.3, 0.6),
        'config__colsample_bylevel': tune.uniform(0.3, 0.6),
        'config__colsample_bynode': tune.uniform(0.3, 0.6),
        
        # Strong regularization
        'config__reg_alpha': tune.loguniform(1.0, 100.0),  # L1 regularization
        'config__reg_lambda': tune.loguniform(10.0, 100.0),  # L2 regularization
    }


def get_xgboost_conservative_search_space() -> Dict[str, Any]:
    """Get conservative XGBoost search space for maximum regularization.
    
    This is an even more conservative space for extreme overfitting cases.
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # Very shallow trees
        'config__max_depth': tune.choice([2, 3, 4]),
        'config__min_child_weight': tune.choice([20, 50, 100, 200]),
        'config__gamma': tune.loguniform(1.0, 20.0),
        
        # Very low learning rate
        'config__learning_rate': tune.loguniform(0.001, 0.05),
        'config__n_estimators': tune.choice([500, 1000, 1500]),
        
        # Heavy subsampling
        'config__subsample': tune.uniform(0.3, 0.5),
        'config__colsample_bytree': tune.uniform(0.2, 0.4),
        
        # Very strong regularization
        'config__reg_alpha': tune.loguniform(10.0, 1000.0),
        'config__reg_lambda': tune.loguniform(50.0, 500.0),
    }


def get_random_forest_enhanced_search_space() -> Dict[str, Any]:
    """Get enhanced Random Forest search space for better generalization.
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # More trees for stability
        'config__n_estimators': tune.choice([300, 500, 1000]),
        
        # Limit tree depth
        'config__max_depth': tune.choice([5, 10, 15, 20]),
        
        # Increase minimum samples requirements
        'config__min_samples_split': tune.choice([10, 20, 50]),
        'config__min_samples_leaf': tune.choice([5, 10, 20]),
        
        # Conservative feature sampling
        'config__max_features': tune.choice(['sqrt', 'log2', 0.3, 0.5]),
        
        # Always use bootstrap for better generalization
        'config__bootstrap': True,
        
        # Add some randomness
        'config__max_samples': tune.uniform(0.5, 0.8),  # Subsample training data
    }


def get_search_space_for_model_enhanced(model_name: str) -> Dict[str, Any]:
    """Get enhanced search space for a model focused on generalization.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Search space dictionary
        
    Raises:
        ValueError: If model_name is not recognized
    """
    enhanced_search_spaces = {
        "xgboost": get_xgboost_enhanced_search_space,
        "xgboost_conservative": get_xgboost_conservative_search_space,
        "random_forest": get_random_forest_enhanced_search_space,
        # Keep original spaces for other models
        "logistic_regression": None,
        "categorical_nb": None,
    }
    
    if model_name not in enhanced_search_spaces:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Valid models are: {list(enhanced_search_spaces.keys())}"
        )
    
    space_func = enhanced_search_spaces[model_name]
    if space_func is None:
        # Fall back to original search space
        from model_comparison.hyperparameter_tuning.search_spaces import get_search_space_for_model
        return get_search_space_for_model(model_name)
    
    return space_func()


def get_adaptive_search_space(model_name: str, overfitting_score: float) -> Dict[str, Any]:
    """Get search space adapted to the level of overfitting observed.
    
    Args:
        model_name: Name of the model
        overfitting_score: Score from 0 (no overfitting) to 1 (severe overfitting)
        
    Returns:
        Adapted search space
    """
    if model_name != "xgboost":
        return get_search_space_for_model_enhanced(model_name)
    
    if overfitting_score > 0.7:
        # Severe overfitting - use conservative space
        return get_xgboost_conservative_search_space()
    elif overfitting_score > 0.4:
        # Moderate overfitting - use enhanced space
        return get_xgboost_enhanced_search_space()
    else:
        # Low overfitting - use original space
        from model_comparison.hyperparameter_tuning.search_spaces import get_xgboost_search_space
        return get_xgboost_search_space()