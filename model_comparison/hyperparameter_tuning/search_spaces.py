"""Hyperparameter search spaces for VA models.

This module defines the search spaces for hyperparameter tuning of each model type.
The search spaces are designed based on best practices and the specific characteristics
of verbal autopsy data.
"""

from typing import Any, Dict

from ray import tune


def get_xgboost_search_space() -> Dict[str, Any]:
    """Get XGBoost hyperparameter search space.
    
    The search space is designed for multi-class classification with
    a focus on preventing overfitting on VA data.
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # Tree-specific parameters
        'config__max_depth': tune.choice([3, 5, 7, 10]),
        'config__learning_rate': tune.loguniform(0.01, 0.3),
        'config__n_estimators': tune.choice([100, 200, 500]),
        
        # Sampling parameters
        'config__subsample': tune.uniform(0.7, 1.0),
        'config__colsample_bytree': tune.uniform(0.7, 1.0),
        
        # Regularization parameters
        'config__reg_alpha': tune.loguniform(1e-4, 1.0),
        'config__reg_lambda': tune.loguniform(1.0, 10.0),
    }


def get_random_forest_search_space() -> Dict[str, Any]:
    """Get Random Forest hyperparameter search space.
    
    The search space emphasizes ensemble diversity and feature sampling
    strategies suitable for high-dimensional VA data.
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # Tree ensemble parameters
        'config__n_estimators': tune.choice([100, 200, 500]),
        'config__max_depth': tune.choice([None, 10, 20, 30]),
        
        # Node splitting parameters
        'config__min_samples_split': tune.choice([2, 5, 10]),
        'config__min_samples_leaf': tune.choice([1, 2, 4]),
        
        # Feature sampling parameters
        'config__max_features': tune.choice(['sqrt', 'log2', 0.5]),
        
        # Bootstrap parameter
        'config__bootstrap': tune.choice([True, False]),
    }


def get_logistic_regression_search_space() -> Dict[str, Any]:
    """Get Logistic Regression hyperparameter search space.
    
    The search space covers different regularization strategies and
    strengths suitable for multi-class classification.
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # Regularization parameters
        'config__C': tune.loguniform(0.001, 100.0),
        'config__penalty': tune.choice(['l1', 'l2', 'elasticnet']),
        
        # Solver is fixed to 'saga' which supports all penalty types
        'config__solver': 'saga',
        
        # L1 ratio for elasticnet (will be filtered if not needed)
        'config__l1_ratio': tune.uniform(0.15, 0.85),
        
        # Optimization parameters
        'config__max_iter': tune.choice([1000, 2000]),
    }


def get_categorical_nb_search_space() -> Dict[str, Any]:
    """Get CategoricalNB hyperparameter search space.
    
    The search space is designed for categorical features common in VA data,
    with focus on smoothing parameters to handle sparse categories.
    
    Returns:
        Dictionary mapping parameter names to Ray Tune search spaces
    """
    return {
        # Smoothing parameter (Laplace/Lidstone)
        'config__alpha': tune.choice([0.001, 0.01, 0.1, 0.5, 1.0, 2.0]),
        
        # Whether to use the same alpha for all features
        'config__force_alpha': tune.choice([True, False]),
        
        # Whether to learn class priors
        'config__fit_prior': tune.choice([True, False]),
    }


def filter_params_for_model(params: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Filter parameters based on model-specific requirements.
    
    Some parameters are conditional and should only be used with
    specific configurations.
    
    Args:
        params: Dictionary of parameters
        model_name: Name of the model ('xgboost', 'random_forest', 'logistic_regression', 'categorical_nb')
        
    Returns:
        Filtered parameter dictionary
    """
    filtered_params = params.copy()
    
    if model_name == "logistic_regression":
        # Remove l1_ratio if not using elasticnet penalty
        if filtered_params.get("config__penalty") != "elasticnet":
            filtered_params.pop("config__l1_ratio", None)
    
    # CategoricalNB doesn't need special filtering, all parameters are always valid
    
    return filtered_params


def get_search_space_for_model(model_name: str) -> Dict[str, Any]:
    """Get the appropriate search space for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Search space dictionary
        
    Raises:
        ValueError: If model_name is not recognized
    """
    search_spaces = {
        "xgboost": get_xgboost_search_space,
        "random_forest": get_random_forest_search_space,
        "logistic_regression": get_logistic_regression_search_space,
        "categorical_nb": get_categorical_nb_search_space,
    }
    
    if model_name not in search_spaces:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Valid models are: {list(search_spaces.keys())}"
        )
    
    return search_spaces[model_name]()