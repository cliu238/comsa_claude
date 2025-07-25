"""Hyperparameter tuning module for VA model comparison.

This module provides distributed hyperparameter optimization using Ray Tune,
integrated with the existing Ray-based model comparison framework.
"""

from model_comparison.hyperparameter_tuning.ray_tuner import RayTuner
from model_comparison.hyperparameter_tuning.search_spaces import (
    get_logistic_regression_search_space,
    get_random_forest_search_space,
    get_xgboost_search_space,
)

__all__ = [
    "RayTuner",
    "get_xgboost_search_space",
    "get_random_forest_search_space",
    "get_logistic_regression_search_space",
]