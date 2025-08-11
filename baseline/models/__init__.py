"""Model implementations for VA cause-of-death prediction.

This package contains model implementations for VA analysis, including
InSilicoVA and ML model implementations like XGBoost.
"""

from baseline.models.hyperparameter_tuning import (
    XGBoostHyperparameterTuner,
    quick_tune_xgboost,
    CategoricalNBHyperparameterTuner,
    quick_tune_categorical_nb,
)
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.logistic_regression_config import LogisticRegressionConfig
from baseline.models.logistic_regression_model import LogisticRegressionModel
from baseline.models.model_config import InSilicoVAConfig
from baseline.models.model_validator import InSilicoVAValidator, ModelValidationResult
from baseline.models.random_forest_config import RandomForestConfig
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.categorical_nb_config import CategoricalNBConfig
from baseline.models.categorical_nb_model import CategoricalNBModel
from baseline.models.ensemble_config import EnsembleConfig
from baseline.models.ensemble_model import DuckVotingEnsemble
from baseline.models.tabicl_config import TabICLConfig
from baseline.models.tabicl_model import TabICLModel

__all__ = [
    "InSilicoVAModel",
    "InSilicoVAConfig",
    "InSilicoVAValidator",
    "ModelValidationResult",
    "XGBoostModel",
    "XGBoostConfig",
    "XGBoostHyperparameterTuner",
    "quick_tune_xgboost",
    "RandomForestModel",
    "RandomForestConfig",
    "LogisticRegressionModel",
    "LogisticRegressionConfig",
    "CategoricalNBModel",
    "CategoricalNBConfig",
    "CategoricalNBHyperparameterTuner",
    "quick_tune_categorical_nb",
    "DuckVotingEnsemble",
    "EnsembleConfig",
    "TabICLModel",
    "TabICLConfig",
]