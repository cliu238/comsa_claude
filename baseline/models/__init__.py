"""Model implementations for VA cause-of-death prediction.

This package contains model implementations for VA analysis, including
InSilicoVA and ML model implementations like XGBoost.
"""

from baseline.models.hyperparameter_tuning import (
    XGBoostHyperparameterTuner,
    quick_tune_xgboost,
)
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.model_config import InSilicoVAConfig
from baseline.models.model_validator import InSilicoVAValidator, ModelValidationResult
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_prior_config import XGBoostPriorConfig
from baseline.models.xgboost_prior_enhanced import XGBoostPriorEnhanced

__all__ = [
    "InSilicoVAModel",
    "InSilicoVAConfig",
    "InSilicoVAValidator",
    "ModelValidationResult",
    "XGBoostModel",
    "XGBoostConfig",
    "XGBoostPriorEnhanced",
    "XGBoostPriorConfig",
    "XGBoostHyperparameterTuner",
    "quick_tune_xgboost",
]