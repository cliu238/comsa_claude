"""Factory for creating VA models with custom configurations."""

from typing import Any, Dict, Union

from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.random_forest_config import RandomForestConfig
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.logistic_regression_config import LogisticRegressionConfig
from baseline.models.logistic_regression_model import LogisticRegressionModel


def create_model(
    model_name: str,
    config: Union[Dict[str, Any], XGBoostConfig, RandomForestConfig, LogisticRegressionConfig, None] = None
) -> Union[XGBoostModel, RandomForestModel, LogisticRegressionModel]:
    """Create a model instance with the given configuration.
    
    Args:
        model_name: Name of the model ('xgboost', 'random_forest', 'logistic_regression')
        config: Model configuration (dict, config object, or None for defaults)
        
    Returns:
        Configured model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "xgboost":
        if config is None:
            return XGBoostModel()
        elif isinstance(config, dict):
            # Create config from dict, merging with defaults
            base_config = XGBoostConfig()
            config_dict = base_config.model_dump()
            config_dict.update(config)
            return XGBoostModel(config=XGBoostConfig(**config_dict))
        elif isinstance(config, XGBoostConfig):
            return XGBoostModel(config=config)
        else:
            raise TypeError(f"Invalid config type for xgboost: {type(config)}")
            
    elif model_name == "random_forest":
        if config is None:
            return RandomForestModel()
        elif isinstance(config, dict):
            # Create config from dict, merging with defaults
            base_config = RandomForestConfig()
            config_dict = base_config.model_dump()
            config_dict.update(config)
            return RandomForestModel(config=RandomForestConfig(**config_dict))
        elif isinstance(config, RandomForestConfig):
            return RandomForestModel(config=config)
        else:
            raise TypeError(f"Invalid config type for random_forest: {type(config)}")
            
    elif model_name == "logistic_regression":
        if config is None:
            return LogisticRegressionModel()
        elif isinstance(config, dict):
            # Create config from dict, merging with defaults
            base_config = LogisticRegressionConfig()
            config_dict = base_config.model_dump()
            # Special handling for logistic regression parameters
            if "penalty" in config:
                config_dict["penalty"] = config["penalty"]
                # Remove l1_ratio if not using elasticnet
                if config["penalty"] != "elasticnet" and "l1_ratio" in config_dict:
                    config_dict.pop("l1_ratio", None)
            config_dict.update(config)
            return LogisticRegressionModel(config=LogisticRegressionConfig(**config_dict))
        elif isinstance(config, LogisticRegressionConfig):
            return LogisticRegressionModel(config=config)
        else:
            raise TypeError(f"Invalid config type for logistic_regression: {type(config)}")
            
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: xgboost, random_forest, logistic_regression"
        )


def get_default_config(model_name: str) -> Union[XGBoostConfig, RandomForestConfig, LogisticRegressionConfig]:
    """Get the default configuration for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Default configuration object
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "xgboost":
        return XGBoostConfig()
    elif model_name == "random_forest":
        return RandomForestConfig()
    elif model_name == "logistic_regression":
        return LogisticRegressionConfig()
    else:
        raise ValueError(f"Unknown model: {model_name}")