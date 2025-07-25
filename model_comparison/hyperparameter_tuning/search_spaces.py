"""Define search spaces for hyperparameter tuning of all models."""

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class SearchSpace(BaseModel):
    """Base class for hyperparameter search spaces."""
    
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., pattern="^(int|float|categorical)$", description="Parameter type")
    values: Union[List[Any], Dict[str, Any]] = Field(
        ..., description="List for categorical, dict with 'low', 'high' for numeric"
    )
    log_scale: bool = Field(default=False, description="Use log scale for numeric parameters")
    
    model_config = {"validate_assignment": True}


class ModelSearchSpace(BaseModel):
    """Complete search space for a model."""
    
    model_name: str = Field(..., description="Name of the model")
    parameters: Dict[str, SearchSpace] = Field(..., description="Parameter search spaces")
    
    model_config = {"validate_assignment": True}


def get_search_space(model_name: str) -> ModelSearchSpace:
    """Get search space for a specific model.
    
    Args:
        model_name: Name of the model ('xgboost', 'random_forest', 'logistic_regression')
        
    Returns:
        ModelSearchSpace with parameter definitions
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "xgboost":
        return ModelSearchSpace(
            model_name="xgboost",
            parameters={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    values={"low": 100, "high": 500}
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    values={"low": 3, "high": 12}
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="float",
                    values={"low": 0.01, "high": 0.3},
                    log_scale=True
                ),
                "subsample": SearchSpace(
                    name="subsample",
                    type="float",
                    values={"low": 0.5, "high": 1.0}
                ),
                "colsample_bytree": SearchSpace(
                    name="colsample_bytree",
                    type="float",
                    values={"low": 0.5, "high": 1.0}
                ),
                "reg_alpha": SearchSpace(
                    name="reg_alpha",
                    type="float",
                    values={"low": 1e-4, "high": 10.0},
                    log_scale=True
                ),
                "reg_lambda": SearchSpace(
                    name="reg_lambda",
                    type="float",
                    values={"low": 1e-4, "high": 10.0},
                    log_scale=True
                ),
            }
        )
    elif model_name == "random_forest":
        return ModelSearchSpace(
            model_name="random_forest",
            parameters={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    values={"low": 100, "high": 500}
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    values={"low": 5, "high": 20}
                ),
                "min_samples_split": SearchSpace(
                    name="min_samples_split",
                    type="int",
                    values={"low": 2, "high": 20}
                ),
                "min_samples_leaf": SearchSpace(
                    name="min_samples_leaf",
                    type="int",
                    values={"low": 1, "high": 10}
                ),
                "max_features": SearchSpace(
                    name="max_features",
                    type="categorical",
                    values=["sqrt", "log2", 0.5, 0.7, 0.9]
                ),
                "bootstrap": SearchSpace(
                    name="bootstrap",
                    type="categorical",
                    values=[True, False]
                ),
                "criterion": SearchSpace(
                    name="criterion",
                    type="categorical",
                    values=["gini", "entropy"]
                ),
            }
        )
    elif model_name == "logistic_regression":
        return ModelSearchSpace(
            model_name="logistic_regression",
            parameters={
                "C": SearchSpace(
                    name="C",
                    type="float",
                    values={"low": 1e-4, "high": 100.0},
                    log_scale=True
                ),
                "penalty": SearchSpace(
                    name="penalty",
                    type="categorical",
                    values=["l1", "l2", "elasticnet"]
                ),
                "solver": SearchSpace(
                    name="solver",
                    type="categorical",
                    values=["saga"]  # SAGA supports all penalties
                ),
                "l1_ratio": SearchSpace(
                    name="l1_ratio",
                    type="float",
                    values={"low": 0.0, "high": 1.0}
                ),
                "max_iter": SearchSpace(
                    name="max_iter",
                    type="int",
                    values={"low": 100, "high": 1000}
                ),
            }
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: xgboost, random_forest, logistic_regression"
        )


def validate_search_space_compatibility(model_name: str, search_space: ModelSearchSpace) -> None:
    """Validate that search space is compatible with model constraints.
    
    Args:
        model_name: Name of the model
        search_space: Search space to validate
        
    Raises:
        ValueError: If search space contains incompatible parameter combinations
    """
    if model_name == "logistic_regression":
        # Check if elasticnet is in penalty values
        params = search_space.parameters
        if "penalty" in params and "l1_ratio" in params:
            penalty_values = params["penalty"].values
            if isinstance(penalty_values, list) and "elasticnet" in penalty_values:
                # Ensure l1_ratio is included when elasticnet is possible
                l1_ratio_range = params["l1_ratio"].values
                if isinstance(l1_ratio_range, dict):
                    if l1_ratio_range.get("low", 0) == 0 and l1_ratio_range.get("high", 1) == 0:
                        raise ValueError(
                            "l1_ratio must have non-zero range when elasticnet penalty is possible"
                        )


def get_grid_search_space(model_name: str, grid_size: str = "small") -> ModelSearchSpace:
    """Get a grid search space with discrete values for each parameter.
    
    Args:
        model_name: Name of the model
        grid_size: Size of the grid ('small', 'medium', 'large')
        
    Returns:
        ModelSearchSpace with discrete values for grid search
    """
    base_space = get_search_space(model_name)
    
    # Define grid points based on size
    grid_points = {
        "small": 3,
        "medium": 5,
        "large": 7
    }
    n_points = grid_points.get(grid_size, 3)
    
    # Convert continuous parameters to discrete grids
    grid_params = {}
    for param_name, param_space in base_space.parameters.items():
        if param_space.type in ["int", "float"]:
            if isinstance(param_space.values, dict):
                low = param_space.values["low"]
                high = param_space.values["high"]
                
                if param_space.type == "int":
                    # Create evenly spaced integers
                    import numpy as np
                    values = np.linspace(low, high, n_points).astype(int).tolist()
                    # Remove duplicates while preserving order
                    values = list(dict.fromkeys(values))
                else:
                    # Create evenly spaced floats (in log space if needed)
                    import numpy as np
                    if param_space.log_scale:
                        values = np.logspace(np.log10(low), np.log10(high), n_points).tolist()
                    else:
                        values = np.linspace(low, high, n_points).tolist()
                
                grid_params[param_name] = SearchSpace(
                    name=param_name,
                    type="categorical",  # Convert to categorical for grid search
                    values=values
                )
            else:
                # Already categorical
                grid_params[param_name] = param_space
        else:
            # Keep categorical as is
            grid_params[param_name] = param_space
    
    return ModelSearchSpace(
        model_name=base_space.model_name,
        parameters=grid_params
    )