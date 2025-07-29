"""Ray Tune integration for hyperparameter optimization.

This module provides the RayTuner class which handles distributed hyperparameter
tuning using Ray Tune, integrated with the existing VA model comparison framework.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from ray import tune
from ray.tune import CheckpointConfig, RunConfig, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from baseline.models.logistic_regression_model import LogisticRegressionModel
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.categorical_nb_model import CategoricalNBModel
from baseline.utils import get_logger
from model_comparison.hyperparameter_tuning.search_spaces import (
    filter_params_for_model,
)

logger = get_logger(__name__, component="hyperparameter_tuning")


class RayTuner:
    """Ray Tune integration for hyperparameter optimization.
    
    This class provides distributed hyperparameter tuning capabilities
    using Ray Tune's efficient search algorithms and schedulers.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        n_cpus_per_trial: float = 1.0,
        max_concurrent_trials: Optional[int] = None,
        search_algorithm: str = "bayesian",
        metric: str = "csmf_accuracy",
        mode: str = "max",
    ):
        """Initialize Ray Tuner.
        
        Args:
            n_trials: Number of hyperparameter combinations to try
            n_cpus_per_trial: CPUs allocated per trial
            max_concurrent_trials: Max trials running in parallel
            search_algorithm: "grid", "random", or "bayesian"
            metric: Metric to optimize
            mode: "min" or "max"
        """
        self.n_trials = n_trials
        self.n_cpus_per_trial = n_cpus_per_trial
        self.max_concurrent_trials = max_concurrent_trials
        self.search_algorithm = search_algorithm
        self.metric = metric
        self.mode = mode
        
        # Initialize scheduler for early stopping
        self.scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=10,  # Maximum iterations per trial
            grace_period=3,  # Minimum iterations before stopping
            reduction_factor=3,
        )
        
    def tune_model(
        self,
        model_name: str,
        search_space: Dict[str, Any],
        train_data: Tuple[pd.DataFrame, pd.Series],
        val_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        cv_folds: int = 5,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter tuning for a model.
        
        Args:
            model_name: Name of the model ('xgboost', 'random_forest', 'logistic_regression')
            search_space: Hyperparameter search space
            train_data: Training data (X, y)
            val_data: Optional validation data
            cv_folds: Number of CV folds if val_data not provided
            experiment_name: Optional name for the experiment
            
        Returns:
            Best hyperparameters and performance metrics
        """
        X_train, y_train = train_data
        
        # Get model class
        model_classes = {
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "logistic_regression": LogisticRegressionModel,
            "categorical_nb": CategoricalNBModel,
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = model_classes[model_name]
        
        # Define objective function
        def objective(config: Dict[str, Any]) -> None:
            """Objective function for Ray Tune."""
            try:
                # Filter parameters for the specific model
                filtered_config = filter_params_for_model(config, model_name)
                
                # Create model with config
                model = model_class()
                try:
                    # Keep the config__ prefix for nested parameters
                    model.set_params(**filtered_config)
                except Exception as param_error:
                    logger.error(f"Failed to set params for {model_name}: {param_error}")
                    logger.error(f"Attempted params: {filtered_config}")
                    raise
                
                if val_data is not None:
                    # Use validation set
                    X_val, y_val = val_data
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    csmf_acc = model.calculate_csmf_accuracy(y_val, y_pred)
                    cod_acc = (y_val == y_pred).mean()
                else:
                    # Use cross-validation
                    try:
                        cv_results = model.cross_validate(
                            X_train, y_train, cv=cv_folds, stratified=True
                        )
                        csmf_acc = cv_results["csmf_accuracy_mean"]
                        cod_acc = cv_results["cod_accuracy_mean"]
                    except (ValueError, IndexError) as e:
                        # Handle cases where stratified CV fails or data issues
                        logger.warning(f"CV failed with stratified=True: {e}. Trying non-stratified.")
                        try:
                            cv_results = model.cross_validate(
                                X_train, y_train, cv=cv_folds, stratified=False
                            )
                            csmf_acc = cv_results["csmf_accuracy_mean"]
                            cod_acc = cv_results["cod_accuracy_mean"]
                        except Exception as e2:
                            logger.error(f"CV failed completely: {e2}")
                            csmf_acc = 0.0
                            cod_acc = 0.0
                
                # Report results to Ray Tune
                tune.report({
                    "csmf_accuracy": csmf_acc,
                    "cod_accuracy": cod_acc,
                })
                
            except Exception as e:
                logger.error(f"Error in trial: {e}")
                # Report poor metrics on error
                tune.report({
                    "csmf_accuracy": 0.0,
                    "cod_accuracy": 0.0,
                })
        
        # Configure search algorithm
        search_alg = None
        if self.search_algorithm == "bayesian":
            try:
                search_alg = OptunaSearch(
                    metric=self.metric,
                    mode=self.mode,
                )
            except ImportError:
                logger.warning(
                    "Optuna not available. Falling back to random search. "
                    "Install with: pip install optuna"
                )
        
        # Configure resources
        resources = {"cpu": self.n_cpus_per_trial}
        
        # Run tuning
        tuner = tune.Tuner(
            tune.with_resources(objective, resources=resources),
            param_space=search_space,
            tune_config=TuneConfig(
                num_samples=self.n_trials,
                scheduler=self.scheduler,
                search_alg=search_alg,
                max_concurrent_trials=self.max_concurrent_trials,
            ),
            run_config=RunConfig(
                name=experiment_name or f"{model_name}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                storage_path=os.path.abspath("./ray_results"),
                verbose=0,
            ),
        )
        
        logger.info(
            f"Starting hyperparameter tuning for {model_name} "
            f"with {self.n_trials} trials"
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric=self.metric, mode=self.mode)
        
        # Filter the best config
        best_config = filter_params_for_model(best_result.config, model_name)
        
        # Prepare return dictionary
        return_dict = {
            "best_params": best_config,
            "best_score": best_result.metrics[self.metric],
            "metrics": {
                "csmf_accuracy": best_result.metrics.get("csmf_accuracy", 0.0),
                "cod_accuracy": best_result.metrics.get("cod_accuracy", 0.0),
            },
            "all_results": results.get_dataframe(),
            "model_name": model_name,
            "n_trials_completed": len(results),
        }
        
        logger.info(
            f"Tuning completed for {model_name}. "
            f"Best {self.metric}: {return_dict['best_score']:.4f}"
        )
        
        return return_dict
    
    def save_results(
        self, results: Dict[str, Any], output_path: Path
    ) -> None:
        """Save tuning results to disk.
        
        Args:
            results: Tuning results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters as JSON
        best_params_path = output_path.with_suffix(".json")
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "model_name": results["model_name"],
                    "best_params": results["best_params"],
                    "best_score": results["best_score"],
                    "metrics": results["metrics"],
                    "n_trials": results["n_trials_completed"],
                },
                f,
                indent=2,
            )
        
        # Save all results as CSV
        if "all_results" in results and results["all_results"] is not None:
            csv_path = output_path.with_suffix(".csv")
            results["all_results"].to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {best_params_path}")


def quick_tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    metric: str = "csmf_accuracy",
) -> Dict[str, Any]:
    """Quick function to tune a model's hyperparameters.
    
    Args:
        model_name: Name of the model to tune
        X: Training features
        y: Training labels
        n_trials: Number of optimization trials
        metric: Metric to optimize
        
    Returns:
        Dictionary with best parameters and trained model
    """
    from model_comparison.hyperparameter_tuning.search_spaces import (
        get_search_space_for_model,
    )
    
    # Get search space
    search_space = get_search_space_for_model(model_name)
    
    # Create tuner
    tuner = RayTuner(n_trials=n_trials, metric=metric)
    
    # Run tuning
    results = tuner.tune_model(
        model_name=model_name,
        search_space=search_space,
        train_data=(X, y),
    )
    
    # Get model class
    model_classes = {
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "categorical_nb": CategoricalNBModel,
    }
    
    # Train final model with best parameters
    model = model_classes[model_name]()
    model.set_params(**results["best_params"])
    model.fit(X, y)
    
    results["trained_model"] = model
    
    logger.info(
        f"Model tuned and trained with best parameters. "
        f"Best {metric}: {results['best_score']:.4f}"
    )
    
    return results