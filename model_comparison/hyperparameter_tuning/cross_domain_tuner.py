"""Cross-domain hyperparameter tuner using Ray Tune.

This module provides a specialized tuner that optimizes hyperparameters
for cross-domain generalization using leave-one-site-out cross-validation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from sklearn.model_selection import StratifiedKFold

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.logistic_regression_model import LogisticRegressionModel
from baseline.models.categorical_nb_model import CategoricalNBModel
from model_comparison.experiments.cross_domain_tuning import CrossDomainCV, create_model


logger = logging.getLogger(__name__)


class CrossDomainTuner:
    """Ray Tune-based hyperparameter tuner optimized for cross-domain generalization.
    
    This tuner uses leave-one-site-out cross-validation to find hyperparameters
    that generalize well across different data collection sites.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        search_algorithm: str = "bayesian",
        metric: str = "csmf_accuracy",
        mode: str = "max",
        n_cpus_per_trial: float = 1.0,
        max_concurrent_trials: Optional[int] = None,
        multi_objective_alpha: float = 0.3,  # Weight more towards out-domain
    ):
        """Initialize the cross-domain tuner.
        
        Args:
            n_trials: Number of hyperparameter configurations to try
            search_algorithm: Algorithm for hyperparameter search ("bayesian", "optuna", "random")
            metric: Metric to optimize
            mode: Optimization mode ("max" or "min")
            n_cpus_per_trial: CPUs allocated per trial
            max_concurrent_trials: Maximum concurrent trials
            multi_objective_alpha: Weight for in-domain performance (1-alpha for out-domain)
        """
        self.n_trials = n_trials
        self.search_algorithm = search_algorithm
        self.metric = metric
        self.mode = mode
        self.n_cpus_per_trial = n_cpus_per_trial
        self.max_concurrent_trials = max_concurrent_trials
        self.multi_objective_alpha = multi_objective_alpha
        
    def _create_search_algorithm(self):
        """Create the search algorithm for Ray Tune."""
        if self.search_algorithm == "bayesian":
            base_search = BayesOptSearch(
                metric=self.metric,
                mode=self.mode,
                random_state=42,
            )
        elif self.search_algorithm == "optuna":
            base_search = OptunaSearch(
                metric=self.metric,
                mode=self.mode,
                seed=42,
            )
        else:  # random
            base_search = None
            
        # Apply concurrency limiter if specified
        if base_search and self.max_concurrent_trials:
            return ConcurrencyLimiter(
                base_search,
                max_concurrent=self.max_concurrent_trials,
            )
        return base_search
    
    def tune_model(
        self,
        model_name: str,
        search_space: Dict[str, Any],
        train_data: Tuple[pd.DataFrame, pd.Series],
        train_sites: pd.Series,
        cv_folds: int = 5,
        experiment_name: str = "cross_domain_tuning",
    ) -> Dict[str, Any]:
        """Tune hyperparameters using cross-domain validation.
        
        Args:
            model_name: Name of the model to tune
            search_space: Ray Tune search space
            train_data: Training data (X, y)
            train_sites: Site labels for training data
            cv_folds: Number of cross-validation folds
            experiment_name: Name for the Ray Tune experiment
            
        Returns:
            Dictionary with best parameters and performance metrics
        """
        X_train, y_train = train_data
        
        # Define the training function for Ray Tune
        def train_fn(config):
            """Training function for a single hyperparameter configuration."""
            # Use cross-domain CV to evaluate this configuration
            cv = CrossDomainCV(n_splits=cv_folds, shuffle=True, random_state=42)
            in_domain_scores = []
            out_domain_scores = []
            
            # Also do regular stratified CV for in-domain performance
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Evaluate out-domain performance
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, train_sites)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Create model with current config
                model = create_model(model_name, config)
                
                try:
                    # Preprocess if needed (for non-InSilico models)
                    if model_name != "insilico":
                        if model_name == "categorical_nb":
                            # CategoricalNB handles its own preprocessing
                            model.fit(X_fold_train, y_fold_train)
                            y_pred = model.predict(X_fold_val)
                        else:
                            # Other models need numeric encoding
                            from model_comparison.orchestration.ray_tasks import _preprocess_features
                            X_fold_train_processed = _preprocess_features(X_fold_train)
                            X_fold_val_processed = _preprocess_features(X_fold_val)
                            model.fit(X_fold_train_processed, y_fold_train)
                            y_pred = model.predict(X_fold_val_processed)
                    else:
                        model.fit(X_fold_train, y_fold_train)
                        y_pred = model.predict(X_fold_val)
                    
                    # Calculate metric
                    if self.metric == "csmf_accuracy":
                        if hasattr(model, 'calculate_csmf_accuracy'):
                            score = model.calculate_csmf_accuracy(y_fold_val, y_pred)
                        else:
                            # Fallback CSMF calculation
                            from model_comparison.metrics.comparison_metrics import calculate_csmf_accuracy
                            score = calculate_csmf_accuracy(y_fold_val, y_pred)
                    else:
                        score = (y_fold_val == y_pred).mean()
                    
                    out_domain_scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Error in cross-domain fold {fold}: {e}")
                    out_domain_scores.append(0.0)
            
            # Evaluate in-domain performance (using regular stratified CV)
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model = create_model(model_name, config)
                
                try:
                    # Same preprocessing logic as above
                    if model_name != "insilico":
                        if model_name == "categorical_nb":
                            model.fit(X_fold_train, y_fold_train)
                            y_pred = model.predict(X_fold_val)
                        else:
                            from model_comparison.orchestration.ray_tasks import _preprocess_features
                            X_fold_train_processed = _preprocess_features(X_fold_train)
                            X_fold_val_processed = _preprocess_features(X_fold_val)
                            model.fit(X_fold_train_processed, y_fold_train)
                            y_pred = model.predict(X_fold_val_processed)
                    else:
                        model.fit(X_fold_train, y_fold_train)
                        y_pred = model.predict(X_fold_val)
                    
                    if self.metric == "csmf_accuracy":
                        if hasattr(model, 'calculate_csmf_accuracy'):
                            score = model.calculate_csmf_accuracy(y_fold_val, y_pred)
                        else:
                            from model_comparison.metrics.comparison_metrics import calculate_csmf_accuracy
                            score = calculate_csmf_accuracy(y_fold_val, y_pred)
                    else:
                        score = (y_fold_val == y_pred).mean()
                    
                    in_domain_scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Error in in-domain fold {fold}: {e}")
                    in_domain_scores.append(0.0)
            
            # Calculate multi-objective score
            in_domain_avg = np.mean(in_domain_scores) if in_domain_scores else 0.0
            out_domain_avg = np.mean(out_domain_scores) if out_domain_scores else 0.0
            
            # Weight out-domain performance more heavily
            multi_objective = (
                self.multi_objective_alpha * in_domain_avg + 
                (1 - self.multi_objective_alpha) * out_domain_avg
            )
            
            # Return all metrics for analysis
            return {
                self.metric: multi_objective,  # Primary metric for optimization
                "in_domain_score": in_domain_avg,
                "out_domain_score": out_domain_avg,
                "generalization_gap": (in_domain_avg - out_domain_avg) / in_domain_avg if in_domain_avg > 0 else 0,
            }
        
        # Configure Ray Tune
        tune_config = tune.TuneConfig(
            num_samples=self.n_trials,
            metric=self.metric,
            mode=self.mode,
            search_alg=self._create_search_algorithm(),
        )
        
        run_config = ray.train.RunConfig(
            name=experiment_name,
            local_dir="/tmp/ray_results",
            verbose=1,
        )
        
        # Run tuning
        tuner = tune.Tuner(
            trainable=tune.with_resources(
                train_fn,
                resources={"cpu": self.n_cpus_per_trial}
            ),
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric=self.metric, mode=self.mode)
        
        return {
            "best_params": best_result.config,
            "best_score": best_result.metrics[self.metric],
            "in_domain_score": best_result.metrics.get("in_domain_score", 0),
            "out_domain_score": best_result.metrics.get("out_domain_score", 0),
            "generalization_gap": best_result.metrics.get("generalization_gap", 0),
            "n_trials_completed": len(results),
        }