"""Random search implementation for hyperparameter tuning."""

import time
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import ParameterSampler

from baseline.utils import get_logger
from model_comparison.hyperparameter_tuning.search_spaces import get_search_space
from model_comparison.hyperparameter_tuning.tuner import BaseTuner, TuningResult

logger = get_logger(__name__)


class RandomSearchTuner(BaseTuner):
    """Random search hyperparameter tuner.
    
    This tuner randomly samples from the hyperparameter space,
    which is often more efficient than grid search for high-dimensional spaces.
    """
    
    def __init__(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        metric: str = "csmf_accuracy",
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_seed: int = 42,
    ):
        """Initialize random search tuner.
        
        Args:
            model_name: Name of the model to tune
            X: Training features
            y: Training labels
            n_trials: Number of random samples to evaluate
            metric: Metric to optimize
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            model_name=model_name,
            X=X,
            y=y,
            metric=metric,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
        self.n_trials = n_trials
    
    def _create_param_distributions(self) -> Dict[str, Any]:
        """Create parameter distributions for random sampling.
        
        Returns:
            Dictionary of parameter distributions
        """
        search_space = get_search_space(self.model_name)
        param_distributions = {}
        
        for param_name, param_space in search_space.parameters.items():
            if param_space.type == "categorical":
                # Use list as-is for categorical
                param_distributions[param_name] = param_space.values
            elif param_space.type in ["int", "float"]:
                # Create scipy distributions for numeric parameters
                if isinstance(param_space.values, dict):
                    low = param_space.values["low"]
                    high = param_space.values["high"]
                    
                    if param_space.type == "int":
                        from scipy.stats import randint
                        param_distributions[param_name] = randint(low, high + 1)
                    else:  # float
                        if param_space.log_scale:
                            from scipy.stats import loguniform
                            param_distributions[param_name] = loguniform(low, high)
                        else:
                            from scipy.stats import uniform
                            param_distributions[param_name] = uniform(low, high - low)
        
        return param_distributions
    
    def tune(self) -> TuningResult:
        """Run random search hyperparameter tuning.
        
        Returns:
            TuningResult with best parameters and performance metrics
        """
        start_time = time.time()
        
        logger.info(
            f"Starting random search for {self.model_name} with "
            f"{self.n_trials} trials, optimizing {self.metric}"
        )
        
        # Create parameter distributions
        param_distributions = self._create_param_distributions()
        
        # Track results
        best_score = -float("inf")
        best_params = None
        all_results = []
        
        # Sample parameters
        sampler = ParameterSampler(
            param_distributions,
            n_iter=self.n_trials,
            random_state=self.random_seed,
        )
        
        # Evaluate each sample
        for i, params in enumerate(sampler):
            try:
                # Handle special cases
                if self.model_name == "logistic_regression":
                    # Remove l1_ratio if not using elasticnet
                    if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
                        params.pop("l1_ratio")
                
                # Create model with sampled parameters
                model = self._create_model(params)
                
                # Perform cross-validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=self.cv_folds,
                    scoring=self._get_scoring_function(),
                    n_jobs=self.n_jobs,
                )
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                # Track results
                result = {
                    "params": params,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "trial_number": i,
                }
                all_results.append(result)
                
                # Update best if needed
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    logger.info(
                        f"New best score: {best_score:.4f} with params: {best_params}"
                    )
                
                # Progress update
                if (i + 1) % 10 == 0 or (i + 1) == self.n_trials:
                    logger.info(f"Progress: {i + 1}/{self.n_trials} trials completed")
                    
            except Exception as e:
                logger.warning(f"Trial {i} failed with params {params}: {str(e)}")
                continue
        
        duration = time.time() - start_time
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("mean_score", ascending=False)
        
        logger.info(
            f"Random search completed in {duration:.1f}s. "
            f"Best {self.metric}: {best_score:.4f}"
        )
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=len(all_results),
            duration_seconds=duration,
            all_trials=results_df,
        )