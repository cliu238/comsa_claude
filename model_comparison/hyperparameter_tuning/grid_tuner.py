"""Grid search implementation for hyperparameter tuning."""

import time
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import ParameterGrid, cross_val_score

from baseline.utils import get_logger
from model_comparison.hyperparameter_tuning.search_spaces import get_grid_search_space
from model_comparison.hyperparameter_tuning.tuner import BaseTuner, TuningResult

logger = get_logger(__name__)


class GridSearchTuner(BaseTuner):
    """Grid search hyperparameter tuner.
    
    This tuner exhaustively searches through a manually specified subset
    of the hyperparameter space.
    """
    
    def __init__(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        grid_size: str = "small",
        metric: str = "csmf_accuracy",
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_seed: int = 42,
    ):
        """Initialize grid search tuner.
        
        Args:
            model_name: Name of the model to tune
            X: Training features
            y: Training labels
            grid_size: Size of the grid ('small', 'medium', 'large')
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
        self.grid_size = grid_size
        
    def _create_parameter_grid(self) -> List[Dict[str, Any]]:
        """Create the parameter grid for search.
        
        Returns:
            List of parameter dictionaries to evaluate
        """
        search_space = get_grid_search_space(self.model_name, self.grid_size)
        
        # Convert search space to sklearn-compatible parameter grid
        param_dict = {}
        for param_name, param_space in search_space.parameters.items():
            if isinstance(param_space.values, list):
                param_dict[param_name] = param_space.values
            else:
                # Should not happen with grid search space
                raise ValueError(f"Expected list of values for {param_name}")
        
        # Special handling for logistic regression
        if self.model_name == "logistic_regression":
            # Filter out invalid combinations
            grid = list(ParameterGrid(param_dict))
            valid_grid = []
            for params in grid:
                # Only include l1_ratio when penalty is elasticnet
                if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
                    params_copy = params.copy()
                    params_copy.pop("l1_ratio")
                    valid_grid.append(params_copy)
                else:
                    valid_grid.append(params)
            return valid_grid
        
        return list(ParameterGrid(param_dict))
    
    def tune(self) -> TuningResult:
        """Run grid search hyperparameter tuning.
        
        Returns:
            TuningResult with best parameters and performance metrics
        """
        start_time = time.time()
        
        logger.info(
            f"Starting grid search for {self.model_name} with "
            f"grid_size={self.grid_size}, optimizing {self.metric}"
        )
        
        # Create parameter grid
        param_grid = self._create_parameter_grid()
        n_combinations = len(param_grid)
        logger.info(f"Evaluating {n_combinations} parameter combinations")
        
        # Track results
        best_score = -float("inf")
        best_params = None
        all_results = []
        
        # Evaluate each combination
        for i, params in enumerate(param_grid):
            try:
                # Create model with current parameters
                model = self._create_model(params)
                
                # Perform cross-validation
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
                if (i + 1) % 10 == 0 or (i + 1) == n_combinations:
                    logger.info(f"Progress: {i + 1}/{n_combinations} combinations evaluated")
                    
            except Exception as e:
                logger.warning(f"Trial {i} failed with params {params}: {str(e)}")
                continue
        
        duration = time.time() - start_time
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("mean_score", ascending=False)
        
        logger.info(
            f"Grid search completed in {duration:.1f}s. "
            f"Best {self.metric}: {best_score:.4f}"
        )
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=len(all_results),
            duration_seconds=duration,
            all_trials=results_df,
        )