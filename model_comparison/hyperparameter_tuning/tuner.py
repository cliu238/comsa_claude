"""Base tuner interface and factory for hyperparameter tuning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class TuningResult:
    """Result from hyperparameter tuning."""
    
    best_params: Dict[str, Any]
    best_score: float
    n_trials_completed: int
    duration_seconds: float
    study: Optional[Any] = None  # Optuna study or similar object
    all_trials: Optional[pd.DataFrame] = None  # History of all trials


class BaseTuner(ABC):
    """Abstract base class for hyperparameter tuners."""
    
    def __init__(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "csmf_accuracy",
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_seed: int = 42,
    ):
        """Initialize base tuner.
        
        Args:
            model_name: Name of the model to tune
            X: Training features
            y: Training labels
            metric: Metric to optimize ('csmf_accuracy' or 'cod_accuracy')
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (set to 1 when using Ray)
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.X = X
        self.y = y
        self.metric = metric
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        
        # Validate metric
        valid_metrics = ["csmf_accuracy", "cod_accuracy", "accuracy"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
    
    @abstractmethod
    def tune(self) -> TuningResult:
        """Run hyperparameter tuning.
        
        Returns:
            TuningResult with best parameters and performance metrics
        """
        pass
    
    def _get_scoring_function(self):
        """Get the scoring function for the specified metric.
        
        Returns:
            Callable scoring function for cross-validation
        """
        from sklearn.metrics import make_scorer
        
        if self.metric == "csmf_accuracy":
            # Import inside try-except for graceful fallback
            try:
                from baseline.metrics.va_metrics import csmf_accuracy
                return make_scorer(csmf_accuracy, greater_is_better=True)
            except ImportError:
                # Fallback to accuracy if VA metrics not available
                from sklearn.metrics import accuracy_score
                return make_scorer(accuracy_score, greater_is_better=True)
        elif self.metric == "cod_accuracy":
            try:
                from baseline.metrics.va_metrics import cod_accuracy
                return make_scorer(cod_accuracy, greater_is_better=True)
            except ImportError:
                from sklearn.metrics import accuracy_score
                return make_scorer(accuracy_score, greater_is_better=True)
        else:  # accuracy
            from sklearn.metrics import accuracy_score
            return make_scorer(accuracy_score, greater_is_better=True)
    
    def _create_model(self, params: Dict[str, Any]):
        """Create a model instance with given parameters.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        # This will be implemented using the model factory
        from baseline.models.model_factory import create_model
        return create_model(self.model_name, params)


def get_tuner(
    method: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    metric: str = "csmf_accuracy",
    cv_folds: int = 5,
    n_jobs: int = 1,
    random_seed: int = 42,
    **kwargs
) -> BaseTuner:
    """Factory function to create a tuner instance.
    
    Args:
        method: Tuning method ('grid', 'random', 'optuna', 'ray_tune')
        model_name: Name of the model to tune
        X: Training features
        y: Training labels
        n_trials: Number of trials/iterations
        timeout: Maximum time in seconds for tuning
        metric: Metric to optimize
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        random_seed: Random seed
        **kwargs: Additional method-specific arguments
        
    Returns:
        Configured tuner instance
        
    Raises:
        ValueError: If method is not recognized
    """
    if method == "grid":
        from model_comparison.hyperparameter_tuning.grid_tuner import GridSearchTuner
        return GridSearchTuner(
            model_name=model_name,
            X=X,
            y=y,
            metric=metric,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_seed=random_seed,
            grid_size=kwargs.get("grid_size", "small"),
        )
    elif method == "random":
        from model_comparison.hyperparameter_tuning.random_tuner import RandomSearchTuner
        return RandomSearchTuner(
            model_name=model_name,
            X=X,
            y=y,
            n_trials=n_trials,
            metric=metric,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
    elif method == "optuna":
        from model_comparison.hyperparameter_tuning.optuna_tuner import OptunaTuner
        return OptunaTuner(
            model_name=model_name,
            X=X,
            y=y,
            n_trials=n_trials,
            timeout=timeout,
            metric=metric,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
    elif method == "ray_tune":
        from model_comparison.hyperparameter_tuning.ray_tuner import RayTuneTuner
        return RayTuneTuner(
            model_name=model_name,
            X=X,
            y=y,
            n_trials=n_trials,
            timeout=timeout,
            metric=metric,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_seed=random_seed,
        )
    else:
        raise ValueError(
            f"Unknown tuning method: {method}. "
            f"Supported methods: grid, random, optuna, ray_tune"
        )