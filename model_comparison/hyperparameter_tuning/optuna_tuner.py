"""Optuna-based Bayesian optimization for hyperparameter tuning."""

import time
from typing import Any, Dict, Optional

import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from baseline.utils import get_logger
from model_comparison.hyperparameter_tuning.search_spaces import get_search_space
from model_comparison.hyperparameter_tuning.tuner import BaseTuner, TuningResult

logger = get_logger(__name__)


class OptunaTuner(BaseTuner):
    """Optuna-based hyperparameter tuner using Bayesian optimization.
    
    This tuner uses Tree-structured Parzen Estimator (TPE) for efficient
    exploration of the hyperparameter space.
    """
    
    def __init__(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        metric: str = "csmf_accuracy",
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_seed: int = 42,
    ):
        """Initialize Optuna tuner.
        
        Args:
            model_name: Name of the model to tune
            X: Training features
            y: Training labels
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
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
        self.timeout = timeout
        
    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        search_space = get_search_space(self.model_name)
        params = {}
        
        for param_name, param_space in search_space.parameters.items():
            if param_space.type == "int":
                if isinstance(param_space.values, dict):
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_space.values["low"],
                        param_space.values["high"],
                    )
            elif param_space.type == "float":
                if isinstance(param_space.values, dict):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_space.values["low"],
                        param_space.values["high"],
                        log=param_space.log_scale,
                    )
            elif param_space.type == "categorical":
                if isinstance(param_space.values, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_space.values,
                    )
        
        # Special handling for logistic regression
        if self.model_name == "logistic_regression":
            # Only include l1_ratio if penalty is elasticnet
            if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
                params.pop("l1_ratio")
        
        return params
    
    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Negative metric value (Optuna minimizes by default)
        """
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        
        try:
            # Create model with suggested parameters
            model = self._create_model(params)
            
            # Try stratified cross-validation first, fall back to regular if needed
            try:
                cv = StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_seed
                )
                scores = cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=cv,
                    scoring=self._get_scoring_function(),
                    n_jobs=self.n_jobs,
                )
            except ValueError as e:
                # Fall back to regular KFold if stratification fails
                logger.warning(f"Stratified CV failed: {e}. Using regular KFold.")
                cv = KFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_seed
                )
                scores = cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=cv,
                    scoring=self._get_scoring_function(),
                    n_jobs=self.n_jobs,
                )
            
            mean_score = scores.mean()
            
            # Log intermediate results
            logger.info(
                f"Trial {trial.number}: {self.metric}={mean_score:.4f}, "
                f"params={params}"
            )
            
            # Return negative score (Optuna minimizes)
            return -mean_score
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            # Return worst possible score
            return float("inf")
    
    def _progress_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback to report progress during optimization.
        
        Args:
            study: Optuna study object
            trial: Completed trial
        """
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_value = study.best_value
            n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            logger.info(
                f"Progress: {n_complete}/{self.n_trials} trials completed. "
                f"Best {self.metric}: {-best_value:.4f}"
            )
    
    def tune(self) -> TuningResult:
        """Run Optuna hyperparameter tuning.
        
        Returns:
            TuningResult with best parameters and performance metrics
        """
        start_time = time.time()
        
        logger.info(
            f"Starting Optuna optimization for {self.model_name} with "
            f"{self.n_trials} trials, optimizing {self.metric}"
        )
        
        # Create study
        study = optuna.create_study(
            direction="minimize",  # We negate scores
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        
        # Optimize
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
                callbacks=[self._progress_callback],
            )
        except Exception as e:
            logger.warning(f"Optimization stopped: {e}")
        
        # Get results
        best_score = -study.best_value
        best_params = study.best_params
        
        # Handle special cases for best params
        if self.model_name == "logistic_regression":
            # Ensure l1_ratio is set correctly for elasticnet
            if best_params.get("penalty") == "elasticnet" and "l1_ratio" not in best_params:
                best_params["l1_ratio"] = 0.5  # Default value
        
        duration = time.time() - start_time
        
        # Create trials dataframe
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    "trial_number": trial.number,
                    "value": -trial.value,  # Convert back to positive
                    "params": trial.params,
                    "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds(),
                }
                trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data) if trials_data else None
        
        logger.info(
            f"Optuna optimization completed in {duration:.1f}s. "
            f"Best {self.metric}: {best_score:.4f}"
        )
        logger.info(f"Best params: {best_params}")
        
        # Log parameter importance if available
        try:
            importances = optuna.importance.get_param_importances(study)
            logger.info("Parameter importances:")
            for param, importance in importances.items():
                logger.info(f"  {param}: {importance:.3f}")
        except Exception:
            # Parameter importance calculation may fail with few trials
            pass
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=len(study.trials),
            duration_seconds=duration,
            study=study,
            all_trials=trials_df,
        )