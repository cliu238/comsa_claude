"""Hyperparameter tuning for XGBoost model using Optuna."""

import logging
from typing import Any, Callable, Dict, List, Optional

import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler

from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.categorical_nb_config import CategoricalNBConfig
from baseline.models.categorical_nb_model import CategoricalNBModel

logger = logging.getLogger(__name__)


class XGBoostHyperparameterTuner:
    """Hyperparameter tuning for XGBoost using Optuna.

    This class provides functionality to automatically tune XGBoost hyperparameters
    using Bayesian optimization via Optuna. It optimizes for CSMF accuracy by default.

    Attributes:
        base_config: Base configuration to use as starting point
        metric: Metric to optimize ('csmf_accuracy' or 'cod_accuracy')
    """

    def __init__(
        self,
        base_config: Optional[XGBoostConfig] = None,
        metric: str = "csmf_accuracy",
    ):
        """Initialize hyperparameter tuner.

        Args:
            base_config: Base XGBoostConfig to use as template
            metric: Metric to optimize. Options: 'csmf_accuracy', 'cod_accuracy'

        Raises:
            ValueError: If metric is not recognized
        """
        self.base_config = base_config or XGBoostConfig()

        valid_metrics = ["csmf_accuracy", "cod_accuracy"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
        self.metric = metric

    def objective(
        self,
        trial: Trial,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            X: Training features
            y: Training labels
            cv: Number of cross-validation folds

        Returns:
            Negative metric value (Optuna minimizes by default)
        """
        # Suggest hyperparameters
        config_dict = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        # Update config with base config values
        full_config_dict = self.base_config.model_dump()
        full_config_dict.update(config_dict)

        # Create config
        config = XGBoostConfig(**full_config_dict)

        # Train and evaluate
        model = XGBoostModel(config=config)

        try:
            cv_results = model.cross_validate(X, y, cv=cv)

            # Get the mean score for the specified metric
            metric_key = f"{self.metric}_mean"
            score = cv_results[metric_key]

            # Log intermediate results
            logger.info(
                f"Trial {trial.number}: {self.metric}={score:.4f}, "
                f"params={config_dict}"
            )

            # Return negative score (Optuna minimizes)
            return -score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            # Return worst possible score
            return float("inf")

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        cv: int = 5,
        study_name: Optional[str] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter tuning.

        Args:
            X: Training features
            y: Training labels
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            study_name: Name for the Optuna study
            timeout: Time limit in seconds for optimization
            n_jobs: Number of parallel jobs for optimization
            callbacks: List of callback functions for Optuna

        Returns:
            Dictionary containing:
                - best_params: Best hyperparameters found
                - best_score: Best metric score achieved
                - study: Optuna study object
                - best_config: Complete XGBoostConfig with best parameters
        """
        logger.info(
            f"Starting hyperparameter tuning with {n_trials} trials, "
            f"optimizing {self.metric}"
        )

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            study_name=study_name,
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y, cv),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=True,
        )

        # Get best results
        best_score = -study.best_value
        best_params = study.best_params

        logger.info(f"Best {self.metric}: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        # Create best config
        full_config_dict = self.base_config.model_dump()
        full_config_dict.update(best_params)
        best_config = XGBoostConfig(**full_config_dict)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
            "best_config": best_config,
            "n_trials": len(study.trials),
        }

    def plot_optimization_history(
        self, study: optuna.Study
    ) -> "optuna.visualization.matplotlib.plot_optimization_history":
        """Plot optimization history.

        Args:
            study: Optuna study object

        Returns:
            Matplotlib figure
        """
        try:
            import optuna.visualization.matplotlib as ovm

            return ovm.plot_optimization_history(study)
        except ImportError:
            logger.warning(
                "Matplotlib not available. Install optuna[matplotlib] for plotting."
            )
            return None

    def plot_param_importances(
        self, study: optuna.Study
    ) -> "optuna.visualization.matplotlib.plot_param_importances":
        """Plot parameter importances.

        Args:
            study: Optuna study object

        Returns:
            Matplotlib figure
        """
        try:
            import optuna.visualization.matplotlib as ovm

            return ovm.plot_param_importances(study)
        except ImportError:
            logger.warning(
                "Matplotlib not available. Install optuna[matplotlib] for plotting."
            )
            return None

    def get_param_importance(self, study: optuna.Study) -> Dict[str, float]:
        """Get parameter importance scores.

        Args:
            study: Optuna study object

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            from optuna.importance import get_param_importances

            return get_param_importances(study)
        except ImportError:
            logger.warning("Parameter importance calculation requires optuna>=2.5.0")
            return {}


def quick_tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    metric: str = "csmf_accuracy",
) -> XGBoostModel:
    """Quick function to tune and return best XGBoost model.

    Args:
        X: Training features
        y: Training labels
        n_trials: Number of optimization trials
        metric: Metric to optimize

    Returns:
        Fitted XGBoostModel with best hyperparameters
    """
    tuner = XGBoostHyperparameterTuner(metric=metric)
    results = tuner.tune(X, y, n_trials=n_trials)

    # Train final model with best config
    best_model = XGBoostModel(config=results["best_config"])
    best_model.fit(X, y)

    logger.info("Final model trained with best hyperparameters")

    return best_model


class CategoricalNBHyperparameterTuner:
    """Hyperparameter tuning for CategoricalNB using Optuna.

    This class provides functionality to automatically tune CategoricalNB hyperparameters
    using Bayesian optimization via Optuna. It optimizes for CSMF accuracy by default.

    Attributes:
        base_config: Base configuration to use as starting point
        metric: Metric to optimize ('csmf_accuracy' or 'cod_accuracy')
    """

    def __init__(
        self,
        base_config: Optional[CategoricalNBConfig] = None,
        metric: str = "csmf_accuracy",
    ):
        """Initialize hyperparameter tuner.

        Args:
            base_config: Base CategoricalNBConfig to use as template
            metric: Metric to optimize. Options: 'csmf_accuracy', 'cod_accuracy'

        Raises:
            ValueError: If metric is not recognized
        """
        self.base_config = base_config or CategoricalNBConfig()

        valid_metrics = ["csmf_accuracy", "cod_accuracy"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
        self.metric = metric

    def objective(
        self,
        trial: Trial,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            X: Training features
            y: Training labels
            cv: Number of cross-validation folds

        Returns:
            Negative metric value (Optuna minimizes by default)
        """
        # Suggest hyperparameters for CategoricalNB
        config_dict = {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0, log=True),
            "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
            "force_alpha": trial.suggest_categorical("force_alpha", [True, False]),
        }

        # Update config with base config values
        full_config_dict = self.base_config.model_dump()
        full_config_dict.update(config_dict)

        # Create config
        config = CategoricalNBConfig(**full_config_dict)

        # Train and evaluate
        model = CategoricalNBModel(config=config)

        try:
            cv_results = model.cross_validate(X, y, cv=cv)

            # Get the mean score for the specified metric
            metric_key = f"{self.metric}_mean"
            score = cv_results[metric_key]

            # Log intermediate results
            logger.info(
                f"Trial {trial.number}: {self.metric}={score:.4f}, "
                f"params={config_dict}"
            )

            # Return negative score (Optuna minimizes)
            return -score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            # Return worst possible score
            return float("inf")

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,  # Fewer trials for simpler model
        cv: int = 5,
        study_name: Optional[str] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter tuning.

        Args:
            X: Training features
            y: Training labels
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
            study_name: Name for the Optuna study
            timeout: Time limit in seconds for optimization
            n_jobs: Number of parallel jobs for optimization
            callbacks: List of callback functions for Optuna

        Returns:
            Dictionary containing:
                - best_params: Best hyperparameters found
                - best_score: Best metric score achieved
                - study: Optuna study object
                - best_config: Complete CategoricalNBConfig with best parameters
        """
        logger.info(
            f"Starting CategoricalNB hyperparameter tuning with {n_trials} trials, "
            f"optimizing {self.metric}"
        )

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            study_name=study_name,
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y, cv),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=True,
        )

        # Get best results
        best_score = -study.best_value
        best_params = study.best_params

        logger.info(f"Best {self.metric}: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        # Create best config
        full_config_dict = self.base_config.model_dump()
        full_config_dict.update(best_params)
        best_config = CategoricalNBConfig(**full_config_dict)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
            "best_config": best_config,
            "n_trials": len(study.trials),
        }

    def plot_optimization_history(
        self, study: optuna.Study
    ) -> "optuna.visualization.matplotlib.plot_optimization_history":
        """Plot optimization history.

        Args:
            study: Optuna study object

        Returns:
            Matplotlib figure
        """
        try:
            import optuna.visualization.matplotlib as ovm

            return ovm.plot_optimization_history(study)
        except ImportError:
            logger.warning(
                "Matplotlib not available. Install optuna[matplotlib] for plotting."
            )
            return None

    def plot_param_importances(
        self, study: optuna.Study
    ) -> "optuna.visualization.matplotlib.plot_param_importances":
        """Plot parameter importances.

        Args:
            study: Optuna study object

        Returns:
            Matplotlib figure
        """
        try:
            import optuna.visualization.matplotlib as ovm

            return ovm.plot_param_importances(study)
        except ImportError:
            logger.warning(
                "Matplotlib not available. Install optuna[matplotlib] for plotting."
            )
            return None

    def get_param_importance(self, study: optuna.Study) -> Dict[str, float]:
        """Get parameter importance scores.

        Args:
            study: Optuna study object

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            from optuna.importance import get_param_importances

            return get_param_importances(study)
        except ImportError:
            logger.warning("Parameter importance calculation requires optuna>=2.5.0")
            return {}


def quick_tune_categorical_nb(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 25,  # Fewer trials for simpler model
    metric: str = "csmf_accuracy",
) -> CategoricalNBModel:
    """Quick function to tune and return best CategoricalNB model.

    Args:
        X: Training features
        y: Training labels
        n_trials: Number of optimization trials
        metric: Metric to optimize

    Returns:
        Fitted CategoricalNBModel with best hyperparameters
    """
    tuner = CategoricalNBHyperparameterTuner(metric=metric)
    results = tuner.tune(X, y, n_trials=n_trials)

    # Train final model with best config
    best_model = CategoricalNBModel(config=results["best_config"])
    best_model.fit(X, y)

    logger.info("Final CategoricalNB model trained with best hyperparameters")

    return best_model
