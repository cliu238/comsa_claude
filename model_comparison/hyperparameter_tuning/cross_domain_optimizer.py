"""Cross-domain hyperparameter optimizer for XGBoost.

This module implements true leave-one-site-out cross-validation for hyperparameter
optimization, optimizing for both in-domain and out-domain performance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig

logger = logging.getLogger(__name__)


class CrossDomainOptimizer:
    """Optimizer for XGBoost hyperparameters using cross-domain validation.
    
    This optimizer uses leave-one-site-out cross-validation to optimize
    hyperparameters for better out-of-domain generalization while maintaining
    good in-domain performance.
    """
    
    def __init__(
        self,
        sites: List[str],
        in_domain_weight: float = 0.5,
        optimization_metric: str = "csmf_accuracy",
        n_folds: int = 5,
        random_seed: int = 42,
    ):
        """Initialize cross-domain optimizer.
        
        Args:
            sites: List of site names for leave-one-out validation
            in_domain_weight: Weight for in-domain performance (0-1)
                             out_domain_weight = 1 - in_domain_weight
            optimization_metric: Metric to optimize ("csmf_accuracy" or "cod_accuracy")
            n_folds: Number of folds for in-domain cross-validation
            random_seed: Random seed for reproducibility
        """
        self.sites = sites
        self.in_domain_weight = in_domain_weight
        self.out_domain_weight = 1 - in_domain_weight
        self.optimization_metric = optimization_metric
        self.n_folds = n_folds
        self.random_seed = random_seed
        
    def objective(
        self,
        trial: optuna.Trial,
        data_by_site: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    ) -> float:
        """Objective function for cross-domain optimization.
        
        Args:
            trial: Optuna trial object
            data_by_site: Dictionary mapping site names to (X, y) tuples
            
        Returns:
            Combined loss (lower is better)
        """
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial)
        
        # Create model with sampled parameters
        config = self._create_config(params)
        
        # Calculate scores for each site
        in_domain_scores = []
        out_domain_scores = []
        
        for held_out_site in self.sites:
            # Split data into train sites and held-out site
            train_sites = [s for s in self.sites if s != held_out_site]
            
            # Combine training data from all train sites
            X_train_list = []
            y_train_list = []
            for site in train_sites:
                X_site, y_site = data_by_site[site]
                X_train_list.append(X_site)
                y_train_list.append(y_site)
            
            X_train_combined = pd.concat(X_train_list, ignore_index=True)
            y_train_combined = pd.concat(y_train_list, ignore_index=True)
            
            # Get held-out test data
            X_test, y_test = data_by_site[held_out_site]
            
            try:
                # Train model on combined training data
                model = XGBoostModel(config=config)
                model.fit(X_train_combined, y_train_combined)
                
                # Evaluate on held-out site (out-domain)
                y_pred_test = model.predict(X_test)
                out_domain_score = self._calculate_metric(y_test, y_pred_test)
                out_domain_scores.append(out_domain_score)
                
                # Also evaluate in-domain performance using CV on training sites
                if self.in_domain_weight > 0:
                    in_domain_score = self._evaluate_in_domain(
                        model, X_train_combined, y_train_combined
                    )
                    in_domain_scores.append(in_domain_score)
                    
            except Exception as e:
                logger.warning(f"Trial failed for site {held_out_site}: {e}")
                return float('inf')  # Return high loss for failed trials
        
        # Calculate combined loss
        avg_in_domain = np.mean(in_domain_scores) if in_domain_scores else 0
        avg_out_domain = np.mean(out_domain_scores)
        
        # Convert to loss (we minimize in Optuna)
        in_domain_loss = 1 - avg_in_domain
        out_domain_loss = 1 - avg_out_domain
        
        # Weighted combination
        combined_loss = (
            self.in_domain_weight * in_domain_loss +
            self.out_domain_weight * out_domain_loss
        )
        
        # Log intermediate values
        trial.set_user_attr("in_domain_score", avg_in_domain)
        trial.set_user_attr("out_domain_score", avg_out_domain)
        trial.set_user_attr("generalization_gap", avg_in_domain - avg_out_domain)
        
        return combined_loss
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for XGBoost.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        # Tree parameters
        max_depth = trial.suggest_int("max_depth", 2, 8)
        min_child_weight = trial.suggest_int("min_child_weight", 10, 100, log=True)
        gamma = trial.suggest_float("gamma", 0.0, 5.0)
        
        # Learning parameters
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
        
        # Sampling parameters - key for generalization
        subsample = trial.suggest_float("subsample", 0.4, 0.9, step=0.1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 0.8, step=0.1)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.3, 0.8, step=0.1)
        colsample_bynode = trial.suggest_float("colsample_bynode", 0.3, 0.8, step=0.1)
        
        # Regularization parameters
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 100.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 100.0, log=True)
        
        return {
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
    
    def _create_config(self, params: Dict[str, Any]) -> XGBoostEnhancedConfig:
        """Create XGBoost configuration from parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            XGBoostEnhancedConfig object
        """
        return XGBoostEnhancedConfig(**params)
    
    def _calculate_metric(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate the optimization metric.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Metric score (higher is better)
        """
        if self.optimization_metric == "csmf_accuracy":
            # Use the same CSMF accuracy calculation as XGBoostModel
            model = XGBoostModel()  # Temporary instance for metric calculation
            return model.calculate_csmf_accuracy(y_true, y_pred)
        elif self.optimization_metric == "cod_accuracy":
            return (y_true == y_pred).mean()
        else:
            raise ValueError(f"Unknown metric: {self.optimization_metric}")
    
    def _evaluate_in_domain(
        self,
        model: XGBoostModel,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Evaluate in-domain performance using cross-validation.
        
        Args:
            model: Configured XGBoost model
            X: Training features
            y: Training labels
            
        Returns:
            Average in-domain score
        """
        scores = []
        kfold = StratifiedKFold(
            n_splits=min(self.n_folds, len(np.unique(y))),
            shuffle=True,
            random_state=self.random_seed
        )
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone model for this fold
            fold_model = XGBoostModel(config=model.config)
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_val)
            score = self._calculate_metric(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def optimize(
        self,
        data_by_site: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        n_trials: int = 100,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], optuna.Study]:
        """Run cross-domain hyperparameter optimization.
        
        Args:
            data_by_site: Dictionary mapping site names to (X, y) tuples
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs for optimization
            study_name: Optional name for the Optuna study
            
        Returns:
            Tuple of (best_params, study)
        """
        # Create study
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name or "cross_domain_optimization",
        )
        
        # Add custom logging callback
        def log_trial(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(
                    f"Trial {trial.number}: "
                    f"Loss={trial.value:.4f}, "
                    f"In-domain={trial.user_attrs.get('in_domain_score', 0):.4f}, "
                    f"Out-domain={trial.user_attrs.get('out_domain_score', 0):.4f}, "
                    f"Gap={trial.user_attrs.get('generalization_gap', 0):.4f}"
                )
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, data_by_site),
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[log_trial],
        )
        
        # Get best parameters
        best_params = study.best_params
        best_trial = study.best_trial
        
        logger.info(
            f"\nBest trial: {best_trial.number}\n"
            f"Best loss: {best_trial.value:.4f}\n"
            f"Best in-domain score: {best_trial.user_attrs.get('in_domain_score', 0):.4f}\n"
            f"Best out-domain score: {best_trial.user_attrs.get('out_domain_score', 0):.4f}\n"
            f"Best generalization gap: {best_trial.user_attrs.get('generalization_gap', 0):.4f}\n"
            f"Best params: {best_params}"
        )
        
        return best_params, study