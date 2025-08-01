"""Ray remote functions for parallel VA model training and evaluation.

This module ensures consistent data preprocessing across all models:
- All models use "cause" as the label column (created from "va34" if needed)
- The same set of label-equivalent columns are dropped for all models
- Feature columns are identical across all model types
"""

import time
import traceback
import uuid
from typing import Dict, Optional, Tuple

import pandas as pd
import ray

from baseline.utils import get_logger
from model_comparison.orchestration.config import ExperimentResult
from model_comparison.experiments.experiment_config import TuningConfig


@ray.remote
def train_and_evaluate_model(
    model_name: str,
    train_data: Tuple[pd.DataFrame, pd.Series],
    test_data: Tuple[pd.DataFrame, pd.Series],
    experiment_metadata: Dict,
    n_bootstrap: int = 100,
) -> ExperimentResult:
    """Train and evaluate a model in a Ray worker.

    This function is executed remotely on Ray workers. All imports are done
    inside the function to ensure proper serialization.

    Args:
        model_name: Name of the model to train ('insilico', 'xgboost', 'random_forest', or 'logistic_regression')
        train_data: Tuple of (X_train, y_train)
        test_data: Tuple of (X_test, y_test)
        experiment_metadata: Dictionary with experiment details
        n_bootstrap: Number of bootstrap iterations for metrics

    Returns:
        ExperimentResult with metrics and metadata
    """
    start_time = time.time()
    retry_count = experiment_metadata.get("retry_count", 0)

    try:
        # Import inside remote function for serialization
        from baseline.models.insilico_model import InSilicoVAModel
        from baseline.models.xgboost_model import XGBoostModel
        from baseline.models.random_forest_model import RandomForestModel
        from baseline.models.categorical_nb_model import CategoricalNBModel
        from model_comparison.metrics.comparison_metrics import calculate_metrics

        # Set up logging for worker
        logger = get_logger(__name__, component="orchestration", console=False)
        logger.info(
            f"Worker starting: {model_name} - {experiment_metadata.get('experiment_id')}"
        )

        # Unpack data
        X_train, y_train = train_data
        X_test, y_test = test_data

        # Apply training size subsampling if specified
        training_size = experiment_metadata.get("training_size", 1.0)
        if training_size < 1.0:
            from sklearn.model_selection import train_test_split
            
            # Subsample the training data while preserving class distribution
            try:
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, 
                    train_size=training_size, 
                    random_state=42,  # Fixed seed for reproducibility
                    stratify=y_train
                )
                logger.info(
                    f"Subsampled training data from {len(train_data[0])} to {len(X_train)} samples "
                    f"(training_size={training_size})"
                )
            except ValueError as e:
                # If stratification fails (e.g., some classes have only 1 sample), 
                # fall back to random sampling
                logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, 
                    train_size=training_size, 
                    random_state=42
                )

        # Initialize model
        if model_name == "insilico":
            model = InSilicoVAModel()
        elif model_name == "xgboost":
            model = XGBoostModel()
        elif model_name == "random_forest":
            model = RandomForestModel()
        elif model_name == "logistic_regression":
            from baseline.models.logistic_regression_model import LogisticRegressionModel
            model = LogisticRegressionModel()
        elif model_name == "categorical_nb":
            model = CategoricalNBModel()
        elif model_name == "ensemble":
            # Handle ensemble model
            from baseline.models.ensemble_model import DuckVotingEnsemble
            from baseline.models.ensemble_config import EnsembleConfig
            
            # Get ensemble configuration from metadata
            ensemble_metadata = experiment_metadata.get("ensemble_config", {})
            
            ensemble_config = EnsembleConfig(
                voting=ensemble_metadata.get("voting", "soft"),
                weight_optimization=ensemble_metadata.get("weight_optimization", "none"),
                estimators=ensemble_metadata.get("estimators", []),
                min_diversity=ensemble_metadata.get("min_diversity", 0.0),
                random_seed=42,
            )
            
            model = DuckVotingEnsemble(config=ensemble_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Preprocess features only for ML models (not InSilicoVA)
        if model_name == "insilico":
            # InSilicoVA needs the original "Y"/"." format
            X_train_processed = X_train
            X_test_processed = X_test
        elif model_name == "categorical_nb":
            # CategoricalNB has its own preprocessing in _prepare_categorical_features
            X_train_processed = X_train
            X_test_processed = X_test
        else:
            # Other models need numeric encoding
            X_train_processed = _preprocess_features(X_train)
            X_test_processed = _preprocess_features(X_test)

        # Train model
        logger.info(f"Training {model_name} on {len(X_train)} samples (training_size={training_size})")
        training_start = time.time()
        
        if model_name == "ensemble":
            # Get OpenVA data for ensemble if available
            train_data_openva_ref = experiment_metadata.get("train_data_openva")
            if train_data_openva_ref:
                # Get data from Ray object store
                train_data_openva = ray.get(train_data_openva_ref)
                X_train_openva, _ = train_data_openva
                model.fit(X_train_processed, y_train, X_openva=X_train_openva)
            else:
                model.fit(X_train_processed, y_train)
        else:
            model.fit(X_train_processed, y_train)
            
        training_time = time.time() - training_start

        # Make predictions
        inference_start = time.time()
        
        if model_name == "ensemble":
            # Get OpenVA test data for ensemble if available
            test_data_openva_ref = experiment_metadata.get("test_data_openva")
            if test_data_openva_ref:
                # Get data from Ray object store
                test_data_openva = ray.get(test_data_openva_ref)
                X_test_openva, _ = test_data_openva
                y_pred = model.predict(X_test_processed, X_openva=X_test_openva)
                try:
                    y_proba = model.predict_proba(X_test_processed, X_openva=X_test_openva)
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
                    y_proba = None
            else:
                y_pred = model.predict(X_test_processed)
                try:
                    y_proba = model.predict_proba(X_test_processed)
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")
                    y_proba = None
        else:
            y_pred = model.predict(X_test_processed)
            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test_processed)
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test, y_pred=y_pred, y_proba=y_proba, n_bootstrap=n_bootstrap
        )
        inference_time = time.time() - inference_start

        # Create result
        result = ExperimentResult(
            experiment_id=experiment_metadata["experiment_id"],
            model_name=model_name,
            experiment_type=experiment_metadata["experiment_type"],
            train_site=experiment_metadata["train_site"],
            test_site=experiment_metadata["test_site"],
            training_size=experiment_metadata.get("training_size", 1.0),
            csmf_accuracy=metrics["csmf_accuracy"],
            cod_accuracy=metrics["cod_accuracy"],
            csmf_accuracy_ci=metrics.get("csmf_accuracy_ci") if isinstance(metrics.get("csmf_accuracy_ci"), list) else None,
            cod_accuracy_ci=metrics.get("cod_accuracy_ci") if isinstance(metrics.get("cod_accuracy_ci"), list) else None,
            n_train=len(y_train),
            n_test=len(y_test),
            execution_time_seconds=time.time() - start_time,
            training_time_seconds=training_time,
            inference_time_seconds=inference_time,
            worker_id=ray.get_runtime_context().get_worker_id(),
            retry_count=retry_count,
        )

        logger.info(
            f"Completed: {model_name} - CSMF: {metrics['csmf_accuracy']:.3f}, "
            f"COD: {metrics['cod_accuracy']:.3f}"
        )

        return result

    except Exception as e:
        # Log error
        error_msg = f"Error in {model_name}: {str(e)}\n{traceback.format_exc()}"
        logger = get_logger(__name__, component="orchestration", console=False)
        logger.error(error_msg)

        # Return result with error
        return ExperimentResult(
            experiment_id=experiment_metadata["experiment_id"],
            model_name=model_name,
            experiment_type=experiment_metadata["experiment_type"],
            train_site=experiment_metadata["train_site"],
            test_site=experiment_metadata["test_site"],
            training_size=experiment_metadata.get("training_size", 1.0),
            csmf_accuracy=0.0,
            cod_accuracy=0.0,
            n_train=0,
            n_test=0,
            execution_time_seconds=time.time() - start_time,
            worker_id=ray.get_runtime_context().get_worker_id(),
            retry_count=retry_count,
            error=str(e),
        )


def _preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features for ML model training.
    
    Converts categorical features (including "Y"/"." format) to numeric codes.
    This is required for scikit-learn based models like XGBoost.
    
    Note: InSilicoVA should NOT use this preprocessing as it requires
    the original "Y"/"." format.

    Args:
        X: Input features DataFrame

    Returns:
        Preprocessed features DataFrame with numeric encoding
    """
    X_processed = X.copy()

    # Encode categorical features
    for col in X_processed.columns:
        if X_processed[col].dtype == "object":
            # Handle NaN values before encoding
            # Replace NaN with a special category before encoding
            X_processed[col] = X_processed[col].fillna("missing")
            # Simple label encoding for categorical features
            # This converts "Y" -> 1, "." -> 0, "missing" -> -1, etc.
            X_processed[col] = pd.Categorical(X_processed[col]).codes

    # Fill any remaining NaN values (numeric columns) with -1
    X_processed = X_processed.fillna(-1)

    return X_processed


@ray.remote
def tune_and_train_model(
    model_name: str,
    train_data: Tuple[pd.DataFrame, pd.Series],
    test_data: Tuple[pd.DataFrame, pd.Series],
    experiment_metadata: Dict,
    tuning_config: Optional[Dict] = None,
    n_bootstrap: int = 100,
) -> ExperimentResult:
    """Tune hyperparameters and train model.
    
    This function performs hyperparameter tuning if enabled, then trains
    the model with the best parameters found.
    
    Args:
        model_name: Name of the model to train
        train_data: Tuple of (X_train, y_train)
        test_data: Tuple of (X_test, y_test)
        experiment_metadata: Dictionary with experiment details
        tuning_config: Tuning configuration dictionary
        n_bootstrap: Number of bootstrap iterations for metrics
        
    Returns:
        ExperimentResult with metrics and metadata
    """
    start_time = time.time()
    retry_count = experiment_metadata.get("retry_count", 0)
    
    try:
        # Import inside remote function for serialization
        from baseline.models.insilico_model import InSilicoVAModel
        from baseline.models.xgboost_model import XGBoostModel
        from baseline.models.random_forest_model import RandomForestModel
        from baseline.models.logistic_regression_model import LogisticRegressionModel
        from baseline.models.categorical_nb_model import CategoricalNBModel
        from model_comparison.hyperparameter_tuning.ray_tuner import RayTuner
        from model_comparison.hyperparameter_tuning.enhanced_search_spaces import (
            get_xgboost_enhanced_search_space,
            get_random_forest_enhanced_search_space,
            get_search_space_for_model_enhanced,
        )
        from model_comparison.hyperparameter_tuning.search_spaces import (
            get_logistic_regression_search_space,
            get_categorical_nb_search_space,
        )
        from model_comparison.metrics.comparison_metrics import calculate_metrics
        from model_comparison.experiments.cross_domain_tuning import (
            CrossDomainCV,
            evaluate_cross_domain_performance,
        )
        
        # Set up logging for worker
        logger = get_logger(__name__, component="orchestration", console=False)
        logger.info(
            f"Worker starting: {model_name} - {experiment_metadata.get('experiment_id')} "
            f"(tuning enabled: {tuning_config and tuning_config.get('enabled', False)})"
        )
        
        # Unpack data
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Apply training size subsampling if specified
        training_size = experiment_metadata.get("training_size", 1.0)
        if training_size < 1.0:
            from sklearn.model_selection import train_test_split
            
            try:
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, 
                    train_size=training_size, 
                    random_state=42,
                    stratify=y_train
                )
                logger.info(
                    f"Subsampled training data from {len(train_data[0])} to {len(X_train)} samples "
                    f"(training_size={training_size})"
                )
            except ValueError as e:
                logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, 
                    train_size=training_size, 
                    random_state=42
                )
        
        # Initialize model with default or tuned parameters
        best_params = None
        tuning_results = None
        
        # Preprocess features based on model type
        if model_name == "insilico":
            # InSilicoVA needs the original "Y"/"." format
            X_train_processed = X_train
            X_test_processed = X_test
        elif model_name == "categorical_nb":
            # CategoricalNB has its own preprocessing in _prepare_categorical_features
            X_train_processed = X_train
            X_test_processed = X_test
        else:
            # Other models need numeric encoding
            X_train_processed = _preprocess_features(X_train)
            X_test_processed = _preprocess_features(X_test)
        
        # Handle hyperparameter tuning for supported models
        if model_name == "insilico":
            # InSilicoVA doesn't support hyperparameter tuning
            model = InSilicoVAModel()
        else:
            # Check if tuning is enabled
            if tuning_config and tuning_config.get("enabled", False):
                logger.info(f"Starting hyperparameter tuning for {model_name}")
                tuning_start = time.time()
                
                # Get model class and search space
                # Use more conservative search space for cross-domain experiments
                is_cross_domain = experiment_metadata.get("train_site") != experiment_metadata.get("test_site")
                
                if model_name == "xgboost" and is_cross_domain and tuning_config.get("use_conservative_space", True):
                    # Use conservative space for cross-domain XGBoost
                    from model_comparison.hyperparameter_tuning.enhanced_search_spaces import get_xgboost_conservative_search_space
                    xgboost_space = get_xgboost_conservative_search_space()
                    logger.info("Using conservative search space for cross-domain XGBoost tuning")
                else:
                    xgboost_space = get_xgboost_enhanced_search_space()
                
                model_classes = {
                    "xgboost": (XGBoostModel, xgboost_space),
                    "random_forest": (RandomForestModel, get_random_forest_enhanced_search_space()),
                    "logistic_regression": (LogisticRegressionModel, get_logistic_regression_search_space()),
                    "categorical_nb": (CategoricalNBModel, get_categorical_nb_search_space())
                }
                
                if model_name not in model_classes:
                    raise ValueError(f"Unknown model for tuning: {model_name}")
                
                model_class, search_space = model_classes[model_name]
                
                # Create tuner with resource constraints
                # Limit concurrent trials to prevent nested Ray worker explosion
                max_concurrent_trials = tuning_config.get("max_concurrent_tuning_trials", 2)
                if max_concurrent_trials is None:
                    # Default to 2 if not specified to prevent resource explosion
                    max_concurrent_trials = 2
                    logger.warning(
                        "No max_concurrent_tuning_trials specified. Defaulting to 2 to prevent resource explosion. "
                        "Consider setting --tuning-max-concurrent-trials explicitly."
                    )
                
                tuner = RayTuner(
                    n_trials=tuning_config.get("n_trials", 100),
                    search_algorithm=tuning_config.get("search_algorithm", "bayesian"),
                    metric=tuning_config.get("tuning_metric", "csmf_accuracy"),
                    n_cpus_per_trial=tuning_config.get("n_cpus_per_trial", 1.0),
                    max_concurrent_trials=max_concurrent_trials,
                )
                
                # Run tuning on a subset of data if it's large
                tuning_data_size = min(len(X_train), 5000)
                if tuning_data_size < len(X_train):
                    from sklearn.model_selection import train_test_split
                    
                    # Filter out rare classes for tuning to avoid CV issues
                    min_samples_per_class = max(tuning_config.get("cv_folds", 5) * 2, 10)
                    class_counts = y_train.value_counts()
                    valid_classes = class_counts[class_counts >= min_samples_per_class].index
                    
                    if len(valid_classes) < len(class_counts):
                        # Filter to only include samples from valid classes
                        valid_mask = y_train.isin(valid_classes)
                        X_train_filtered = X_train[valid_mask]
                        y_train_filtered = y_train[valid_mask]
                        logger.info(
                            f"Filtered {len(class_counts) - len(valid_classes)} rare classes "
                            f"(< {min_samples_per_class} samples) for tuning. "
                            f"Keeping {len(valid_classes)} classes with {len(X_train_filtered)} samples."
                        )
                    else:
                        X_train_filtered = X_train
                        y_train_filtered = y_train
                    
                    # Use original data for tuning split, not preprocessed
                    try:
                        X_tune, _, y_tune, _ = train_test_split(
                            X_train_filtered, y_train_filtered,
                            train_size=min(tuning_data_size, len(X_train_filtered)),
                            random_state=42,
                            stratify=y_train_filtered
                        )
                        logger.info(f"Using {len(X_tune)} samples for tuning with {len(y_tune.unique())} classes")
                    except ValueError as e:
                        # If stratification still fails, use random sampling
                        logger.warning(f"Stratified sampling failed for tuning: {e}. Using random sampling.")
                        X_tune, _, y_tune, _ = train_test_split(
                            X_train_filtered, y_train_filtered,
                            train_size=min(tuning_data_size, len(X_train_filtered)),
                            random_state=42
                        )
                else:
                    # Check if we need to filter rare classes even with full data
                    min_samples_per_class = max(tuning_config.get("cv_folds", 5) * 2, 10)
                    class_counts = y_train.value_counts()
                    valid_classes = class_counts[class_counts >= min_samples_per_class].index
                    
                    if len(valid_classes) < len(class_counts):
                        valid_mask = y_train.isin(valid_classes)
                        X_tune = X_train[valid_mask]
                        y_tune = y_train[valid_mask]
                        logger.info(
                            f"Filtered {len(class_counts) - len(valid_classes)} rare classes for tuning. "
                            f"Using {len(X_tune)} samples with {len(valid_classes)} classes."
                        )
                    else:
                        X_tune, y_tune = X_train, y_train
                
                # Check if we should use cross-domain CV for better generalization
                use_cross_domain = (
                    tuning_config.get("use_cross_domain_cv", False) and 
                    experiment_metadata.get("train_site") != experiment_metadata.get("test_site")
                )
                
                if use_cross_domain and 'site' in X_train.columns:
                    # Use cross-domain CV if we have site information
                    logger.info("Using cross-domain CV for hyperparameter tuning")
                    
                    # For cross-domain CV, we need the site column
                    site_col = X_train['site'] if 'site' in X_train.columns else pd.Series([experiment_metadata['train_site']] * len(X_train))
                    
                    # Create a custom scoring function for Ray Tune
                    def cross_domain_objective(config):
                        results = evaluate_cross_domain_performance(
                            model_name=model_name,
                            params=config,
                            X=X_tune.drop(columns=['site']) if 'site' in X_tune.columns else X_tune,
                            y=y_tune,
                            sites=site_col.iloc[X_tune.index] if len(site_col) > len(X_tune) else site_col,
                            metric=tuning_config.get("tuning_metric", "csmf_accuracy"),
                            n_splits=min(tuning_config.get("cv_folds", 5), len(site_col.unique())),
                        )
                        return {tuning_config.get("tuning_metric", "csmf_accuracy"): results["mean"]}
                    
                    # Note: This requires modifying RayTuner to accept custom objectives
                    # For now, fall back to regular tuning with a warning
                    logger.warning("Cross-domain CV requested but not fully implemented in RayTuner. Using standard CV.")
                    unique_experiment_name = f"{experiment_metadata.get('experiment_id')}_{model_name}_{uuid.uuid4().hex[:8]}"
                    tuning_results = tuner.tune_model(
                        model_name=model_name,
                        search_space=search_space,
                        train_data=(X_tune, y_tune),
                        cv_folds=tuning_config.get("cv_folds", 5),
                        experiment_name=unique_experiment_name,
                    )
                else:
                    # Standard tuning
                    unique_experiment_name = f"{experiment_metadata.get('experiment_id')}_{model_name}_{uuid.uuid4().hex[:8]}"
                    tuning_results = tuner.tune_model(
                        model_name=model_name,
                        search_space=search_space,
                        train_data=(X_tune, y_tune),
                        cv_folds=tuning_config.get("cv_folds", 5),
                        experiment_name=unique_experiment_name,
                    )
                
                tuning_time = time.time() - tuning_start
                best_params = tuning_results["best_params"]
                logger.info(
                    f"Tuning completed in {tuning_time:.1f}s. Best {tuning_config.get('tuning_metric', 'csmf_accuracy')}: "
                    f"{tuning_results['best_score']:.4f}"
                )
                
                # Create model with best parameters
                # Keep the config__ prefix for nested parameters
                model = model_class()
                model.set_params(**best_params)
            else:
                # Use default parameters
                if model_name == "xgboost":
                    model = XGBoostModel()
                elif model_name == "random_forest":
                    model = RandomForestModel()
                elif model_name == "logistic_regression":
                    model = LogisticRegressionModel()
                elif model_name == "categorical_nb":
                    model = CategoricalNBModel()
                else:
                    raise ValueError(f"Unknown model: {model_name}")
        
        # Train model
        logger.info(f"Training {model_name} on {len(X_train)} samples")
        training_start = time.time()
        model.fit(X_train_processed, y_train)
        training_time = time.time() - training_start
        
        # Make predictions
        inference_start = time.time()
        y_pred = model.predict(X_test_processed)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_processed)
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test, y_pred=y_pred, y_proba=y_proba, n_bootstrap=n_bootstrap
        )
        inference_time = time.time() - inference_start
        
        # Add tuning metadata to experiment metadata
        if tuning_results:
            experiment_metadata["tuning_results"] = {
                "best_params": best_params,
                "best_score": tuning_results["best_score"],
                "n_trials": tuning_results["n_trials_completed"],
            }
        
        # Create result
        result = ExperimentResult(
            experiment_id=experiment_metadata["experiment_id"],
            model_name=model_name,
            experiment_type=experiment_metadata["experiment_type"],
            train_site=experiment_metadata["train_site"],
            test_site=experiment_metadata["test_site"],
            training_size=experiment_metadata.get("training_size", 1.0),
            csmf_accuracy=metrics["csmf_accuracy"],
            cod_accuracy=metrics["cod_accuracy"],
            csmf_accuracy_ci=metrics.get("csmf_accuracy_ci") if isinstance(metrics.get("csmf_accuracy_ci"), list) else None,
            cod_accuracy_ci=metrics.get("cod_accuracy_ci") if isinstance(metrics.get("cod_accuracy_ci"), list) else None,
            n_train=len(y_train),
            n_test=len(y_test),
            execution_time_seconds=time.time() - start_time,
            tuning_time_seconds=tuning_time if 'tuning_time' in locals() else None,
            training_time_seconds=training_time,
            inference_time_seconds=inference_time,
            worker_id=ray.get_runtime_context().get_worker_id(),
            retry_count=retry_count,
        )
        
        logger.info(
            f"Completed: {model_name} - CSMF: {metrics['csmf_accuracy']:.3f}, "
            f"COD: {metrics['cod_accuracy']:.3f}"
        )
        
        return result
        
    except Exception as e:
        # Log error
        error_msg = f"Error in {model_name}: {str(e)}\n{traceback.format_exc()}"
        logger = get_logger(__name__, component="orchestration", console=False)
        logger.error(error_msg)
        
        # Return result with error
        return ExperimentResult(
            experiment_id=experiment_metadata["experiment_id"],
            model_name=model_name,
            experiment_type=experiment_metadata["experiment_type"],
            train_site=experiment_metadata["train_site"],
            test_site=experiment_metadata["test_site"],
            training_size=experiment_metadata.get("training_size", 1.0),
            csmf_accuracy=0.0,
            cod_accuracy=0.0,
            n_train=0,
            n_test=0,
            execution_time_seconds=time.time() - start_time,
            worker_id=ray.get_runtime_context().get_worker_id(),
            retry_count=retry_count,
            error=str(e),
        )


@ray.remote
def prepare_data_for_site(
    data: pd.DataFrame, site: str, test_size: float = 0.2, random_seed: int = 42
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Prepare train/test split for a specific site.

    This is a Ray remote function to parallelize data preparation.

    Args:
        data: Full dataset
        site: Site to filter for
        test_size: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) or None if insufficient data
    """
    try:
        from sklearn.model_selection import train_test_split

        # Filter to site
        site_data = data[data["site"] == site]

        if len(site_data) < 50:  # Skip sites with too little data
            return None

        # Drop label columns - must match VADataProcessor._get_label_equivalent_columns()
        # This ensures no label-equivalent columns leak into features
        label_columns = [
            "cause",  # Primary label column used by all models
            "site",  # Site information (used for stratification, not features)
            "module",  # Module type
            "gs_code34", "gs_text34", "va34",  # 34-cause classification
            "gs_code46", "gs_text46", "va46",  # 46-cause classification  
            "gs_code55", "gs_text55", "va55",  # 55-cause classification
            "gs_comorbid1", "gs_comorbid2",  # Comorbidity information
            "gs_level",  # Gold standard level
            "cod5",  # 5-cause grouping
            "newid"  # ID column
        ]
        columns_to_drop = [col for col in label_columns if col in site_data.columns]
        X = site_data.drop(columns=columns_to_drop)
        y = site_data["cause"]  # All models use "cause" as the label
        
        # Log feature column count for consistency validation
        logger = get_logger(__name__, component="orchestration", console=False)
        logger.info(
            f"Site {site}: {len(X.columns)} features after dropping {len(columns_to_drop)} columns"
        )

        # Try stratified split, fall back to random if necessary
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger = get_logger(__name__, component="orchestration", console=False)
        logger.error(f"Error preparing data for site {site}: {e}")
        return None


@ray.remote
class ProgressReporter:
    """Ray actor for centralized progress reporting."""

    def __init__(self, total_experiments: int):
        """Initialize progress reporter.

        Args:
            total_experiments: Total number of experiments to track
        """
        self.total_experiments = total_experiments
        self.completed_experiments = 0
        self.failed_experiments = 0
        self.start_time = time.time()
        self.results: List[ExperimentResult] = []

    def report_completion(self, result: ExperimentResult) -> None:
        """Report experiment completion."""
        self.completed_experiments += 1
        if result.error:
            self.failed_experiments += 1
        self.results.append(result)

    def get_progress(self) -> Dict:
        """Get current progress statistics."""
        elapsed = time.time() - self.start_time
        completion_rate = self.completed_experiments / self.total_experiments

        if self.completed_experiments > 0:
            avg_time_per_experiment = elapsed / self.completed_experiments
            estimated_remaining = (
                self.total_experiments - self.completed_experiments
            ) * avg_time_per_experiment
        else:
            estimated_remaining = 0

        return {
            "total": self.total_experiments,
            "completed": self.completed_experiments,
            "failed": self.failed_experiments,
            "completion_rate": completion_rate,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": estimated_remaining,
        }

    def get_results(self) -> list:
        """Get all collected results."""
        return self.results