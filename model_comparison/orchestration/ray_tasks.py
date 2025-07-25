"""Ray remote functions for parallel VA model training and evaluation.

This module ensures consistent data preprocessing across all models:
- All models use "cause" as the label column (created from "va34" if needed)
- The same set of label-equivalent columns are dropped for all models
- Feature columns are identical across all model types
"""

import time
import traceback
from typing import Dict, Optional, Tuple

import pandas as pd
import ray

from baseline.utils import get_logger
from model_comparison.orchestration.config import ExperimentResult


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
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Preprocess features only for ML models (not InSilicoVA)
        if model_name == "insilico":
            # InSilicoVA needs the original "Y"/"." format
            X_train_processed = X_train
            X_test_processed = X_test
        else:
            # Other models need numeric encoding
            X_train_processed = _preprocess_features(X_train)
            X_test_processed = _preprocess_features(X_test)

        # Train model
        logger.info(f"Training {model_name} on {len(X_train)} samples (training_size={training_size})")
        model.fit(X_train_processed, y_train)

        # Make predictions
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