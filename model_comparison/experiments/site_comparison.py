"""Main experiment runner for VA34 site-based model comparison."""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from baseline.data.data_loader import VADataProcessor
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.random_forest_model import RandomForestModel
from baseline.utils import get_logger

from ..metrics.comparison_metrics import calculate_metrics
from .experiment_config import ExperimentConfig

logger = get_logger(__name__, component="model_comparison")


class SiteComparisonExperiment:
    """VA34 site-based model comparison experiment."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration."""
        self.config = config
        self.processor = VADataProcessor()
        self.results = []
        self.progress_callback: Optional[Callable] = None
        self.checkpoint_callback: Optional[Callable] = None
        self.enable_parallel = False

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates.

        Args:
            callback: Function to call with progress updates
        """
        self.progress_callback = callback

    def set_checkpoint_callback(self, callback: Callable) -> None:
        """Set callback for checkpoint saves.

        Args:
            callback: Function to call for checkpoint saves
        """
        self.checkpoint_callback = callback

    def enable_parallel_execution(self, enabled: bool = True) -> None:
        """Enable or disable parallel execution mode.

        Args:
            enabled: Whether to enable parallel mode
        """
        self.enable_parallel = enabled

    def run_experiment(self) -> pd.DataFrame:
        """Run complete experiment across all configurations."""
        logger.info("Starting VA34 site comparison experiment")

        # Load and prepare data
        data = self._load_data()
        logger.info(f"Loaded data with shape: {data.shape}")

        # Run in-domain experiments
        logger.info("Running in-domain experiments...")
        in_domain_results = self._run_in_domain_experiments(data)
        
        # Save checkpoint after in-domain experiments
        if self.checkpoint_callback and len(in_domain_results) > 0:
            self.checkpoint_callback(in_domain_results, "in_domain_complete")

        # Run out-domain experiments
        logger.info("Running out-domain experiments...")
        out_domain_results = self._run_out_domain_experiments(data)
        
        # Save checkpoint after out-domain experiments
        if self.checkpoint_callback and len(out_domain_results) > 0:
            self.checkpoint_callback(out_domain_results, "out_domain_complete")

        # Run training size experiments
        logger.info("Running training size experiments...")
        size_results = self._run_training_size_experiments(data)
        
        # Save checkpoint after training size experiments
        if self.checkpoint_callback and len(size_results) > 0:
            self.checkpoint_callback(size_results, "training_size_complete")

        # Combine all results
        all_results = pd.concat(
            [in_domain_results, out_domain_results, size_results], ignore_index=True
        )

        # Save results
        self._save_results(all_results)

        # Generate visualizations if requested
        if self.config.generate_plots:
            logger.info("Generating visualizations...")
            self._generate_visualizations(all_results)

        logger.info("Experiment completed successfully")
        return all_results

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare VA data."""
        # Load data using VADataProcessor
        data = self.processor.load_data(self.config.data_path)

        # Filter to specified sites if provided
        if self.config.sites:
            data = data[data["site"].isin(self.config.sites)]
            logger.info(f"Filtered to {len(self.config.sites)} sites")

        # Handle va34 column (rename to cause for compatibility)
        if "va34" in data.columns and "cause" not in data.columns:
            data["cause"] = data["va34"].astype(str)  # Convert to string for consistency
            logger.info("Renamed va34 column to cause")
            # Don't drop va34 yet - will be dropped with other label columns later

        # Ensure we have the expected columns
        required_cols = ["site", "cause"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        return data

    def _get_model(self, model_name: str):
        """Get model instance by name."""
        if model_name == "insilico":
            return InSilicoVAModel()
        elif model_name == "xgboost":
            return XGBoostModel()
        elif model_name == "random_forest":
            return RandomForestModel()
        elif model_name == "logistic_regression":
            from baseline.models import LogisticRegressionModel
            return LogisticRegressionModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for model training.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Preprocessed features DataFrame
        """
        X_processed = X.copy()
        
        # Encode categorical features
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                # Simple label encoding for categorical features
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        return X_processed

    def _run_in_domain_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Test performance when training and testing on same site."""
        results = []

        for site in tqdm(self.config.sites, desc="In-domain sites"):
            # Get site data
            site_data = data[data["site"] == site]

            if len(site_data) < 50:  # Skip sites with too little data
                logger.warning(f"Skipping site {site} - insufficient data")
                continue

            # Split into train/test using simple train_test_split
            from sklearn.model_selection import train_test_split
            
            # Drop all label columns to avoid data leakage
            label_columns = ["cause", "site", "va34", "cod5"]
            columns_to_drop = [col for col in label_columns if col in site_data.columns]
            X = site_data.drop(columns=columns_to_drop)
            y = site_data["cause"]
            
            # Try stratified split, fall back to random if necessary
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.random_seed, stratify=y
                )
            except ValueError as e:
                if "least populated class" in str(e):
                    logger.warning(f"Stratified split failed for {site}, using random split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=self.config.random_seed
                    )
                else:
                    raise

            # Train and evaluate each model
            for model_name in self.config.models:
                try:
                    result = self._evaluate_model(
                        model_name=model_name,
                        train_data=(X_train, y_train),
                        test_data=(X_test, y_test),
                        experiment_type="in_domain",
                        train_site=site,
                        test_site=site,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {site}: {e}")

        return pd.DataFrame(results)

    def _run_out_domain_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Test performance when training on one site and testing on others."""
        results = []

        for train_site in tqdm(self.config.sites, desc="Out-domain train sites"):
            # Get training data
            train_data = data[data["site"] == train_site]

            if len(train_data) < 50:
                logger.warning(f"Skipping train site {train_site} - insufficient data")
                continue

            # Prepare training data - drop all label columns
            label_columns = ["cause", "site", "va34", "cod5"]
            columns_to_drop = [col for col in label_columns if col in train_data.columns]
            X_train = train_data.drop(columns=columns_to_drop)
            y_train = train_data["cause"]

            # Test on other sites
            test_sites = [s for s in self.config.sites if s != train_site]

            for test_site in test_sites:
                # Get test data
                test_data = data[data["site"] == test_site]

                if len(test_data) < 20:
                    logger.warning(
                        f"Skipping test site {test_site} - insufficient data"
                    )
                    continue

                # Drop all label columns
                label_columns = ["cause", "site", "va34", "cod5"]
                columns_to_drop = [col for col in label_columns if col in test_data.columns]
                X_test = test_data.drop(columns=columns_to_drop)
                y_test = test_data["cause"]

                # Train and evaluate each model
                for model_name in self.config.models:
                    try:
                        result = self._evaluate_model(
                            model_name=model_name,
                            train_data=(X_train, y_train),
                            test_data=(X_test, y_test),
                            experiment_type="out_domain",
                            train_site=train_site,
                            test_site=test_site,
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(
                            f"Error evaluating {model_name} "
                            f"{train_site}->{test_site}: {e}"
                        )

        return pd.DataFrame(results)

    def _run_training_size_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Test impact of training data size on performance."""
        results = []

        # Use first site as primary training site
        primary_site = self.config.sites[0]
        train_data = data[data["site"] == primary_site]

        if len(train_data) < 100:
            logger.warning("Insufficient data for training size experiments")
            return pd.DataFrame()

        # Split data for consistent test set
        from sklearn.model_selection import train_test_split
        
        # Drop all label columns
        label_columns = ["cause", "site", "va34", "cod5"]
        columns_to_drop = [col for col in label_columns if col in train_data.columns]
        X = train_data.drop(columns=columns_to_drop)
        y = train_data["cause"]
        
        # Try stratified split, fall back to random if necessary
        try:
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.config.random_seed, stratify=y
            )
        except ValueError as e:
            if "least populated class" in str(e):
                logger.warning(f"Stratified split failed for training size exp, using random split")
                X_train_full, X_test, y_train_full, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.random_seed
                )
            else:
                raise

        # Test different training sizes
        for train_size in tqdm(
            self.config.training_sizes, desc="Training size fractions"
        ):
            # Sample training data
            n_train = int(len(X_train_full) * train_size)
            if n_train < 10:
                continue

            indices = np.random.RandomState(self.config.random_seed).choice(
                len(X_train_full), n_train, replace=False
            )

            X_train = X_train_full.iloc[indices]
            y_train = y_train_full.iloc[indices]

            # Evaluate each model
            for model_name in self.config.models:
                try:
                    result = self._evaluate_model(
                        model_name=model_name,
                        train_data=(X_train, y_train),
                        test_data=(X_test, y_test),
                        experiment_type="training_size",
                        train_site=primary_site,
                        test_site=primary_site,
                        training_fraction=train_size,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error evaluating {model_name} " f"with size {train_size}: {e}"
                    )

        return pd.DataFrame(results)

    def _evaluate_model(
        self,
        model_name: str,
        train_data: Tuple[pd.DataFrame, pd.Series],
        test_data: Tuple[pd.DataFrame, pd.Series],
        experiment_type: str,
        train_site: str,
        test_site: str,
        training_fraction: float = 1.0,
    ) -> Dict:
        """Evaluate a model on given train/test data."""
        X_train, y_train = train_data
        X_test, y_test = test_data

        # Preprocess features for ML models (XGBoost, Random Forest)
        if model_name in ["xgboost", "random_forest"]:
            X_train = self._preprocess_features(X_train)
            X_test = self._preprocess_features(X_test)

        # Get model instance
        model = self._get_model(model_name)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            n_bootstrap=self.config.n_bootstrap,
        )

        # Prepare result
        result = {
            "experiment_type": experiment_type,
            "train_site": train_site,
            "test_site": test_site,
            "model": model_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "training_fraction": training_fraction,
            **metrics,
        }

        # Call progress callback if set
        if self.progress_callback:
            self.progress_callback(result)

        return result

    def _save_results(self, results: pd.DataFrame):
        """Save experiment results to files."""
        # Save full results
        results_path = Path(self.config.output_dir) / "full_results.csv"
        results.to_csv(results_path, index=False)
        logger.info(f"Saved full results to {results_path}")

        # Save subset results
        for exp_type in results["experiment_type"].unique():
            subset = results[results["experiment_type"] == exp_type]
            subset_path = Path(self.config.output_dir) / f"{exp_type}_results.csv"
            subset.to_csv(subset_path, index=False)

        # Save summary statistics
        summary = self._calculate_summary_statistics(results)
        summary_path = Path(self.config.output_dir) / "summary_statistics.csv"
        summary.to_csv(summary_path)
        logger.info(f"Saved summary statistics to {summary_path}")

    def _calculate_summary_statistics(self, results: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics from results."""
        # Group by experiment type and model
        summary = (
            results.groupby(["experiment_type", "model"])
            .agg(
                {
                    "csmf_accuracy": ["mean", "std", "min", "max"],
                    "cod_accuracy": ["mean", "std", "min", "max"],
                    "n_train": "mean",
                    "n_test": "mean",
                }
            )
            .round(4)
        )

        return summary

    def _generate_visualizations(self, results: pd.DataFrame):
        """Generate visualization plots."""
        # Import here to avoid circular dependency
        from ..visualization.comparison_plots import plot_model_comparison

        output_path = Path(self.config.output_dir) / "model_comparison.png"
        plot_model_comparison(results, str(output_path))
        logger.info(f"Saved visualization to {output_path}")
