"""Parallel experiment runner extending SiteComparisonExperiment."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ray
from sklearn.model_selection import train_test_split

from baseline.utils import get_logger
from model_comparison.experiments.experiment_config import ExperimentConfig
from model_comparison.experiments.site_comparison import SiteComparisonExperiment
from model_comparison.monitoring.progress_tracker import (
    PerformanceMonitor,
    RayProgressTracker,
)
from model_comparison.orchestration.checkpoint_manager import CheckpointManager
from model_comparison.orchestration.config import ExperimentResult, ParallelConfig
from model_comparison.orchestration.ray_tasks import (
    ProgressReporter,
    train_and_evaluate_model,
)

logger = get_logger(__name__, component="model_comparison")


class ParallelSiteComparisonExperiment(SiteComparisonExperiment):
    """Parallel version of SiteComparisonExperiment using Ray."""

    def __init__(self, config: ExperimentConfig, parallel_config: ParallelConfig):
        """Initialize parallel experiment runner.

        Args:
            config: Experiment configuration
            parallel_config: Parallel execution configuration
        """
        super().__init__(config)
        self.parallel_config = parallel_config
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(Path(self.config.output_dir) / "checkpoints")
        )
        self.performance_monitor = PerformanceMonitor()

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(**self.parallel_config.to_ray_init_kwargs())
            logger.info(
                f"Ray initialized with dashboard at "
                f"http://localhost:{self.parallel_config.ray_dashboard_port}"
            )

    def run_experiment(self) -> pd.DataFrame:
        """Run complete experiment in parallel."""
        logger.info("Starting parallel VA34 site comparison experiment")
        start_time = time.time()

        # Load and prepare data
        data = self._load_data()
        logger.info(f"Loaded data with shape: {data.shape}")

        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(self.config.model_dump())
        existing_results = []
        if checkpoint:
            existing_results = self.checkpoint_manager.load_partial_results(checkpoint)
            logger.info(f"Resuming from checkpoint with {len(existing_results)} results")

        # Generate all experiment configurations
        all_experiments = self._generate_all_experiments(data)

        # Filter out completed experiments if resuming
        if checkpoint:
            all_experiments = self.checkpoint_manager.filter_completed_experiments(
                all_experiments, checkpoint
            )

        # Initialize progress tracking
        total_experiments = len(all_experiments) + len(existing_results)
        progress_actor = ProgressReporter.remote(total_experiments)
        progress_tracker = RayProgressTracker(
            total_experiments=total_experiments,
            description="VA34 Parallel Experiments",
            progress_actor=progress_actor,
        )

        # Add existing results to progress
        for result in existing_results:
            progress_tracker.update(result)

        # Run experiments in parallel
        all_results = existing_results + self._run_experiments_parallel(
            all_experiments, progress_tracker
        )

        # Convert to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in all_results])
        
        # Rename model_name to model for compatibility with base class
        if "model_name" in results_df.columns:
            results_df = results_df.rename(columns={"model_name": "model"})
            
        # Add n_train and n_test columns if not present (for compatibility)
        if "n_train" not in results_df.columns:
            results_df["n_train"] = 0  # These will be filled by the model training
        if "n_test" not in results_df.columns:
            results_df["n_test"] = 0

        # Save results
        self._save_results(results_df)

        # Generate visualizations if requested
        if self.config.generate_plots:
            logger.info("Generating visualizations...")
            self._generate_visualizations(results_df)

        # Log performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        logger.info(f"Performance summary: {perf_summary}")

        total_time = time.time() - start_time
        logger.info(
            f"Parallel experiment completed in {total_time:.1f}s "
            f"({len(all_results)} experiments)"
        )

        return results_df

    def _generate_all_experiments(self, data: pd.DataFrame) -> List[Dict]:
        """Generate all experiment configurations.

        Args:
            data: Full dataset

        Returns:
            List of experiment configurations
        """
        experiments = []

        # Prepare data splits for all sites
        site_data_splits = self._prepare_all_site_data(data)

        # In-domain experiments
        for site, splits in site_data_splits.items():
            if splits is None:
                continue

            for model_name in self.config.models:
                experiment_id = self.checkpoint_manager.create_experiment_id(
                    model_name=model_name,
                    experiment_type="in_domain",
                    train_site=site,
                    test_site=site,
                )

                experiments.append(
                    {
                        "model_name": model_name,
                        "train_data": ray.put(splits["train_data"]),
                        "test_data": ray.put(splits["test_data"]),
                        "experiment_metadata": {
                            "experiment_id": experiment_id,
                            "experiment_type": "in_domain",
                            "train_site": site,
                            "test_site": site,
                        },
                        "n_bootstrap": self.config.n_bootstrap,
                    }
                )

        # Out-domain experiments
        for train_site, train_splits in site_data_splits.items():
            if train_splits is None:
                continue

            for test_site, test_splits in site_data_splits.items():
                if test_splits is None or train_site == test_site:
                    continue

                for model_name in self.config.models:
                    experiment_id = self.checkpoint_manager.create_experiment_id(
                        model_name=model_name,
                        experiment_type="out_domain",
                        train_site=train_site,
                        test_site=test_site,
                    )

                    experiments.append(
                        {
                            "model_name": model_name,
                            "train_data": ray.put(train_splits["train_data"]),
                            "test_data": ray.put(test_splits["test_data"]),
                            "experiment_metadata": {
                                "experiment_id": experiment_id,
                                "experiment_type": "out_domain",
                                "train_site": train_site,
                                "test_site": test_site,
                            },
                            "n_bootstrap": self.config.n_bootstrap,
                        }
                    )

        # Training size experiments
        if self.config.sites:
            primary_site = self.config.sites[0]
            if primary_site in site_data_splits and site_data_splits[primary_site]:
                base_splits = site_data_splits[primary_site]

                for training_size in self.config.training_sizes:
                    # Subsample training data
                    X_train_full, y_train_full = base_splits["train_data"]
                    n_samples = int(len(X_train_full) * training_size)

                    if n_samples < 10:  # Skip if too few samples
                        continue

                    X_train_subset = X_train_full.iloc[:n_samples]
                    y_train_subset = y_train_full.iloc[:n_samples]

                    for model_name in self.config.models:
                        experiment_id = self.checkpoint_manager.create_experiment_id(
                            model_name=model_name,
                            experiment_type="training_size",
                            train_site=primary_site,
                            test_site=primary_site,
                            training_size=training_size,
                        )

                        experiments.append(
                            {
                                "model_name": model_name,
                                "train_data": ray.put((X_train_subset, y_train_subset)),
                                "test_data": ray.put(base_splits["test_data"]),
                                "experiment_metadata": {
                                    "experiment_id": experiment_id,
                                    "experiment_type": "training_size",
                                    "train_site": primary_site,
                                    "test_site": primary_site,
                                    "training_size": training_size,
                                },
                                "n_bootstrap": self.config.n_bootstrap,
                            }
                        )

        logger.info(f"Generated {len(experiments)} experiment configurations")
        return experiments

    def _prepare_all_site_data(
        self, data: pd.DataFrame
    ) -> Dict[str, Optional[Dict[str, Tuple]]]:
        """Prepare train/test splits for all sites.

        Args:
            data: Full dataset

        Returns:
            Dictionary mapping site to data splits
        """
        site_splits = {}

        for site in self.config.sites:
            site_data = data[data["site"] == site]

            if len(site_data) < 50:
                logger.warning(f"Skipping site {site} - insufficient data")
                site_splits[site] = None
                continue

            # Drop label columns
            label_columns = ["cause", "site", "va34", "cod5"]
            columns_to_drop = [col for col in label_columns if col in site_data.columns]
            X = site_data.drop(columns=columns_to_drop)
            y = site_data["cause"]

            # Try stratified split, fall back to random if necessary
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.random_seed, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.config.random_seed
                )

            site_splits[site] = {
                "train_data": (X_train, y_train),
                "test_data": (X_test, y_test),
            }

        return site_splits

    def _run_experiments_parallel(
        self, experiments: List[Dict], progress_tracker: RayProgressTracker
    ) -> List[ExperimentResult]:
        """Run experiments in parallel using Ray.

        Args:
            experiments: List of experiment configurations
            progress_tracker: Progress tracker

        Returns:
            List of experiment results
        """
        results = []

        # Batch experiments to avoid overwhelming Ray
        batch_size = self.parallel_config.batch_size
        experiment_batches = [
            experiments[i : i + batch_size]
            for i in range(0, len(experiments), batch_size)
        ]

        for batch_idx, batch in enumerate(experiment_batches):
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(experiment_batches)} "
                f"({len(batch)} experiments)"
            )

            # Submit batch to Ray
            result_refs = []
            for exp in batch:
                result_ref = train_and_evaluate_model.remote(**exp)
                result_refs.append(result_ref)

            # Wait for results with progress updates
            batch_results = []
            while result_refs:
                ready_refs, result_refs = ray.wait(
                    result_refs, num_returns=1, timeout=1.0
                )

                if ready_refs:
                    ready_results = ray.get(ready_refs)
                    for result in ready_results:
                        batch_results.append(result)
                        progress_tracker.update(result)
                        self.performance_monitor.record_experiment(result)

            results.extend(batch_results)

            # Save checkpoint periodically
            if (batch_idx + 1) % self.parallel_config.checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    results,
                    self.config.model_dump(),
                    len(experiments),
                    progress_tracker.get_statistics()["elapsed_seconds"],
                )

        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(
            results,
            self.config.model_dump(),
            len(experiments),
            progress_tracker.get_statistics()["elapsed_seconds"],
        )

        progress_tracker.close()
        return results

    def cleanup(self):
        """Clean up resources."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")