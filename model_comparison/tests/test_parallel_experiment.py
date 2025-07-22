"""Tests for parallel experiment execution."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray

from model_comparison.experiments.experiment_config import ExperimentConfig
from model_comparison.experiments.parallel_experiment import (
    ParallelSiteComparisonExperiment,
)
from model_comparison.orchestration.config import ParallelConfig


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing."""
    ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def test_data_file():
    """Create a temporary test data file."""
    np.random.seed(42)
    n_samples = 300

    # Create synthetic VA data
    data = pd.DataFrame(
        {
            f"symptom_{i}": np.random.choice([0, 1], n_samples)
            for i in range(20)
        }
    )

    # Add cause labels
    data["va34"] = np.random.choice(["cause_A", "cause_B", "cause_C"], n_samples)
    
    # Add sites - ensure each site has at least 100 samples for tests to pass
    sites = []
    for site in ["site_1", "site_2", "site_3"]:
        sites.extend([site] * 100)
    data["site"] = sites[:n_samples]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def experiment_config(test_data_file, tmp_path):
    """Create experiment configuration."""
    return ExperimentConfig(
        data_path=test_data_file,
        sites=["site_1", "site_2", "site_3"],
        models=["xgboost"],  # Only XGBoost for faster tests
        training_sizes=[0.5, 1.0],
        n_bootstrap=10,  # Small for testing
        random_seed=42,
        output_dir=str(tmp_path / "results"),
        generate_plots=False,
    )


@pytest.fixture
def parallel_config():
    """Create parallel configuration for testing."""
    return ParallelConfig(
        n_workers=2,
        batch_size=10,
        checkpoint_interval=5,
        ray_dashboard=False,
        prefect_dashboard=False,
    )


class TestParallelSiteComparisonExperiment:
    """Test parallel experiment execution."""

    def test_initialization(self, ray_context, experiment_config, parallel_config):
        """Test parallel experiment initialization."""
        experiment = ParallelSiteComparisonExperiment(
            experiment_config, parallel_config
        )

        assert experiment.config == experiment_config
        assert experiment.parallel_config == parallel_config
        assert experiment.checkpoint_manager is not None
        assert ray.is_initialized()

    def test_generate_all_experiments(
        self, ray_context, experiment_config, parallel_config, test_data_file
    ):
        """Test generation of experiment configurations."""
        experiment = ParallelSiteComparisonExperiment(
            experiment_config, parallel_config
        )

        # Load data
        data = pd.read_csv(test_data_file)
        if "cause" not in data.columns:
            data["cause"] = data["va34"]

        # Generate experiments
        all_experiments = experiment._generate_all_experiments(data)

        # Check experiment count
        n_sites = len(experiment_config.sites)
        n_models = len(experiment_config.models)
        n_training_sizes = len(experiment_config.training_sizes)

        # Expected: in-domain + out-domain + training sizes
        expected_in_domain = n_sites * n_models
        expected_out_domain = n_sites * (n_sites - 1) * n_models
        expected_training_size = n_training_sizes * n_models

        # Some sites might be skipped due to insufficient data
        assert len(all_experiments) > 0
        assert len(all_experiments) <= (
            expected_in_domain + expected_out_domain + expected_training_size
        )

        # Check experiment structure
        for exp in all_experiments:
            assert "model_name" in exp
            assert "train_data" in exp
            assert "test_data" in exp
            assert "experiment_metadata" in exp
            assert "n_bootstrap" in exp

    def test_parallel_execution_small(
        self, ray_context, experiment_config, parallel_config, tmp_path
    ):
        """Test parallel execution with small dataset."""
        # Modify config for faster test
        experiment_config.sites = ["site_1", "site_2"]
        experiment_config.training_sizes = [1.0]
        experiment_config.output_dir = str(tmp_path / "parallel_test")

        experiment = ParallelSiteComparisonExperiment(
            experiment_config, parallel_config
        )

        # Run experiment
        results = experiment.run_experiment()

        # Validate results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert "model" in results.columns  # Renamed from model_name for compatibility
        assert "csmf_accuracy" in results.columns
        assert "cod_accuracy" in results.columns

        # Check that results were saved
        output_path = Path(experiment_config.output_dir) / "full_results.csv"
        assert output_path.exists()

        # Cleanup
        experiment.cleanup()

    def test_checkpoint_resume(
        self, ray_context, experiment_config, parallel_config, tmp_path
    ):
        """Test checkpoint and resume functionality."""
        experiment_config.sites = ["site_1", "site_2"]
        experiment_config.output_dir = str(tmp_path / "checkpoint_test")

        # First run - interrupt after some experiments
        experiment1 = ParallelSiteComparisonExperiment(
            experiment_config, parallel_config
        )

        # Generate experiments
        data = pd.read_csv(experiment_config.data_path)
        if "cause" not in data.columns:
            data["cause"] = data["va34"]
        all_experiments = experiment1._generate_all_experiments(data)

        # Run only first few experiments
        partial_experiments = all_experiments[:3]
        from model_comparison.monitoring.progress_tracker import RayProgressTracker

        progress_tracker = RayProgressTracker(
            total_experiments=len(partial_experiments), show_progress_bar=False
        )

        partial_results = experiment1._run_experiments_parallel(
            partial_experiments, progress_tracker
        )

        # Should have checkpoint
        checkpoint = experiment1.checkpoint_manager.load_checkpoint(
            experiment_config.model_dump()
        )
        assert checkpoint is not None
        assert len(checkpoint.completed_experiments) == 3

        # Cleanup first experiment
        experiment1.cleanup()

        # Second run - should resume from checkpoint
        experiment2 = ParallelSiteComparisonExperiment(
            experiment_config, parallel_config
        )

        # Run full experiment
        results = experiment2.run_experiment()

        # Should have all results (not just remaining)
        assert len(results) >= len(all_experiments)

        # Cleanup
        experiment2.cleanup()

    def test_error_handling(self, ray_context, experiment_config, parallel_config, tmp_path):
        """Test error handling in parallel execution."""
        # Create data that will cause some experiments to fail
        bad_data = pd.DataFrame(
            {
                "feature_1": [1, 2, 3] * 10,
                "va34": ["A"] * 30,  # Only one class - will fail stratification
                "site": ["site_1"] * 15 + ["site_2"] * 15,
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            bad_data.to_csv(f, index=False)
            bad_data_path = f.name

        try:
            experiment_config.data_path = bad_data_path
            experiment_config.output_dir = str(tmp_path / "error_test")

            experiment = ParallelSiteComparisonExperiment(
                experiment_config, parallel_config
            )

            # Should handle errors gracefully
            results = experiment.run_experiment()

            # Some experiments might succeed, some might fail
            assert isinstance(results, pd.DataFrame)

            # Cleanup
            experiment.cleanup()

        finally:
            Path(bad_data_path).unlink(missing_ok=True)