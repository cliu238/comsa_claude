"""Tests for Ray remote tasks."""

import numpy as np
import pandas as pd
import pytest
import ray

from model_comparison.orchestration.config import ExperimentResult
from model_comparison.orchestration.ray_tasks import (
    ProgressReporter,
    prepare_data_for_site,
    train_and_evaluate_model,
)


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing."""
    ray.init(local_mode=True)
    yield
    ray.shutdown()


@pytest.fixture
def test_data():
    """Create test dataset."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Create synthetic data
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice(["A", "B", "C"], n_samples))

    # Add site column
    data = X.copy()
    data["cause"] = y
    data["site"] = np.random.choice(["site_1", "site_2"], n_samples)

    return data


class TestRayTasks:
    """Test Ray remote functions."""

    def test_train_and_evaluate_model_xgboost(self, ray_context, test_data):
        """Test XGBoost model training and evaluation."""
        # Prepare data
        X_train = test_data.drop(columns=["cause", "site"]).iloc[:100]
        y_train = test_data["cause"].iloc[:100]
        X_test = test_data.drop(columns=["cause", "site"]).iloc[100:]
        y_test = test_data["cause"].iloc[100:]

        # Create experiment metadata
        metadata = {
            "experiment_id": "test_xgboost_1",
            "experiment_type": "test",
            "train_site": "site_1",
            "test_site": "site_1",
        }

        # Run remote task
        result_ref = train_and_evaluate_model.remote(
            model_name="xgboost",
            train_data=(X_train, y_train),
            test_data=(X_test, y_test),
            experiment_metadata=metadata,
            n_bootstrap=10,  # Small for testing
        )

        result = ray.get(result_ref)

        # Validate result
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == "test_xgboost_1"
        assert result.model_name == "xgboost"
        assert 0 <= result.csmf_accuracy <= 1
        assert 0 <= result.cod_accuracy <= 1
        assert result.execution_time_seconds > 0
        assert result.error is None

    def test_train_and_evaluate_model_error_handling(self, ray_context):
        """Test error handling in remote task."""
        # Create invalid data
        X_train = pd.DataFrame({"col1": [1, 2, 3]})
        y_train = pd.Series([1, 2])  # Mismatched length

        metadata = {
            "experiment_id": "test_error",
            "experiment_type": "test",
            "train_site": "site_1",
            "test_site": "site_1",
        }

        # Run should not raise but return error in result
        result_ref = train_and_evaluate_model.remote(
            model_name="xgboost",
            train_data=(X_train, y_train),
            test_data=(X_train, y_train),
            experiment_metadata=metadata,
        )

        result = ray.get(result_ref)

        assert isinstance(result, ExperimentResult)
        assert result.error is not None
        assert result.csmf_accuracy == 0.0
        assert result.cod_accuracy == 0.0

    def test_prepare_data_for_site(self, ray_context, test_data):
        """Test data preparation for site."""
        # Test with sufficient data
        result_ref = prepare_data_for_site.remote(
            test_data, "site_1", test_size=0.2, random_seed=42
        )
        result = ray.get(result_ref)

        assert result is not None
        X_train, X_test, y_train, y_test = result
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == len(test_data[test_data["site"] == "site_1"])

    def test_prepare_data_insufficient_samples(self, ray_context):
        """Test data preparation with insufficient samples."""
        # Create small dataset
        small_data = pd.DataFrame(
            {
                "feature_1": range(10),
                "cause": ["A"] * 10,
                "site": ["site_1"] * 10,
            }
        )

        result_ref = prepare_data_for_site.remote(small_data, "site_1")
        result = ray.get(result_ref)

        assert result is None  # Should return None for insufficient data

    def test_progress_reporter(self, ray_context):
        """Test ProgressReporter actor."""
        # Create reporter
        reporter = ProgressReporter.remote(total_experiments=10)

        # Report some completions
        results = []
        for i in range(5):
            result = ExperimentResult(
                experiment_id=f"test_{i}",
                model_name="xgboost",
                experiment_type="test",
                train_site="site_1",
                test_site="site_1",
                csmf_accuracy=0.8,
                cod_accuracy=0.7,
                execution_time_seconds=1.0,
            )
            ray.get(reporter.report_completion.remote(result))
            results.append(result)

        # Get progress
        progress = ray.get(reporter.get_progress.remote())

        assert progress["total"] == 10
        assert progress["completed"] == 5
        assert progress["failed"] == 0
        assert progress["completion_rate"] == 0.5
        assert progress["elapsed_seconds"] > 0

        # Get results
        reported_results = ray.get(reporter.get_results.remote())
        assert len(reported_results) == 5


class TestParallelExecution:
    """Test parallel execution patterns."""

    def test_batch_parallel_execution(self, ray_context, test_data):
        """Test batch parallel execution of experiments."""
        # Prepare multiple experiments
        experiments = []
        for i in range(10):
            X_train = test_data.drop(columns=["cause", "site"]).iloc[:100]
            y_train = test_data["cause"].iloc[:100]
            X_test = test_data.drop(columns=["cause", "site"]).iloc[100:]
            y_test = test_data["cause"].iloc[100:]

            experiments.append(
                {
                    "model_name": "xgboost",
                    "train_data": (X_train, y_train),
                    "test_data": (X_test, y_test),
                    "experiment_metadata": {
                        "experiment_id": f"batch_test_{i}",
                        "experiment_type": "test",
                        "train_site": "site_1",
                        "test_site": "site_1",
                    },
                    "n_bootstrap": 10,
                }
            )

        # Submit all experiments
        result_refs = [train_and_evaluate_model.remote(**exp) for exp in experiments]

        # Wait for all results
        results = ray.get(result_refs)

        assert len(results) == 10
        assert all(isinstance(r, ExperimentResult) for r in results)
        assert all(r.error is None for r in results)

    def test_progressive_result_collection(self, ray_context, test_data):
        """Test progressive collection of results as they complete."""
        # Submit multiple experiments with varying execution times
        experiments = []
        for i in range(5):
            X_train = test_data.drop(columns=["cause", "site"]).iloc[:100]
            y_train = test_data["cause"].iloc[:100]
            X_test = test_data.drop(columns=["cause", "site"]).iloc[100:]
            y_test = test_data["cause"].iloc[100:]

            experiments.append(
                train_and_evaluate_model.remote(
                    model_name="xgboost",
                    train_data=(X_train, y_train),
                    test_data=(X_test, y_test),
                    experiment_metadata={
                        "experiment_id": f"progressive_test_{i}",
                        "experiment_type": "test",
                        "train_site": "site_1",
                        "test_site": "site_1",
                    },
                    n_bootstrap=10 * (i + 1),  # Varying computation time
                )
            )

        # Collect results progressively
        result_refs = experiments
        collected_results = []

        while result_refs:
            # Wait for any task to complete
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1, timeout=10)

            if ready_refs:
                results = ray.get(ready_refs)
                collected_results.extend(results)

        assert len(collected_results) == 5
        assert all(isinstance(r, ExperimentResult) for r in collected_results)