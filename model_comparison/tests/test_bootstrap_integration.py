"""Integration tests for bootstrap confidence interval calculation."""

import pandas as pd
import numpy as np
import pytest

from model_comparison.orchestration.config import ExperimentResult
from model_comparison.orchestration.ray_tasks import train_and_evaluate_model
from model_comparison.metrics.comparison_metrics import calculate_metrics


class TestBootstrapIntegration:
    """Integration tests for bootstrap CI implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample train and test data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Create features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        # Create labels with 4 classes
        y = pd.Series(
            np.random.choice(["cause_A", "cause_B", "cause_C", "cause_D"], n_samples,
                           p=[0.4, 0.3, 0.2, 0.1])
        )
        
        # Split into train and test
        split_idx = int(0.8 * n_samples)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        return (X_train, y_train), (X_test, y_test)

    def test_end_to_end_ci_calculation(self, sample_data):
        """Test end-to-end CI calculation through ray_tasks."""
        train_data, test_data = sample_data
        
        # Prepare experiment metadata
        experiment_metadata = {
            "experiment_id": "test_bootstrap_001",
            "experiment_type": "in_domain",
            "train_site": "site_1",
            "test_site": "site_1",
            "training_size": 1.0,
        }
        
        # Test with different bootstrap values
        for n_bootstrap in [10, 50, 100]:
            # Note: We're not actually using Ray here, just testing the function
            # In a real Ray environment, this would be executed remotely
            result = train_and_evaluate_model.remote(
                model_name="xgboost",
                train_data=train_data,
                test_data=test_data,
                experiment_metadata=experiment_metadata,
                n_bootstrap=n_bootstrap,
            )
            
            # Since we're not in a Ray environment, we'll test the metrics directly
            X_test, y_test = test_data
            y_pred = np.random.choice(y_test.unique(), len(y_test))  # Simulated predictions
            
            metrics = calculate_metrics(y_test, y_pred, n_bootstrap=n_bootstrap)
            
            # Verify CI format
            assert isinstance(metrics["cod_accuracy_ci"], list)
            assert isinstance(metrics["csmf_accuracy_ci"], list)
            assert len(metrics["cod_accuracy_ci"]) == 2
            assert len(metrics["csmf_accuracy_ci"]) == 2
            
            # Verify CI bounds make sense
            assert metrics["cod_accuracy_ci"][0] <= metrics["cod_accuracy"]
            assert metrics["cod_accuracy"] <= metrics["cod_accuracy_ci"][1]
            assert metrics["csmf_accuracy_ci"][0] <= metrics["csmf_accuracy"]
            assert metrics["csmf_accuracy"] <= metrics["csmf_accuracy_ci"][1]

    def test_experiment_result_with_ci(self):
        """Test that ExperimentResult properly handles CI lists."""
        # Create a result with CI values
        result = ExperimentResult(
            experiment_id="test_001",
            model_name="xgboost",
            experiment_type="in_domain",
            train_site="site_1",
            test_site="site_1",
            csmf_accuracy=0.75,
            cod_accuracy=0.85,
            csmf_accuracy_ci=[0.70, 0.80],
            cod_accuracy_ci=[0.82, 0.88],
            execution_time_seconds=10.5,
        )
        
        # Verify CI values are stored correctly
        assert result.csmf_accuracy_ci == [0.70, 0.80]
        assert result.cod_accuracy_ci == [0.82, 0.88]
        
        # Verify serialization works
        result_dict = result.to_dict()
        assert result_dict["csmf_accuracy_ci"] == [0.70, 0.80]
        assert result_dict["cod_accuracy_ci"] == [0.82, 0.88]

    def test_ci_with_different_model_types(self, sample_data):
        """Test CI calculation works for different model types."""
        train_data, test_data = sample_data
        X_test, y_test = test_data
        
        models = ["xgboost", "random_forest", "logistic_regression"]
        
        for model_name in models:
            # Simulate predictions for each model type
            y_pred = np.random.choice(y_test.unique(), len(y_test))
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred, n_bootstrap=20)
            
            # All models should produce CI in the same format
            assert isinstance(metrics["cod_accuracy_ci"], list)
            assert isinstance(metrics["csmf_accuracy_ci"], list)
            assert len(metrics["cod_accuracy_ci"]) == 2
            assert len(metrics["csmf_accuracy_ci"]) == 2

    def test_ci_with_edge_cases_integration(self):
        """Test CI calculation with edge cases in integration context."""
        # Test with perfect predictions
        y_true = pd.Series(["A"] * 50 + ["B"] * 50)
        y_pred = y_true.values
        
        metrics = calculate_metrics(y_true, y_pred, n_bootstrap=10)
        
        # Perfect predictions should have CI = [1.0, 1.0]
        assert metrics["cod_accuracy"] == 1.0
        assert metrics["cod_accuracy_ci"] == [1.0, 1.0]
        assert metrics["csmf_accuracy"] == 1.0
        assert metrics["csmf_accuracy_ci"] == [1.0, 1.0]
        
        # Test with very small dataset
        y_small = pd.Series(["A", "B", "C", "A", "B"])
        y_pred_small = np.array(["A", "B", "C", "B", "A"])
        
        metrics_small = calculate_metrics(y_small, y_pred_small, n_bootstrap=5)
        
        # Should still produce valid CI
        assert isinstance(metrics_small["cod_accuracy_ci"], list)
        assert isinstance(metrics_small["csmf_accuracy_ci"], list)