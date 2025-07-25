"""Tests for comparison metrics module."""

import numpy as np
import pandas as pd
import pytest

from model_comparison.metrics.comparison_metrics import (
    bootstrap_metric,
    calculate_csmf_accuracy,
    calculate_metrics,
    calculate_per_cause_accuracy,
)


class TestComparisonMetrics:
    """Test metric calculation functions."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 5

        y_true = pd.Series(
            np.random.choice([f"cause_{i}" for i in range(n_classes)], n_samples)
        )

        # Create predictions with some accuracy
        y_pred = y_true.copy()
        # Introduce some errors
        error_indices = np.random.choice(n_samples, size=20, replace=False)
        for idx in error_indices:
            y_pred.iloc[idx] = np.random.choice(
                [f"cause_{i}" for i in range(n_classes)]
            )

        return y_true, y_pred.values

    def test_calculate_csmf_accuracy(self, sample_predictions):
        """Test CSMF accuracy calculation."""
        y_true, y_pred = sample_predictions

        # Test basic calculation
        csmf_acc = calculate_csmf_accuracy(y_true, y_pred)
        assert 0 <= csmf_acc <= 1

        # Test perfect prediction
        perfect_csmf = calculate_csmf_accuracy(y_true, y_true.values)
        assert perfect_csmf == 1.0

        # Test worst case (all predictions wrong)
        wrong_pred = pd.Series(["wrong_cause"] * len(y_true))
        worst_csmf = calculate_csmf_accuracy(y_true, wrong_pred.values)
        assert worst_csmf >= 0

        # Test single cause edge case
        single_cause = pd.Series(["cause_A"] * 10)
        single_csmf = calculate_csmf_accuracy(single_cause, single_cause.values)
        assert single_csmf == 1.0

    def test_bootstrap_metric(self, sample_predictions):
        """Test bootstrap confidence interval calculation."""
        y_true, y_pred = sample_predictions

        # Test with accuracy metric
        from sklearn.metrics import accuracy_score

        lower, upper = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap=50)

        # Check bounds
        assert 0 <= lower <= upper <= 1

        # Point estimate should be within bounds
        point_estimate = accuracy_score(y_true, y_pred)
        assert lower <= point_estimate <= upper

    def test_calculate_metrics(self, sample_predictions):
        """Test comprehensive metric calculation."""
        y_true, y_pred = sample_predictions

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, n_bootstrap=20)

        # Check all expected metrics are present
        expected_metrics = [
            "cod_accuracy",
            "cod_accuracy_ci_lower",
            "cod_accuracy_ci_upper",
            "csmf_accuracy",
            "csmf_accuracy_ci_lower",
            "csmf_accuracy_ci_upper",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1

        # Check confidence intervals make sense
        assert metrics["cod_accuracy_ci_lower"] <= metrics["cod_accuracy"]
        assert metrics["cod_accuracy"] <= metrics["cod_accuracy_ci_upper"]
        assert metrics["csmf_accuracy_ci_lower"] <= metrics["csmf_accuracy"]
        assert metrics["csmf_accuracy"] <= metrics["csmf_accuracy_ci_upper"]

    def test_calculate_per_cause_accuracy(self, sample_predictions):
        """Test per-cause accuracy calculation."""
        y_true, y_pred = sample_predictions

        # Calculate per-cause accuracy
        cause_acc = calculate_per_cause_accuracy(y_true, y_pred)

        # Check structure
        assert isinstance(cause_acc, dict)
        assert len(cause_acc) == y_true.nunique()

        # Check values
        for cause, acc in cause_acc.items():
            assert 0 <= acc <= 1
            assert isinstance(acc, float)

    def test_edge_cases(self):
        """Test edge cases in metric calculations."""
        # Empty predictions
        empty_true = pd.Series([], dtype=str)
        empty_pred = np.array([], dtype=str)

        # Should handle empty data gracefully
        with pytest.raises(Exception):
            calculate_csmf_accuracy(empty_true, empty_pred)

        # All same class
        same_class = pd.Series(["A"] * 100)
        csmf = calculate_csmf_accuracy(same_class, same_class.values)
        assert csmf == 1.0

        # Mismatched lengths should fail
        y_true = pd.Series(["A", "B", "C"])
        y_pred = np.array(["A", "B"])

        with pytest.raises(Exception):
            calculate_metrics(y_true, y_pred)

    def test_csmf_accuracy_properties(self):
        """Test mathematical properties of CSMF accuracy."""
        # Create controlled distribution
        true_dist = pd.Series(["A"] * 40 + ["B"] * 30 + ["C"] * 20 + ["D"] * 10)

        # Perfect prediction
        perfect = calculate_csmf_accuracy(true_dist, true_dist.values)
        assert perfect == 1.0

        # Slightly off prediction
        pred_slight = pd.Series(["A"] * 38 + ["B"] * 32 + ["C"] * 20 + ["D"] * 10)
        slight_acc = calculate_csmf_accuracy(true_dist, pred_slight.values)
        assert 0 < slight_acc < 1.0

        # Very wrong prediction (swap A and D)
        pred_wrong = pd.Series(["D"] * 40 + ["B"] * 30 + ["C"] * 20 + ["A"] * 10)
        wrong_acc = calculate_csmf_accuracy(true_dist, pred_wrong.values)
        assert wrong_acc < slight_acc

    def test_calculate_metrics_with_bootstrap_ci_list_format(self, sample_predictions):
        """Test that calculate_metrics returns CI in list format."""
        y_true, y_pred = sample_predictions
        
        # Calculate with bootstrap
        metrics = calculate_metrics(y_true, y_pred, n_bootstrap=50)
        
        # Check CI is in list format
        assert isinstance(metrics["cod_accuracy_ci"], list)
        assert len(metrics["cod_accuracy_ci"]) == 2
        assert isinstance(metrics["csmf_accuracy_ci"], list)
        assert len(metrics["csmf_accuracy_ci"]) == 2
        
        # Check values are sensible
        assert metrics["cod_accuracy_ci"][0] <= metrics["cod_accuracy"]
        assert metrics["cod_accuracy"] <= metrics["cod_accuracy_ci"][1]
        
        # Check backward compatibility
        assert metrics["cod_accuracy_ci_lower"] == metrics["cod_accuracy_ci"][0]
        assert metrics["cod_accuracy_ci_upper"] == metrics["cod_accuracy_ci"][1]

    def test_calculate_metrics_without_bootstrap(self):
        """Test that CI is None when n_bootstrap=0."""
        y_true = pd.Series(["A", "B", "A", "B"])
        y_pred = np.array(["A", "B", "A", "B"])
        
        metrics = calculate_metrics(y_true, y_pred, n_bootstrap=0)
        
        assert metrics["cod_accuracy_ci"] is None
        assert metrics["csmf_accuracy_ci"] is None
        # No old format fields when n_bootstrap=0
        assert "cod_accuracy_ci_lower" not in metrics
        assert "cod_accuracy_ci_upper" not in metrics

    def test_bootstrap_with_small_sample(self):
        """Test bootstrap handles small samples gracefully."""
        # Only 3 samples - bootstrap should still work
        y_true = pd.Series(["A", "B", "A"])
        y_pred = np.array(["A", "B", "B"])
        
        metrics = calculate_metrics(y_true, y_pred, n_bootstrap=20)
        
        assert metrics["cod_accuracy_ci"] is not None
        assert 0 <= metrics["cod_accuracy_ci"][0] <= metrics["cod_accuracy_ci"][1] <= 1

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap results are reproducible."""
        y_true = pd.Series(["A"] * 50 + ["B"] * 50)
        y_pred = np.array(["A"] * 45 + ["B"] * 55)
        
        # Run twice - should get same results
        metrics1 = calculate_metrics(y_true, y_pred, n_bootstrap=100)
        metrics2 = calculate_metrics(y_true, y_pred, n_bootstrap=100)
        
        assert metrics1["cod_accuracy_ci"] == metrics2["cod_accuracy_ci"]
        assert metrics1["csmf_accuracy_ci"] == metrics2["csmf_accuracy_ci"]
