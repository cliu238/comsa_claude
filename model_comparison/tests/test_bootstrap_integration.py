"""Integration test for bootstrap confidence interval calculation."""

import pandas as pd
import numpy as np

from model_comparison.metrics.comparison_metrics import calculate_metrics
from model_comparison.orchestration.config import ExperimentResult


def test_metrics_integration_with_experiment_result():
    """Test that metrics integrate properly with ExperimentResult."""
    # Create test data
    y_true = pd.Series(["cause_1"] * 30 + ["cause_2"] * 30 + ["cause_3"] * 40)
    y_pred = np.array(["cause_1"] * 25 + ["cause_2"] * 35 + ["cause_3"] * 40)
    
    # Calculate metrics with bootstrap
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=100)
    
    # Create ExperimentResult - this should work without errors
    result = ExperimentResult(
        experiment_id="test_001",
        model_name="test_model",
        experiment_type="bootstrap_test",
        train_site="site_A",
        test_site="site_B",
        training_size=1.0,
        csmf_accuracy=metrics["csmf_accuracy"],
        cod_accuracy=metrics["cod_accuracy"],
        csmf_accuracy_ci=metrics.get("csmf_accuracy_ci"),
        cod_accuracy_ci=metrics.get("cod_accuracy_ci"),
        n_train=100,
        n_test=100,
        execution_time_seconds=1.5,
    )
    
    # Verify CI is stored correctly
    assert isinstance(result.csmf_accuracy_ci, list)
    assert len(result.csmf_accuracy_ci) == 2
    assert isinstance(result.cod_accuracy_ci, list)
    assert len(result.cod_accuracy_ci) == 2
    
    # Verify values are sensible
    assert 0 <= result.csmf_accuracy_ci[0] <= result.csmf_accuracy_ci[1] <= 1
    assert 0 <= result.cod_accuracy_ci[0] <= result.cod_accuracy_ci[1] <= 1
    assert result.csmf_accuracy_ci[0] <= result.csmf_accuracy <= result.csmf_accuracy_ci[1]
    assert result.cod_accuracy_ci[0] <= result.cod_accuracy <= result.cod_accuracy_ci[1]


def test_metrics_integration_without_bootstrap():
    """Test ExperimentResult when no bootstrap CI is calculated."""
    # Create test data
    y_true = pd.Series(["cause_1"] * 50 + ["cause_2"] * 50)
    y_pred = np.array(["cause_1"] * 48 + ["cause_2"] * 52)
    
    # Calculate metrics without bootstrap
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=0)
    
    # Create ExperimentResult
    result = ExperimentResult(
        experiment_id="test_002",
        model_name="test_model",
        experiment_type="no_bootstrap_test",
        train_site="site_A",
        test_site="site_B",
        training_size=1.0,
        csmf_accuracy=metrics["csmf_accuracy"],
        cod_accuracy=metrics["cod_accuracy"],
        csmf_accuracy_ci=metrics.get("csmf_accuracy_ci"),
        cod_accuracy_ci=metrics.get("cod_accuracy_ci"),
        n_train=100,
        n_test=100,
        execution_time_seconds=0.5,
    )
    
    # Verify CI is None
    assert result.csmf_accuracy_ci is None
    assert result.cod_accuracy_ci is None


def test_ray_task_compatibility():
    """Test that metrics output is compatible with ray_tasks.py expectations."""
    # Create test data
    y_true = pd.Series(["A", "B", "C"] * 20)
    y_pred = np.array(["A", "B", "C"] * 19 + ["A", "B", "A"])
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=50)
    
    # Simulate ray_tasks.py check (lines 141-142)
    assert isinstance(metrics.get("csmf_accuracy_ci"), list)
    assert isinstance(metrics.get("cod_accuracy_ci"), list)
    
    # The CI should be usable in ExperimentResult
    csmf_ci = metrics.get("csmf_accuracy_ci") if isinstance(metrics.get("csmf_accuracy_ci"), list) else None
    cod_ci = metrics.get("cod_accuracy_ci") if isinstance(metrics.get("cod_accuracy_ci"), list) else None
    
    assert csmf_ci is not None
    assert cod_ci is not None
    assert len(csmf_ci) == 2
    assert len(cod_ci) == 2