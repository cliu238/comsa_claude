"""Unit tests for hyperparameter tuning module."""

import pandas as pd
import pytest
import ray
from ray import tune
from sklearn.datasets import make_classification

from baseline.models.logistic_regression_model import LogisticRegressionModel
from baseline.models.random_forest_model import RandomForestModel
from baseline.models.xgboost_model import XGBoostModel
from model_comparison.hyperparameter_tuning.ray_tuner import RayTuner, quick_tune_model
from model_comparison.hyperparameter_tuning.search_spaces import (
    filter_params_for_model,
    get_logistic_regression_search_space,
    get_random_forest_search_space,
    get_search_space_for_model,
    get_xgboost_search_space,
)


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for the test module."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    # Ray shutdown happens automatically after tests


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    
    # Convert to string labels to match VA data format
    y_series = y_series.astype(str)
    
    return X_df, y_series


class TestSearchSpaces:
    """Test search space definitions."""
    
    def test_xgboost_search_space(self):
        """Test XGBoost search space structure."""
        space = get_xgboost_search_space()
        
        # Check required parameters
        assert "config__max_depth" in space
        assert "config__learning_rate" in space
        assert "config__n_estimators" in space
        assert "config__subsample" in space
        assert "config__colsample_bytree" in space
        assert "config__reg_alpha" in space
        assert "config__reg_lambda" in space
        
        # Check parameter types
        assert isinstance(space["config__max_depth"], tune.search.sample.Categorical)
        assert isinstance(space["config__learning_rate"], tune.search.sample.Float)
        assert isinstance(space["config__n_estimators"], tune.search.sample.Categorical)
    
    def test_random_forest_search_space(self):
        """Test Random Forest search space structure."""
        space = get_random_forest_search_space()
        
        # Check required parameters
        assert "config__n_estimators" in space
        assert "config__max_depth" in space
        assert "config__min_samples_split" in space
        assert "config__min_samples_leaf" in space
        assert "config__max_features" in space
        assert "config__bootstrap" in space
        
    def test_logistic_regression_search_space(self):
        """Test Logistic Regression search space structure."""
        space = get_logistic_regression_search_space()
        
        # Check required parameters
        assert "config__C" in space
        assert "config__penalty" in space
        assert "config__solver" in space
        assert "config__l1_ratio" in space
        assert "config__max_iter" in space
        
        # Check solver is fixed to saga
        assert space["config__solver"] == "saga"
    
    def test_parameter_prefix(self):
        """Test that all parameters have correct prefix."""
        for get_space in [
            get_xgboost_search_space,
            get_random_forest_search_space,
            get_logistic_regression_search_space,
        ]:
            space = get_space()
            for key in space.keys():
                assert key.startswith("config__"), f"Parameter {key} missing config__ prefix"
    
    def test_filter_params_for_model(self):
        """Test parameter filtering for conditional parameters."""
        # Test logistic regression with elasticnet
        params = {
            "config__C": 1.0,
            "config__penalty": "elasticnet",
            "config__l1_ratio": 0.5,
        }
        filtered = filter_params_for_model(params, "logistic_regression")
        assert "config__l1_ratio" in filtered
        
        # Test logistic regression without elasticnet
        params = {
            "config__C": 1.0,
            "config__penalty": "l2",
            "config__l1_ratio": 0.5,
        }
        filtered = filter_params_for_model(params, "logistic_regression")
        assert "config__l1_ratio" not in filtered
    
    def test_get_search_space_for_model(self):
        """Test getting search space by model name."""
        # Valid models
        for model_name in ["xgboost", "random_forest", "logistic_regression"]:
            space = get_search_space_for_model(model_name)
            assert isinstance(space, dict)
            assert len(space) > 0
        
        # Invalid model
        with pytest.raises(ValueError):
            get_search_space_for_model("invalid_model")


class TestRayTuner:
    """Test RayTuner class."""
    
    def test_tuner_initialization(self):
        """Test RayTuner initialization."""
        tuner = RayTuner(
            n_trials=10,
            n_cpus_per_trial=1.0,
            metric="csmf_accuracy",
        )
        
        assert tuner.n_trials == 10
        assert tuner.n_cpus_per_trial == 1.0
        assert tuner.metric == "csmf_accuracy"
        assert tuner.mode == "max"
        assert tuner.scheduler is not None
    
    def test_tuner_with_invalid_params(self):
        """Test RayTuner with various parameter configurations."""
        # Valid initialization
        tuner = RayTuner(
            search_algorithm="random",
            metric="cod_accuracy",
            mode="min",
        )
        assert tuner.search_algorithm == "random"
        assert tuner.metric == "cod_accuracy"
        assert tuner.mode == "min"
    
    @pytest.mark.parametrize("model_name", ["xgboost", "random_forest", "logistic_regression"])
    def test_tune_model_basic(self, ray_context, sample_data, model_name):
        """Test basic tuning functionality for each model."""
        X, y = sample_data
        
        # Get search space
        search_space = get_search_space_for_model(model_name)
        
        # Limit search space for faster testing
        limited_space = {}
        for key, value in search_space.items():
            if isinstance(value, tune.search.sample.Categorical):
                # Take only first option
                limited_space[key] = value.categories[0]
            elif isinstance(value, tune.search.sample.Float):
                # Use midpoint
                limited_space[key] = (value.lower + value.upper) / 2
            else:
                limited_space[key] = value
        
        # Create tuner with minimal trials
        tuner = RayTuner(
            n_trials=2,
            n_cpus_per_trial=1.0,
            max_concurrent_trials=1,
        )
        
        # Run tuning
        results = tuner.tune_model(
            model_name=model_name,
            search_space=limited_space,
            train_data=(X, y),
            cv_folds=2,
        )
        
        # Check results
        assert "best_params" in results
        assert "best_score" in results
        assert "metrics" in results
        assert results["model_name"] == model_name
        assert results["n_trials_completed"] > 0
        assert 0 <= results["best_score"] <= 1
    
    def test_tune_model_with_validation_data(self, ray_context, sample_data):
        """Test tuning with separate validation data."""
        X, y = sample_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Simple search space
        search_space = {"config__max_depth": 5}
        
        tuner = RayTuner(n_trials=1)
        results = tuner.tune_model(
            model_name="xgboost",
            search_space=search_space,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
        )
        
        assert results["best_score"] >= 0
        assert results["n_trials_completed"] == 1
    
    def test_save_results(self, tmp_path, ray_context, sample_data):
        """Test saving tuning results."""
        X, y = sample_data
        
        # Run minimal tuning
        tuner = RayTuner(n_trials=1)
        results = tuner.tune_model(
            model_name="xgboost",
            search_space={"config__max_depth": 5},
            train_data=(X, y),
            cv_folds=2,
        )
        
        # Save results
        output_path = tmp_path / "tuning_results"
        tuner.save_results(results, output_path)
        
        # Check files exist
        assert (output_path.with_suffix(".json")).exists()
        
        # Load and verify JSON
        import json
        with open(output_path.with_suffix(".json"), "r") as f:
            saved_data = json.load(f)
        
        assert saved_data["model_name"] == "xgboost"
        assert "best_params" in saved_data
        assert "best_score" in saved_data


class TestQuickTuneFunction:
    """Test the quick_tune_model convenience function."""
    
    def test_quick_tune_xgboost(self, ray_context, sample_data):
        """Test quick tuning for XGBoost."""
        X, y = sample_data
        
        results = quick_tune_model(
            model_name="xgboost",
            X=X,
            y=y,
            n_trials=2,
            metric="csmf_accuracy",
        )
        
        # Check results
        assert "best_params" in results
        assert "best_score" in results
        assert "trained_model" in results
        
        # Check model is fitted
        model = results["trained_model"]
        assert isinstance(model, XGBoostModel)
        assert model._is_fitted
        
        # Test predictions
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_quick_tune_with_different_metrics(self, ray_context, sample_data):
        """Test quick tuning with different optimization metrics."""
        X, y = sample_data
        
        # Tune for CSMF accuracy
        results_csmf = quick_tune_model(
            model_name="random_forest",
            X=X,
            y=y,
            n_trials=2,
            metric="csmf_accuracy",
        )
        
        # Tune for COD accuracy
        results_cod = quick_tune_model(
            model_name="random_forest",
            X=X,
            y=y,
            n_trials=2,
            metric="cod_accuracy",
        )
        
        # Both should return valid results
        assert results_csmf["best_score"] >= 0
        assert results_cod["best_score"] >= 0


class TestIntegrationWithModels:
    """Test integration with actual model classes."""
    
    def test_set_params_integration(self):
        """Test that tuned parameters can be set on models."""
        # XGBoost
        xgb_params = {
            "config__max_depth": 7,
            "config__learning_rate": 0.1,
            "config__n_estimators": 200,
        }
        xgb_model = XGBoostModel()
        xgb_model.set_params(**xgb_params)
        assert xgb_model.config.max_depth == 7
        assert xgb_model.config.learning_rate == 0.1
        assert xgb_model.config.n_estimators == 200
        
        # Random Forest
        rf_params = {
            "config__n_estimators": 300,
            "config__max_depth": 20,
            "config__min_samples_split": 5,
        }
        rf_model = RandomForestModel()
        rf_model.set_params(**rf_params)
        assert rf_model.config.n_estimators == 300
        assert rf_model.config.max_depth == 20
        assert rf_model.config.min_samples_split == 5
        
        # Logistic Regression
        lr_params = {
            "config__C": 10.0,
            "config__penalty": "l2",
            "config__max_iter": 2000,
        }
        lr_model = LogisticRegressionModel()
        lr_model.set_params(**lr_params)
        assert lr_model.config.C == 10.0
        assert lr_model.config.penalty == "l2"
        assert lr_model.config.max_iter == 2000
    
    def test_csmf_accuracy_consistency(self, sample_data):
        """Test that CSMF accuracy calculation is consistent across models."""
        X, y = sample_data
        
        # Train each model with default params
        models = [
            XGBoostModel(),
            RandomForestModel(),
            LogisticRegressionModel(),
        ]
        
        for model in models:
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate CSMF accuracy
            csmf_acc = model.calculate_csmf_accuracy(y, y_pred)
            
            # Should be between 0 and 1
            assert 0 <= csmf_acc <= 1
            
            # For perfect predictions on training data, should be high
            if (y == y_pred).all():
                assert csmf_acc == 1.0


@pytest.fixture(autouse=True)
def cleanup_ray_results(tmp_path, monkeypatch):
    """Clean up Ray results directory after tests."""
    # Set Ray results to temp directory
    monkeypatch.chdir(tmp_path)
    yield
    # Cleanup happens automatically with tmp_path