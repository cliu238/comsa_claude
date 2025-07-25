"""Tests for hyperparameter tuning functionality."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray

from model_comparison.hyperparameter_tuning import (
    ModelSearchSpace,
    SearchSpace,
    get_search_space,
    get_tuner,
)
from model_comparison.hyperparameter_tuning.utils import (
    clear_cache,
    load_cached_params,
    save_cached_params,
)


class TestSearchSpaces:
    """Test search space generation and validation."""
    
    def test_search_space_generation(self):
        """Test that search spaces are generated correctly for each model."""
        for model_name in ["xgboost", "random_forest", "logistic_regression"]:
            space = get_search_space(model_name)
            assert isinstance(space, ModelSearchSpace)
            assert space.model_name == model_name
            assert len(space.parameters) > 0
            
            # Check that all parameters have valid types
            for param_name, param_space in space.parameters.items():
                assert isinstance(param_space, SearchSpace)
                assert param_space.type in ["int", "float", "categorical"]
                assert param_space.values is not None
    
    def test_invalid_model_name(self):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_search_space("invalid_model")
    
    def test_xgboost_search_space(self):
        """Test XGBoost search space specifics."""
        space = get_search_space("xgboost")
        
        # Check key parameters exist
        assert "n_estimators" in space.parameters
        assert "max_depth" in space.parameters
        assert "learning_rate" in space.parameters
        
        # Check log scale for appropriate parameters
        assert space.parameters["learning_rate"].log_scale is True
        assert space.parameters["reg_alpha"].log_scale is True
    
    def test_logistic_regression_search_space(self):
        """Test Logistic Regression search space specifics."""
        space = get_search_space("logistic_regression")
        
        # Check that solver is fixed to saga
        assert space.parameters["solver"].type == "categorical"
        assert space.parameters["solver"].values == ["saga"]
        
        # Check penalty options
        assert "l1" in space.parameters["penalty"].values
        assert "l2" in space.parameters["penalty"].values
        assert "elasticnet" in space.parameters["penalty"].values


class TestTunerFactory:
    """Test tuner factory and base functionality."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.randint(0, 5, 100))
        return X, y
    
    def test_tuner_factory(self, dummy_data):
        """Test tuner creation for different methods."""
        X, y = dummy_data
        
        for method in ["grid", "optuna"]:
            tuner = get_tuner(
                method=method,
                model_name="xgboost",
                X=X,
                y=y,
                n_trials=5,
            )
            assert tuner is not None
            assert tuner.model_name == "xgboost"
    
    def test_invalid_tuner_method(self, dummy_data):
        """Test that invalid tuner methods raise ValueError."""
        X, y = dummy_data
        
        with pytest.raises(ValueError, match="Unknown tuning method"):
            get_tuner(
                method="invalid_method",
                model_name="xgboost",
                X=X,
                y=y,
            )


class TestOptunaTuner:
    """Test Optuna tuner implementation."""
    
    @pytest.fixture
    def small_data(self):
        """Create small dataset for faster testing."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=5,
            random_state=42,
        )
        return pd.DataFrame(X), pd.Series(y)
    
    def test_optuna_tuner_basic(self, small_data):
        """Test Optuna tuner runs successfully."""
        X, y = small_data
        
        tuner = get_tuner(
            method="optuna",
            model_name="xgboost",
            X=X,
            y=y,
            n_trials=10,
            cv_folds=3,
            metric="accuracy",  # Use simple accuracy for speed
        )
        
        result = tuner.tune()
        assert result.best_params is not None
        assert result.best_score > 0
        assert result.n_trials_completed <= 10
        assert result.duration_seconds > 0
    
    def test_optuna_tuner_all_models(self, small_data):
        """Test Optuna tuner works for all supported models."""
        X, y = small_data
        
        for model_name in ["xgboost", "random_forest", "logistic_regression"]:
            tuner = get_tuner(
                method="optuna",
                model_name=model_name,
                X=X,
                y=y,
                n_trials=5,
                cv_folds=2,
                metric="accuracy",
            )
            
            result = tuner.tune()
            assert result.best_params is not None
            assert result.best_score > 0
            
            # Check model-specific parameters
            if model_name == "logistic_regression":
                # Check penalty-specific logic
                if result.best_params.get("penalty") == "elasticnet":
                    assert "l1_ratio" in result.best_params
                else:
                    assert "l1_ratio" not in result.best_params


class TestCaching:
    """Test parameter caching functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        original_cache_dir = Path("cache/tuned_params")
        
        # Temporarily override cache directory
        import model_comparison.hyperparameter_tuning.utils as utils
        utils.get_cache_dir = lambda: Path(temp_dir)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
        utils.get_cache_dir = lambda: original_cache_dir
    
    def test_save_and_load_params(self, temp_cache_dir):
        """Test saving and loading cached parameters."""
        test_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
        }
        cache_key = "test_experiment_xgboost"
        
        # Save parameters
        save_cached_params(cache_key, test_params)
        
        # Load parameters
        loaded = load_cached_params(cache_key)
        assert loaded == test_params
    
    def test_load_nonexistent_params(self, temp_cache_dir):
        """Test loading parameters that don't exist."""
        loaded = load_cached_params("nonexistent_key")
        assert loaded is None
    
    def test_clear_cache(self, temp_cache_dir):
        """Test clearing cached parameters."""
        # Save multiple parameter sets
        save_cached_params("exp1_xgboost", {"n_estimators": 100})
        save_cached_params("exp1_rf", {"n_estimators": 200})
        save_cached_params("exp2_xgboost", {"n_estimators": 300})
        
        # Clear all exp1 caches
        cleared = clear_cache("exp1")
        assert cleared == 2
        
        # Check that exp2 still exists
        assert load_cached_params("exp2_xgboost") is not None
        assert load_cached_params("exp1_xgboost") is None


class TestRayIntegration:
    """Test integration with Ray for distributed tuning."""
    
    @pytest.fixture
    def ray_init(self):
        """Initialize Ray for testing."""
        if not ray.is_initialized():
            ray.init(local_mode=True)
        yield
        ray.shutdown()
    
    def test_ray_tuning_task(self, ray_init):
        """Test hyperparameter tuning via Ray task."""
        from model_comparison.orchestration.ray_tasks import tune_hyperparameters
        
        # Create test data
        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.randint(0, 5, 100))
        
        tuning_config = {
            "method": "optuna",
            "n_trials": 5,
            "cv_folds": 3,
            "metric": "accuracy",
        }
        
        # Run tuning task
        future = tune_hyperparameters.remote(
            "xgboost",
            (X, y),
            tuning_config,
            "test_exp_001",
        )
        
        result = ray.get(future)
        assert "best_params" in result
        assert "best_score" in result
        assert "tuning_time_seconds" in result
        assert "from_cache" in result
    
    def test_ray_tuning_with_cache(self, ray_init, temp_cache_dir):
        """Test that Ray task uses cached parameters."""
        from model_comparison.orchestration.ray_tasks import tune_hyperparameters
        
        X = pd.DataFrame(np.random.rand(50, 5))
        y = pd.Series(np.random.randint(0, 3, 50))
        
        tuning_config = {"method": "optuna", "n_trials": 3}
        experiment_id = "cache_test_exp"
        
        # First run - should tune
        future1 = tune_hyperparameters.remote(
            "xgboost", (X, y), tuning_config, experiment_id
        )
        result1 = ray.get(future1)
        assert result1["from_cache"] is False
        assert result1["tuning_time_seconds"] > 0
        
        # Second run - should use cache
        future2 = tune_hyperparameters.remote(
            "xgboost", (X, y), tuning_config, experiment_id
        )
        result2 = ray.get(future2)
        assert result2["from_cache"] is True
        assert result2["tuning_time_seconds"] == 0.0
        assert result2["best_params"] == result1["best_params"]


def test_grid_tuner_basic():
    """Test grid search tuner basic functionality."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=3, random_state=42
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    tuner = get_tuner(
        method="grid",
        model_name="xgboost",
        X=X,
        y=y,
        grid_size="small",
        cv_folds=2,
    )
    
    result = tuner.tune()
    assert result.best_params is not None
    assert result.best_score > 0
    assert result.n_trials_completed > 0
    assert result.all_trials is not None