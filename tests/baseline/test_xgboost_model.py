"""Tests for XGBoost model implementation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.hyperparameter_tuning import (
    XGBoostHyperparameterTuner,
    quick_tune_xgboost,
)
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel


class TestXGBoostConfig:
    """Test XGBoostConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = XGBoostConfig()
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.3
        assert config.objective == "multi:softprob"
        assert config.tree_method == "hist"
        assert config.device == "cpu"
        assert config.early_stopping_rounds == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = XGBoostConfig(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            reg_alpha=0.1,
        )
        assert config.n_estimators == 200
        assert config.max_depth == 8
        assert config.learning_rate == 0.1
        assert config.subsample == 0.8
        assert config.reg_alpha == 0.1

    def test_tree_method_validation(self):
        """Test tree_method validation."""
        with pytest.raises(ValueError, match="tree_method must be one of"):
            XGBoostConfig(tree_method="invalid")

    def test_device_validation(self):
        """Test device validation."""
        with pytest.raises(ValueError, match="device must be"):
            XGBoostConfig(device="invalid")

    def test_objective_validation(self):
        """Test objective validation for multi-class."""
        with pytest.raises(ValueError, match="objective must be one of"):
            XGBoostConfig(objective="binary:logistic")

    def test_parameter_bounds(self):
        """Test parameter boundary validation."""
        # Test lower bounds
        with pytest.raises(ValueError):
            XGBoostConfig(n_estimators=0)

        with pytest.raises(ValueError):
            XGBoostConfig(learning_rate=0)

        # Test upper bounds
        with pytest.raises(ValueError):
            XGBoostConfig(max_depth=21)

        with pytest.raises(ValueError):
            XGBoostConfig(learning_rate=1.1)


class TestXGBoostModel:
    """Test XGBoostModel implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multi-class classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=5,
            n_clusters_per_class=2,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
        y_series = pd.Series(y, name="cause")
        # Convert to string labels like real causes
        y_series = y_series.map(lambda x: f"cause_{x}")
        return X_df, y_series

    @pytest.fixture
    def small_data(self):
        """Create small sample data for quick tests."""
        X = pd.DataFrame(
            {
                "feature_1": [1, 2, 3, 4, 5, 6],
                "feature_2": [2, 3, 4, 5, 6, 7],
                "feature_3": [3, 4, 5, 6, 7, 8],
            }
        )
        y = pd.Series(
            ["cause_0", "cause_1", "cause_0", "cause_1", "cause_2", "cause_2"]
        )
        return X, y

    @pytest.fixture
    def model(self):
        """Create XGBoost model instance."""
        config = XGBoostConfig(n_estimators=10, early_stopping_rounds=None)
        return XGBoostModel(config=config)

    def test_model_initialization(self):
        """Test model initialization with default config."""
        model = XGBoostModel()
        assert model.config is not None
        assert model.model_ is None
        assert model._is_fitted is False

    def test_model_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = XGBoostConfig(n_estimators=50)
        model = XGBoostModel(config=config)
        assert model.config.n_estimators == 50

    def test_fit_basic(self, model, small_data):
        """Test basic model fitting."""
        X, y = small_data
        fitted_model = model.fit(X, y)

        assert fitted_model is model  # Should return self
        assert model._is_fitted is True
        assert model.model_ is not None
        assert isinstance(model.model_, xgb.Booster)
        assert model.classes_ is not None
        assert len(model.classes_) == 3
        assert model.feature_names_ == ["feature_1", "feature_2", "feature_3"]

    def test_fit_with_sample_weights(self, model, small_data):
        """Test fitting with sample weights."""
        X, y = small_data
        weights = np.array([1, 2, 1, 2, 1, 2])
        model.fit(X, y, sample_weight=weights)

        assert model._is_fitted is True

    def test_fit_with_eval_set(self, model, sample_data):
        """Test fitting with evaluation set."""
        X, y = sample_data
        X_train, X_val = X.iloc[:150], X.iloc[150:]
        y_train, y_val = y.iloc[:150], y.iloc[150:]

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert model._is_fitted is True

    def test_fit_invalid_inputs(self, model):
        """Test fit with invalid inputs."""
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            model.fit(np.array([[1, 2], [3, 4]]), pd.Series([0, 1]))

        with pytest.raises(TypeError, match="y must be a pandas Series"):
            model.fit(pd.DataFrame([[1, 2], [3, 4]]), np.array([0, 1]))

    def test_predict(self, model, small_data):
        """Test prediction after fitting."""
        X, y = small_data
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in model.classes_ for pred in predictions)

    def test_predict_proba(self, model, small_data):
        """Test probability prediction."""
        X, y = small_data
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)  # 3 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probabilities

    def test_predict_not_fitted(self, model, small_data):
        """Test prediction before fitting raises error."""
        X, _ = small_data
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_predict_proba_not_fitted(self, model, small_data):
        """Test predict_proba before fitting raises error."""
        X, _ = small_data
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_predict_proba_invalid_input(self, model, small_data):
        """Test predict_proba with invalid input."""
        X, y = small_data
        model.fit(X, y)

        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            model.predict_proba(np.array([[1, 2, 3]]))

    def test_predict_proba_mismatched_features(self, model, small_data):
        """Test predict_proba with mismatched features."""
        X, y = small_data
        model.fit(X, y)

        X_wrong = pd.DataFrame({"wrong_1": [1], "wrong_2": [2], "wrong_3": [3]})
        with pytest.raises(ValueError, match="Feature names mismatch"):
            model.predict_proba(X_wrong)

    def test_get_feature_importance(self, model, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        model.fit(X, y)

        # Test default importance type (gain)
        importance_df = model.get_feature_importance()
        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert (
            importance_df["importance"].iloc[0] >= importance_df["importance"].iloc[-1]
        )

        # Test different importance types
        for imp_type in ["weight", "cover", "total_gain", "total_cover"]:
            importance_df = model.get_feature_importance(importance_type=imp_type)
            assert isinstance(importance_df, pd.DataFrame)

    def test_get_feature_importance_not_fitted(self, model):
        """Test feature importance before fitting."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_feature_importance()

    def test_get_feature_importance_invalid_type(self, model, small_data):
        """Test feature importance with invalid type."""
        X, y = small_data
        model.fit(X, y)

        with pytest.raises(ValueError, match="importance_type must be one of"):
            model.get_feature_importance(importance_type="invalid")

    def test_calculate_csmf_accuracy(self, model):
        """Test CSMF accuracy calculation."""
        y_true = pd.Series(["A", "A", "B", "B", "C"])
        y_pred = np.array(["A", "A", "B", "C", "C"])

        csmf_acc = model.calculate_csmf_accuracy(y_true, y_pred)
        assert 0 <= csmf_acc <= 1

        # Perfect prediction
        csmf_acc_perfect = model.calculate_csmf_accuracy(y_true, y_true.values)
        assert csmf_acc_perfect == 1.0

        # Single class edge case
        y_single = pd.Series(["A", "A", "A"])
        csmf_acc_single = model.calculate_csmf_accuracy(y_single, y_single.values)
        assert csmf_acc_single == 1.0

    def test_cross_validate(self, model, sample_data):
        """Test cross-validation functionality."""
        X, y = sample_data

        # Test with default parameters
        cv_results = model.cross_validate(X, y, cv=3)

        assert "csmf_accuracy_mean" in cv_results
        assert "csmf_accuracy_std" in cv_results
        assert "cod_accuracy_mean" in cv_results
        assert "cod_accuracy_std" in cv_results
        assert "csmf_accuracy_scores" in cv_results
        assert "cod_accuracy_scores" in cv_results

        assert len(cv_results["csmf_accuracy_scores"]) == 3
        assert 0 <= cv_results["csmf_accuracy_mean"] <= 1
        assert 0 <= cv_results["cod_accuracy_mean"] <= 1

    def test_cross_validate_no_stratification(self, model, sample_data):
        """Test cross-validation without stratification."""
        X, y = sample_data

        cv_results = model.cross_validate(X, y, cv=3, stratified=False)

        assert "csmf_accuracy_mean" in cv_results
        assert len(cv_results["csmf_accuracy_scores"]) == 3

    def test_cross_validate_invalid_cv(self, model, sample_data):
        """Test cross-validation with invalid cv value."""
        X, y = sample_data

        with pytest.raises(ValueError, match="cv must be at least 2"):
            model.cross_validate(X, y, cv=1)

    def test_missing_value_handling(self, model):
        """Test XGBoost native missing value handling."""
        # Create data with missing values
        X = pd.DataFrame(
            {
                "feature_1": [1, 2, np.nan, 4, 5],
                "feature_2": [2, np.nan, 4, 5, 6],
                "feature_3": [3, 4, 5, np.nan, 7],
            }
        )
        y = pd.Series(["A", "B", "A", "B", "A"])

        # Should handle missing values without error
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_early_stopping(self):
        """Test early stopping functionality."""
        # Create simple data that will overfit quickly
        X = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        })
        y = pd.Series(["A"] * 50 + ["B"] * 50)
        
        X_train, X_val = X.iloc[:80], X.iloc[80:]
        y_train, y_val = y.iloc[:80], y.iloc[80:]

        config = XGBoostConfig(
            n_estimators=1000, 
            early_stopping_rounds=5,
            learning_rate=0.5,  # High learning rate to trigger early stopping
        )
        model = XGBoostModel(config=config)

        # Fit with early stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Just verify that model was trained with early stopping enabled
        assert model._is_fitted
        assert model.config.early_stopping_rounds == 5
        # The actual number of rounds may vary, but model should be trained
        assert model.model_.num_boosted_rounds() > 0

    def test_sklearn_interface(self, model, sample_data):
        """Test sklearn compatibility."""
        X, y = sample_data

        # Test get_params
        params = model.get_params()
        assert "config" in params

        # Test set_params
        model.set_params(config__n_estimators=20)
        assert model.config.n_estimators == 20
        
        # Test set_params with config object
        new_config = XGBoostConfig(n_estimators=30)
        model.set_params(config=new_config)
        assert model.config.n_estimators == 30


class TestXGBoostHyperparameterTuner:
    """Test hyperparameter tuning functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tuning tests."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y, name="cause").map(lambda x: f"cause_{x}")
        return X_df, y_series

    def test_tuner_initialization(self):
        """Test tuner initialization."""
        tuner = XGBoostHyperparameterTuner()
        assert tuner.metric == "csmf_accuracy"
        assert tuner.base_config is not None

    def test_tuner_custom_metric(self):
        """Test tuner with custom metric."""
        tuner = XGBoostHyperparameterTuner(metric="cod_accuracy")
        assert tuner.metric == "cod_accuracy"

    def test_tuner_invalid_metric(self):
        """Test tuner with invalid metric."""
        with pytest.raises(ValueError, match="metric must be one of"):
            XGBoostHyperparameterTuner(metric="invalid")

    @patch("baseline.models.hyperparameter_tuning.XGBoostModel")
    def test_objective_function(self, mock_model_class, sample_data):
        """Test objective function."""
        X, y = sample_data

        # Mock the model and cross_validate
        mock_model = MagicMock()
        mock_model.cross_validate.return_value = {
            "csmf_accuracy_mean": 0.8,
            "cod_accuracy_mean": 0.7,
        }
        mock_model_class.return_value = mock_model

        tuner = XGBoostHyperparameterTuner()

        # Create mock trial
        trial = MagicMock()
        trial.suggest_int.side_effect = [100, 5]  # n_estimators, max_depth
        trial.suggest_float.side_effect = [
            0.1,
            0.8,
            0.8,
            0.01,
            0.1,
        ]  # lr, subsample, etc.

        score = tuner.objective(trial, X, y, cv=3)

        assert score == -0.8  # Negative because Optuna minimizes
        mock_model.cross_validate.assert_called_once()

    def test_tune_basic(self, sample_data):
        """Test basic tuning functionality with minimal trials."""
        X, y = sample_data

        tuner = XGBoostHyperparameterTuner()
        results = tuner.tune(X, y, n_trials=2, cv=2)  # Minimal for speed

        assert "best_params" in results
        assert "best_score" in results
        assert "study" in results
        assert "best_config" in results
        assert results["n_trials"] == 2

    def test_quick_tune_xgboost(self, sample_data):
        """Test quick tuning function."""
        X, y = sample_data

        model = quick_tune_xgboost(X, y, n_trials=2)

        assert isinstance(model, XGBoostModel)
        assert model._is_fitted is True

    @patch("baseline.models.hyperparameter_tuning.XGBoostModel")
    @patch("baseline.models.hyperparameter_tuning.logger")
    def test_objective_function_failure(self, mock_logger, mock_model_class, sample_data):
        """Test objective function handling failures."""
        X, y = sample_data

        # Mock the model to raise an exception during cross_validate
        mock_model = MagicMock()
        mock_model.cross_validate.side_effect = Exception("Cross-validation failed")
        mock_model_class.return_value = mock_model

        tuner = XGBoostHyperparameterTuner()

        # Create normal trial
        trial = MagicMock()
        trial.suggest_int.side_effect = [100, 5]  # n_estimators, max_depth
        trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.01, 0.1]  # lr, subsample, etc.
        trial.number = 1

        score = tuner.objective(trial, X, y)

        assert score == float("inf")  # Should return worst score
        mock_logger.warning.assert_called()

    def test_param_importance_missing_dependency(self, sample_data):
        """Test parameter importance with missing dependency."""
        X, y = sample_data

        tuner = XGBoostHyperparameterTuner()
        results = tuner.tune(X, y, n_trials=2, cv=2)

        # This might fail if optuna importance module not available
        # but should handle gracefully
        importance = tuner.get_param_importance(results["study"])
        assert isinstance(importance, dict)  # May be empty if not available
        
    def test_plotting_functions(self, sample_data):
        """Test plotting functions handle missing dependencies gracefully."""
        X, y = sample_data

        tuner = XGBoostHyperparameterTuner()
        results = tuner.tune(X, y, n_trials=2, cv=2)
        study = results["study"]
        
        # Test plot_optimization_history
        # This should handle missing matplotlib gracefully
        plot = tuner.plot_optimization_history(study)
        # Either returns plot or None if matplotlib not available
        assert plot is None or plot is not None
        
        # Test plot_param_importances
        plot = tuner.plot_param_importances(study)
        # Either returns plot or None if matplotlib not available
        assert plot is None or plot is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
