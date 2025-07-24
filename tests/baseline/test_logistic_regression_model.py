"""Tests for Logistic Regression model implementation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.logistic_regression_config import LogisticRegressionConfig
from baseline.models.logistic_regression_model import LogisticRegressionModel


class TestLogisticRegressionConfig:
    """Test LogisticRegressionConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogisticRegressionConfig()
        assert config.penalty == "l2"
        assert config.C == 1.0
        assert config.solver == "saga"
        assert config.max_iter == 100
        assert config.tol == 1e-4
        assert config.multi_class == "auto"
        assert config.class_weight == "balanced"
        assert config.fit_intercept is True
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LogisticRegressionConfig(
            penalty="l1",
            C=0.5,
            solver="liblinear",
            max_iter=200,
            class_weight=None,
        )
        assert config.penalty == "l1"
        assert config.C == 0.5
        assert config.solver == "liblinear"
        assert config.max_iter == 200
        assert config.class_weight is None

    def test_solver_penalty_compatibility(self):
        """Test that invalid solver-penalty combinations raise errors."""
        # L1 penalty with incompatible solver
        with pytest.raises(ValueError, match="does not support L1"):
            LogisticRegressionConfig(penalty="l1", solver="lbfgs")
        
        with pytest.raises(ValueError, match="does not support L1"):
            LogisticRegressionConfig(penalty="l1", solver="newton-cg")
        
        # ElasticNet penalty with incompatible solver
        with pytest.raises(ValueError, match="does not support ElasticNet"):
            LogisticRegressionConfig(penalty="elasticnet", solver="lbfgs", l1_ratio=0.5)
        
        # None penalty with liblinear
        with pytest.raises(ValueError, match="does not support no penalty"):
            LogisticRegressionConfig(penalty=None, solver="liblinear")

    def test_valid_solver_penalty_combinations(self):
        """Test valid solver-penalty combinations."""
        # Saga supports all penalties
        for penalty in ["l1", "l2", "elasticnet", None]:
            if penalty == "elasticnet":
                config = LogisticRegressionConfig(
                    penalty=penalty, solver="saga", l1_ratio=0.5
                )
            else:
                config = LogisticRegressionConfig(penalty=penalty, solver="saga")
            assert config.penalty == penalty
            assert config.solver == "saga"
        
        # Liblinear supports L1 and L2
        for penalty in ["l1", "l2"]:
            config = LogisticRegressionConfig(penalty=penalty, solver="liblinear")
            assert config.penalty == penalty
            assert config.solver == "liblinear"
        
        # LBFGS supports L2 and None
        for penalty in ["l2", None]:
            config = LogisticRegressionConfig(penalty=penalty, solver="lbfgs")
            assert config.penalty == penalty
            assert config.solver == "lbfgs"

    def test_multinomial_solver_compatibility(self):
        """Test multinomial support across solvers."""
        # Liblinear should raise warning about OvR
        with pytest.raises(ValueError, match="does not support 'multinomial'"):
            LogisticRegressionConfig(multi_class="multinomial", solver="liblinear")
        
        # Other solvers should support multinomial
        for solver in ["lbfgs", "newton-cg", "sag", "saga"]:
            config = LogisticRegressionConfig(multi_class="multinomial", solver=solver)
            assert config.multi_class == "multinomial"
            assert config.solver == solver

    def test_l1_ratio_validation(self):
        """Test l1_ratio validation for elasticnet penalty."""
        # l1_ratio required for elasticnet
        with pytest.raises(ValueError, match="l1_ratio must be specified"):
            LogisticRegressionConfig(penalty="elasticnet", solver="saga")
        
        # l1_ratio not allowed for other penalties
        with pytest.raises(ValueError, match="l1_ratio is only used"):
            LogisticRegressionConfig(penalty="l2", l1_ratio=0.5)
        
        # Valid elasticnet config
        config = LogisticRegressionConfig(
            penalty="elasticnet", solver="saga", l1_ratio=0.7
        )
        assert config.penalty == "elasticnet"
        assert config.l1_ratio == 0.7

    def test_parameter_bounds(self):
        """Test parameter boundary validation."""
        # Test lower bounds
        with pytest.raises(ValueError):
            LogisticRegressionConfig(C=0)  # Must be > 0
        
        with pytest.raises(ValueError):
            LogisticRegressionConfig(max_iter=0)  # Must be >= 1
        
        with pytest.raises(ValueError):
            LogisticRegressionConfig(tol=0)  # Must be > 0
        
        # Test l1_ratio bounds
        with pytest.raises(ValueError):
            LogisticRegressionConfig(penalty="elasticnet", solver="saga", l1_ratio=-0.1)
        
        with pytest.raises(ValueError):
            LogisticRegressionConfig(penalty="elasticnet", solver="saga", l1_ratio=1.1)

    def test_class_weight_validation(self):
        """Test class_weight parameter validation."""
        # Valid string
        config = LogisticRegressionConfig(class_weight="balanced")
        assert config.class_weight == "balanced"
        
        # Invalid string
        with pytest.raises(ValueError, match="class_weight string must be"):
            LogisticRegressionConfig(class_weight="invalid")
        
        # Valid dict
        config = LogisticRegressionConfig(class_weight={0: 1.0, 1: 2.0})
        assert config.class_weight == {0: 1.0, 1: 2.0}
        
        # Invalid dict (negative weight)
        with pytest.raises(ValueError, match="must be positive"):
            LogisticRegressionConfig(class_weight={0: -1.0})

    def test_n_jobs_validation(self):
        """Test n_jobs parameter validation."""
        # Valid values
        for n_jobs in [None, -1, 1, 4]:
            config = LogisticRegressionConfig(n_jobs=n_jobs)
            assert config.n_jobs == n_jobs
        
        # Invalid value
        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            LogisticRegressionConfig(n_jobs=0)


class TestLogisticRegressionModel:
    """Test LogisticRegressionModel implementation."""

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
        y_series = pd.Series(y, name="target")
        # Convert to string labels for realistic VA data
        label_map = {0: "Cause_A", 1: "Cause_B", 2: "Cause_C", 3: "Cause_D", 4: "Cause_E"}
        y_series = y_series.map(label_map)
        return X_df, y_series

    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted model for testing."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)
        return model

    def test_default_initialization(self):
        """Test model initialization with default config."""
        model = LogisticRegressionModel()
        assert isinstance(model.config, LogisticRegressionConfig)
        assert model.config.penalty == "l2"
        assert model.config.solver == "saga"
        assert model._is_fitted is False
        assert model.model_ is None

    def test_custom_initialization(self):
        """Test model initialization with custom config."""
        config = LogisticRegressionConfig(
            penalty="l1",
            C=0.5,
            solver="liblinear",
        )
        model = LogisticRegressionModel(config=config)
        assert model.config.penalty == "l1"
        assert model.config.C == 0.5
        assert model.config.solver == "liblinear"

    def test_get_params(self):
        """Test get_params method."""
        config = LogisticRegressionConfig(C=0.5, max_iter=200)
        model = LogisticRegressionModel(config=config)
        
        # Shallow params
        params = model.get_params(deep=False)
        assert params == {"config": config}
        
        # Deep params
        params = model.get_params(deep=True)
        assert params["config"] == config
        assert params["config__C"] == 0.5
        assert params["config__max_iter"] == 200
        assert "config__penalty" in params

    def test_set_params(self):
        """Test set_params method."""
        model = LogisticRegressionModel()
        
        # Set entire config
        new_config = LogisticRegressionConfig(C=0.1)
        model.set_params(config=new_config)
        assert model.config.C == 0.1
        
        # Set nested parameters
        model.set_params(config__C=0.5, config__max_iter=500)
        assert model.config.C == 0.5
        assert model.config.max_iter == 500

    def test_fit_basic(self, sample_data):
        """Test basic model fitting."""
        X, y = sample_data
        model = LogisticRegressionModel()
        
        # Fit model
        fitted = model.fit(X, y)
        
        # Check return type
        assert fitted is model
        
        # Check fitted attributes
        assert model._is_fitted is True
        assert model.model_ is not None
        assert isinstance(model.model_, LogisticRegression)
        assert model.feature_names_ == X.columns.tolist()
        assert len(model.classes_) == 5
        assert model.label_encoder_ is not None

    def test_fit_with_sample_weight(self, sample_data):
        """Test fitting with custom sample weights."""
        X, y = sample_data
        model = LogisticRegressionModel()
        
        # Create custom weights
        sample_weight = np.ones(len(y))
        sample_weight[y == "Cause_A"] = 2.0  # Upweight one class
        
        # Fit with weights
        model.fit(X, y, sample_weight=sample_weight)
        assert model._is_fitted is True

    def test_fit_invalid_inputs(self):
        """Test fit with invalid inputs."""
        model = LogisticRegressionModel()
        
        # Test non-DataFrame X
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            model.fit(np.array([[1, 2], [3, 4]]), pd.Series([0, 1]))
        
        # Test non-Series y
        with pytest.raises(TypeError, match="y must be a pandas Series"):
            model.fit(pd.DataFrame([[1, 2], [3, 4]]), np.array([0, 1]))

    def test_predict(self, fitted_model, sample_data):
        """Test prediction."""
        X, y = sample_data
        
        # Make predictions
        predictions = fitted_model.predict(X[:10])
        
        # Check output
        assert len(predictions) == 10
        assert all(pred in fitted_model.classes_ for pred in predictions)
        assert isinstance(predictions, np.ndarray)

    def test_predict_proba(self, fitted_model, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        # Get probabilities
        proba = fitted_model.predict_proba(X[:10])
        
        # Check shape and properties
        assert proba.shape == (10, 5)  # 10 samples, 5 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probabilities

    def test_predict_not_fitted(self, sample_data):
        """Test prediction on unfitted model."""
        X, y = sample_data
        model = LogisticRegressionModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_predict_feature_mismatch(self, fitted_model):
        """Test prediction with mismatched features."""
        # Wrong number of features
        X_wrong = pd.DataFrame(np.random.randn(5, 10))
        
        with pytest.raises(ValueError, match="Feature names mismatch"):
            fitted_model.predict(X_wrong)

    def test_feature_importance_multiclass(self, fitted_model):
        """Test feature importance extraction for multiclass."""
        importance = fitted_model.get_feature_importance()
        
        # Check structure
        assert isinstance(importance, pd.DataFrame)
        assert list(importance.columns) == ["feature", "importance"]
        assert len(importance) == 20  # Number of features
        
        # Check sorting
        assert importance["importance"].is_monotonic_decreasing
        
        # Check values
        assert importance["importance"].min() >= 0  # Absolute values

    def test_feature_importance_binary(self):
        """Test feature importance for binary classification."""
        # Create binary classification data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y).map({0: "Class_0", 1: "Class_1"})
        
        # Fit model
        model = LogisticRegressionModel()
        model.fit(X_df, y_series)
        
        # Get importance
        importance = model.get_feature_importance()
        assert len(importance) == 10
        assert importance["importance"].min() >= 0

    def test_csmf_accuracy_calculation(self, fitted_model):
        """Test CSMF accuracy calculation."""
        # Create known distribution
        y_true = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
        y_pred = np.array(["A"] * 45 + ["B"] * 35 + ["C"] * 20)
        
        csmf_acc = fitted_model.calculate_csmf_accuracy(y_true, y_pred)
        
        # Check properties
        assert 0 <= csmf_acc <= 1
        
        # Perfect prediction
        csmf_perfect = fitted_model.calculate_csmf_accuracy(y_true, y_true.values)
        assert csmf_perfect == 1.0
        
        # Single class edge case
        y_single = pd.Series(["A"] * 100)
        csmf_single = fitted_model.calculate_csmf_accuracy(y_single, y_single.values)
        assert csmf_single == 1.0

    def test_cross_validation(self, sample_data):
        """Test cross-validation functionality."""
        X, y = sample_data
        model = LogisticRegressionModel()
        
        # Run cross-validation
        cv_results = model.cross_validate(X, y, cv=3)
        
        # Check structure
        assert "csmf_accuracy_mean" in cv_results
        assert "csmf_accuracy_std" in cv_results
        assert "cod_accuracy_mean" in cv_results
        assert "cod_accuracy_std" in cv_results
        assert "csmf_accuracy_scores" in cv_results
        assert "cod_accuracy_scores" in cv_results
        
        # Check values
        assert 0 <= cv_results["csmf_accuracy_mean"] <= 1
        assert 0 <= cv_results["cod_accuracy_mean"] <= 1
        assert len(cv_results["csmf_accuracy_scores"]) == 3
        assert len(cv_results["cod_accuracy_scores"]) == 3

    def test_cross_validation_invalid_cv(self, sample_data):
        """Test cross-validation with invalid cv parameter."""
        X, y = sample_data
        model = LogisticRegressionModel()
        
        with pytest.raises(ValueError, match="cv must be at least 2"):
            model.cross_validate(X, y, cv=1)

    def test_different_regularization_penalties(self, sample_data):
        """Test model with different regularization penalties."""
        X, y = sample_data
        
        # Test L1 regularization
        config_l1 = LogisticRegressionConfig(penalty="l1", solver="saga")
        model_l1 = LogisticRegressionModel(config=config_l1)
        model_l1.fit(X, y)
        assert model_l1._is_fitted
        
        # Test L2 regularization (default)
        model_l2 = LogisticRegressionModel()
        model_l2.fit(X, y)
        assert model_l2._is_fitted
        
        # Test ElasticNet regularization
        config_elastic = LogisticRegressionConfig(
            penalty="elasticnet", solver="saga", l1_ratio=0.5
        )
        model_elastic = LogisticRegressionModel(config=config_elastic)
        model_elastic.fit(X, y)
        assert model_elastic._is_fitted
        
        # Test no regularization
        config_none = LogisticRegressionConfig(penalty=None, solver="saga")
        model_none = LogisticRegressionModel(config=config_none)
        model_none.fit(X, y)
        assert model_none._is_fitted

    def test_sklearn_params_conversion(self):
        """Test conversion of config to sklearn parameters."""
        # Test basic conversion
        config = LogisticRegressionConfig(C=0.5, max_iter=200)
        model = LogisticRegressionModel(config=config)
        params = model._get_sklearn_params()
        
        assert params["C"] == 0.5
        assert params["max_iter"] == 200
        assert params["penalty"] == "l2"
        assert params["solver"] == "saga"
        
        # Test elasticnet params
        config_elastic = LogisticRegressionConfig(
            penalty="elasticnet", solver="saga", l1_ratio=0.7
        )
        model_elastic = LogisticRegressionModel(config=config_elastic)
        params_elastic = model_elastic._get_sklearn_params()
        
        assert params_elastic["penalty"] == "elasticnet"
        assert params_elastic["l1_ratio"] == 0.7

    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        X, y = sample_data
        model = LogisticRegressionModel()
        
        # Single class data
        y_single = pd.Series(["A"] * len(y))
        model.fit(X, y_single)
        predictions = model.predict(X[:5])
        assert all(pred == "A" for pred in predictions)
        
        # Very few samples
        model_few = LogisticRegressionModel()
        model_few.fit(X[:10], y[:10])
        assert model_few._is_fitted

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random_state."""
        X, y = sample_data
        
        # Train two models with same config
        config = LogisticRegressionConfig(random_state=42)
        model1 = LogisticRegressionModel(config=config)
        model2 = LogisticRegressionModel(config=config)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should be identical
        pred1 = model1.predict(X[:20])
        pred2 = model2.predict(X[:20])
        assert np.array_equal(pred1, pred2)
        
        # Feature importance should be identical
        imp1 = model1.get_feature_importance()
        imp2 = model2.get_feature_importance()
        pd.testing.assert_frame_equal(imp1, imp2)