"""Tests for Random Forest model implementation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.random_forest_config import RandomForestConfig
from baseline.models.random_forest_model import RandomForestModel


class TestRandomForestConfig:
    """Test RandomForestConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RandomForestConfig()
        assert config.n_estimators == 100
        assert config.max_depth is None
        assert config.min_samples_split == 2
        assert config.min_samples_leaf == 1
        assert config.max_features == "sqrt"
        assert config.bootstrap is True
        assert config.class_weight == "balanced"
        assert config.n_jobs == -1
        assert config.random_state == 42
        assert config.criterion == "gini"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RandomForestConfig(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.5,
            class_weight=None,
        )
        assert config.n_estimators == 200
        assert config.max_depth == 10
        assert config.min_samples_split == 5
        assert config.min_samples_leaf == 2
        assert config.max_features == 0.5
        assert config.class_weight is None

    def test_max_features_validation(self):
        """Test max_features validation."""
        # Valid string values
        config = RandomForestConfig(max_features="sqrt")
        assert config.max_features == "sqrt"
        
        config = RandomForestConfig(max_features="log2")
        assert config.max_features == "log2"
        
        # Valid numeric values
        config = RandomForestConfig(max_features=0.5)
        assert config.max_features == 0.5
        
        config = RandomForestConfig(max_features=10)
        assert config.max_features == 10
        
        # Invalid string
        with pytest.raises(ValueError, match="max_features string must be one of"):
            RandomForestConfig(max_features="invalid")
        
        # Invalid float
        with pytest.raises(ValueError, match="max_features float must be in"):
            RandomForestConfig(max_features=1.5)
        
        # Invalid int
        with pytest.raises(ValueError, match="max_features int must be >= 1"):
            RandomForestConfig(max_features=0)

    def test_criterion_validation(self):
        """Test criterion validation."""
        # Valid values
        for criterion in ["gini", "entropy", "log_loss"]:
            config = RandomForestConfig(criterion=criterion)
            assert config.criterion == criterion
        
        # Invalid value
        with pytest.raises(ValueError, match="criterion must be one of"):
            RandomForestConfig(criterion="invalid")

    def test_class_weight_validation(self):
        """Test class_weight validation."""
        # Valid string values
        config = RandomForestConfig(class_weight="balanced")
        assert config.class_weight == "balanced"
        
        config = RandomForestConfig(class_weight="balanced_subsample")
        assert config.class_weight == "balanced_subsample"
        
        # Valid dict
        config = RandomForestConfig(class_weight={0: 1.0, 1: 2.0})
        assert config.class_weight == {0: 1.0, 1: 2.0}
        
        # Invalid string
        with pytest.raises(ValueError, match="class_weight string must be one of"):
            RandomForestConfig(class_weight="invalid")
        
        # Invalid dict (negative weight)
        with pytest.raises(ValueError, match="Class weight for .* must be positive"):
            RandomForestConfig(class_weight={0: -1.0})

    def test_parameter_bounds(self):
        """Test parameter boundary validation."""
        # Test lower bounds
        with pytest.raises(ValueError):
            RandomForestConfig(n_estimators=0)
        
        with pytest.raises(ValueError):
            RandomForestConfig(min_samples_split=1)
        
        with pytest.raises(ValueError):
            RandomForestConfig(min_samples_leaf=0)
        
        # Test upper bounds
        with pytest.raises(ValueError):
            RandomForestConfig(n_estimators=5001)
        
        with pytest.raises(ValueError):
            RandomForestConfig(min_weight_fraction_leaf=0.6)


class TestRandomForestModel:
    """Test RandomForestModel implementation."""

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
        y_series = pd.Series(y).map(lambda x: f"Cause_{x}")
        return X_df, y_series

    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted Random Forest model."""
        X, y = sample_data
        model = RandomForestModel()
        model.fit(X, y)
        return model

    def test_initialization(self):
        """Test model initialization."""
        # Default config
        model = RandomForestModel()
        assert isinstance(model.config, RandomForestConfig)
        assert model.model_ is None
        assert model._is_fitted is False
        
        # Custom config
        config = RandomForestConfig(n_estimators=50)
        model = RandomForestModel(config=config)
        assert model.config.n_estimators == 50

    def test_get_set_params(self):
        """Test get_params and set_params methods."""
        model = RandomForestModel()
        
        # Get params
        params = model.get_params()
        assert "config" in params
        
        # Get deep params
        deep_params = model.get_params(deep=True)
        assert "config__n_estimators" in deep_params
        assert deep_params["config__n_estimators"] == 100
        
        # Set params
        model.set_params(config__n_estimators=200)
        assert model.config.n_estimators == 200
        
        # Set config directly
        new_config = RandomForestConfig(n_estimators=300)
        model.set_params(config=new_config)
        assert model.config.n_estimators == 300

    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        model = RandomForestModel()
        
        # Fit model
        fitted_model = model.fit(X, y)
        
        # Check attributes
        assert fitted_model is model
        assert model._is_fitted is True
        assert model.model_ is not None
        assert model.label_encoder_ is not None
        assert model.feature_names_ == X.columns.tolist()
        assert len(model.classes_) == 5
        
        # Check sklearn model
        assert hasattr(model.model_, "predict")
        assert hasattr(model.model_, "predict_proba")

    def test_fit_invalid_inputs(self, sample_data):
        """Test fit with invalid inputs."""
        X, y = sample_data
        model = RandomForestModel()
        
        # X not DataFrame
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            model.fit(X.values, y)
        
        # y not Series
        with pytest.raises(TypeError, match="y must be a pandas Series"):
            model.fit(X, y.values)

    def test_predict(self, fitted_model, sample_data):
        """Test prediction."""
        X, _ = sample_data
        
        # Predict
        predictions = fitted_model.predict(X)
        
        # Check output
        assert len(predictions) == len(X)
        assert all(pred.startswith("Cause_") for pred in predictions)
        assert set(predictions).issubset(set(fitted_model.classes_))

    def test_predict_proba(self, fitted_model, sample_data):
        """Test probability prediction."""
        X, _ = sample_data
        
        # Predict probabilities
        proba = fitted_model.predict_proba(X)
        
        # Check output shape
        assert proba.shape == (len(X), len(fitted_model.classes_))
        
        # Check probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        
        # Check range [0, 1]
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_predict_not_fitted(self, sample_data):
        """Test prediction without fitting."""
        X, _ = sample_data
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_predict_feature_mismatch(self, fitted_model):
        """Test prediction with mismatched features."""
        # Wrong number of features
        X_wrong = pd.DataFrame(np.random.randn(10, 15))
        
        with pytest.raises(ValueError, match="Feature names mismatch"):
            fitted_model.predict(X_wrong)

    def test_get_feature_importance_mdi(self, fitted_model, sample_data):
        """Test MDI feature importance."""
        importance = fitted_model.get_feature_importance("mdi")
        
        # Check output
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == 20  # Number of features
        
        # Check importance properties
        assert importance["importance"].sum() == pytest.approx(1.0, rel=1e-5)
        assert all(importance["importance"] >= 0)
        assert importance["importance"].iloc[0] >= importance["importance"].iloc[-1]  # Sorted

    def test_get_feature_importance_permutation(self, fitted_model, sample_data):
        """Test permutation feature importance."""
        X, y = sample_data
        importance = fitted_model.get_feature_importance("permutation", X, y)
        
        # Check output
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert "importance_std" in importance.columns
        assert len(importance) == 20
        
        # Check sorted
        assert importance["importance"].iloc[0] >= importance["importance"].iloc[-1]

    def test_get_feature_importance_invalid(self, fitted_model, sample_data):
        """Test feature importance with invalid inputs."""
        X, y = sample_data
        
        # Invalid type
        with pytest.raises(ValueError, match="importance_type must be one of"):
            fitted_model.get_feature_importance("invalid")
        
        # Missing X, y for permutation
        with pytest.raises(ValueError, match="X and y must be provided"):
            fitted_model.get_feature_importance("permutation")

    def test_get_feature_importance_not_fitted(self):
        """Test feature importance without fitting."""
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_feature_importance("mdi")

    def test_calculate_csmf_accuracy(self, fitted_model):
        """Test CSMF accuracy calculation."""
        # Perfect prediction
        y_true = pd.Series(["A", "A", "B", "B", "C"])
        y_pred = np.array(["A", "A", "B", "B", "C"])
        csmf_acc = fitted_model.calculate_csmf_accuracy(y_true, y_pred)
        assert csmf_acc == 1.0
        
        # Imperfect prediction
        y_true = pd.Series(["A", "A", "B", "B", "C"])
        y_pred = np.array(["A", "B", "B", "B", "C"])
        csmf_acc = fitted_model.calculate_csmf_accuracy(y_true, y_pred)
        assert 0 < csmf_acc < 1
        
        # All wrong (worst case)
        y_true = pd.Series(["A", "A", "A", "A", "A"])
        y_pred = np.array(["B", "B", "B", "B", "B"])
        csmf_acc = fitted_model.calculate_csmf_accuracy(y_true, y_pred)
        assert csmf_acc == 0.0
        
        # Single class edge case
        y_true = pd.Series(["A", "A", "A"])
        y_pred = np.array(["A", "A", "A"])
        csmf_acc = fitted_model.calculate_csmf_accuracy(y_true, y_pred)
        assert csmf_acc == 1.0

    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        model = RandomForestModel(config=RandomForestConfig(n_estimators=10))  # Fast
        
        # Run cross-validation
        cv_results = model.cross_validate(X, y, cv=3)
        
        # Check output structure
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
        
        # Test non-stratified
        cv_results_no_strat = model.cross_validate(X, y, cv=3, stratified=False)
        assert "csmf_accuracy_mean" in cv_results_no_strat

    def test_cross_validate_invalid_cv(self, sample_data):
        """Test cross-validation with invalid cv parameter."""
        X, y = sample_data
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="cv must be at least 2"):
            model.cross_validate(X, y, cv=1)

    def test_oob_score(self, sample_data):
        """Test out-of-bag score functionality."""
        X, y = sample_data
        config = RandomForestConfig(oob_score=True, bootstrap=True)
        model = RandomForestModel(config=config)
        
        # Fit with OOB
        model.fit(X, y)
        
        # Check OOB score is computed
        assert hasattr(model.model_, "oob_score_")
        assert 0 <= model.model_.oob_score_ <= 1

    def test_integration_with_va_data_pattern(self):
        """Test integration pattern matching VA data usage."""
        # Simulate VA data structure
        n_samples = 100
        n_features = 50
        
        # Create data similar to VA format
        X = pd.DataFrame(
            np.random.randint(0, 2, size=(n_samples, n_features)),
            columns=[f"symptom_{i}" for i in range(n_features)]
        )
        y = pd.Series(
            np.random.choice(
                ["Malaria", "TB", "HIV/AIDS", "Pneumonia", "Other"], 
                size=n_samples, 
                p=[0.3, 0.2, 0.2, 0.2, 0.1]
            )
        )
        
        # Test full pipeline
        model = RandomForestModel()
        model.fit(X, y)
        predictions = model.predict(X)
        proba = model.predict_proba(X)
        importance = model.get_feature_importance("mdi")
        
        # Verify outputs
        assert len(predictions) == n_samples
        assert proba.shape == (n_samples, 5)
        assert len(importance) == n_features