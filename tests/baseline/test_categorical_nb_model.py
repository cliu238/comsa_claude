"""Tests for CategoricalNB model implementation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import CategoricalNB

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.categorical_nb_config import CategoricalNBConfig
from baseline.models.categorical_nb_model import CategoricalNBModel


class TestCategoricalNBConfig:
    """Test CategoricalNBConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CategoricalNBConfig()
        assert config.alpha == 1.0
        assert config.fit_prior is True
        assert config.class_prior is None
        assert config.force_alpha is False
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CategoricalNBConfig(
            alpha=0.5,
            fit_prior=False,
            force_alpha=True,
            random_state=123,
        )
        assert config.alpha == 0.5
        assert config.fit_prior is False
        assert config.force_alpha is True
        assert config.random_state == 123

    def test_alpha_validation(self):
        """Test alpha parameter validation."""
        # Valid positive alpha
        config = CategoricalNBConfig(alpha=0.5)
        assert config.alpha == 0.5
        
        # Invalid alpha (zero) - Pydantic ValidationError
        with pytest.raises(Exception):  # Catches pydantic ValidationError
            CategoricalNBConfig(alpha=0)
        
        # Invalid alpha (negative) - Pydantic ValidationError
        with pytest.raises(Exception):  # Catches pydantic ValidationError
            CategoricalNBConfig(alpha=-1.0)

    def test_class_prior_validation(self):
        """Test class_prior parameter validation."""
        # Valid None
        config = CategoricalNBConfig(class_prior=None)
        assert config.class_prior is None
        
        # Valid numpy array
        prior = np.array([0.3, 0.7])
        config = CategoricalNBConfig(class_prior=prior)
        np.testing.assert_array_equal(config.class_prior, prior)
        
        # Valid list (should be converted to numpy array)
        config = CategoricalNBConfig(class_prior=[0.25, 0.75])
        expected = np.array([0.25, 0.75])
        np.testing.assert_array_equal(config.class_prior, expected)
        
        # Invalid - negative values
        with pytest.raises(ValueError, match="must be non-negative"):
            CategoricalNBConfig(class_prior=[-0.1, 1.1])
        
        # Invalid - doesn't sum to 1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            CategoricalNBConfig(class_prior=[0.3, 0.5])

    def test_random_state_validation(self):
        """Test random_state parameter validation."""
        # Valid positive random_state
        config = CategoricalNBConfig(random_state=42)
        assert config.random_state == 42
        
        # Valid zero random_state
        config = CategoricalNBConfig(random_state=0)
        assert config.random_state == 0
        
        # Invalid negative random_state
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            CategoricalNBConfig(random_state=-1)


class TestCategoricalNBModel:
    """Test CategoricalNBModel implementation."""

    @pytest.fixture
    def sample_va_data(self):
        """Create sample VA-like categorical data."""
        # Create categorical VA data with Y/N/./DK patterns
        X_data = {
            'symptom1': ['Y', 'N', '.', 'Y', 'DK', 'N', 'Y', '.'],
            'symptom2': ['N', 'Y', 'Y', '.', 'N', 'DK', 'Y', 'N'],
            'symptom3': ['.', 'DK', 'Y', 'N', 'Y', 'Y', 'N', '.'],
        }
        X = pd.DataFrame(X_data)
        y = pd.Series(['cause1', 'cause2', 'cause1', 'cause2', 'cause1', 'cause2', 'cause1', 'cause2'])
        return X, y

    @pytest.fixture
    def sample_multiclass_data(self):
        """Create sample multi-class classification data with categorical features."""
        # Generate base data
        X_numeric, y_numeric = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        
        # Convert to categorical VA-like format
        X_categorical = pd.DataFrame()
        for i in range(X_numeric.shape[1]):
            # Convert numeric features to categorical Y/N/. based on quantiles
            col = X_numeric[:, i]
            q33, q67 = np.percentile(col, [33, 67])
            categorical_col = []
            for val in col:
                if val < q33:
                    categorical_col.append('N')
                elif val < q67:
                    categorical_col.append('Y')
                else:
                    categorical_col.append('.')
            X_categorical[f'feature_{i}'] = categorical_col
        
        # Convert numeric labels to cause names
        label_map = {0: "Cause_A", 1: "Cause_B", 2: "Cause_C", 3: "Cause_D"}
        y_categorical = pd.Series([label_map[y] for y in y_numeric], name="target")
        
        return X_categorical, y_categorical

    @pytest.fixture
    def fitted_model(self, sample_multiclass_data):
        """Create a fitted model for testing."""
        X, y = sample_multiclass_data
        model = CategoricalNBModel()
        model.fit(X, y)
        return model

    def test_default_initialization(self):
        """Test model initialization with default config."""
        model = CategoricalNBModel()
        assert isinstance(model.config, CategoricalNBConfig)
        assert model.config.alpha == 1.0
        assert model.config.fit_prior is True
        assert model._is_fitted is False
        assert model.model_ is None

    def test_custom_initialization(self):
        """Test model initialization with custom config."""
        config = CategoricalNBConfig(
            alpha=0.5,
            fit_prior=False,
            force_alpha=True,
        )
        model = CategoricalNBModel(config=config)
        assert model.config.alpha == 0.5
        assert model.config.fit_prior is False
        assert model.config.force_alpha is True

    def test_get_params(self):
        """Test get_params method."""
        config = CategoricalNBConfig(alpha=0.5, fit_prior=False)
        model = CategoricalNBModel(config=config)
        
        # Shallow params
        params = model.get_params(deep=False)
        assert params == {"config": config}
        
        # Deep params
        params = model.get_params(deep=True)
        assert params["config"] == config
        assert params["config__alpha"] == 0.5
        assert params["config__fit_prior"] is False
        assert "config__random_state" in params

    def test_set_params(self):
        """Test set_params method."""
        model = CategoricalNBModel()
        
        # Set entire config
        new_config = CategoricalNBConfig(alpha=0.1)
        model.set_params(config=new_config)
        assert model.config.alpha == 0.1
        
        # Set nested parameters
        model.set_params(config__alpha=0.5, config__fit_prior=False)
        assert model.config.alpha == 0.5
        assert model.config.fit_prior is False

    def test_fit_basic(self, sample_va_data):
        """Test basic model fitting with VA data."""
        X, y = sample_va_data
        model = CategoricalNBModel()
        
        # Fit model
        fitted = model.fit(X, y)
        
        # Check return type
        assert fitted is model
        
        # Check fitted attributes
        assert model._is_fitted is True
        assert model.model_ is not None
        assert isinstance(model.model_, CategoricalNB)
        assert model.feature_names_ == X.columns.tolist()
        assert len(model.classes_) == 2  # cause1, cause2
        assert model.label_encoder_ is not None

    def test_fit_multiclass(self, sample_multiclass_data):
        """Test fitting with multiclass data."""
        X, y = sample_multiclass_data
        model = CategoricalNBModel()
        
        model.fit(X, y)
        
        assert model._is_fitted is True
        assert len(model.classes_) == 4  # 4 classes
        assert len(model.feature_names_) == 5  # 5 features

    def test_fit_invalid_inputs(self):
        """Test fit with invalid inputs."""
        model = CategoricalNBModel()
        
        # Test non-DataFrame X
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            model.fit(np.array([['Y', 'N'], ['N', 'Y']]), pd.Series(['A', 'B']))
        
        # Test non-Series y
        with pytest.raises(TypeError, match="y must be a pandas Series"):
            model.fit(pd.DataFrame([['Y', 'N'], ['N', 'Y']]), np.array(['A', 'B']))

    def test_categorical_encoding(self):
        """Test VA categorical encoding works correctly."""
        model = CategoricalNBModel()
        
        # Test various VA encodings
        X = pd.DataFrame({
            'feature1': ['Y', 'N', '.', 'DK', np.nan, 'yes', 'no', ''],
            'feature2': [1, 0, '.', 'unknown', None, 'True', 'False', 'missing']
        })
        
        encoded = model._prepare_categorical_features(X)
        
        # Check shape
        assert encoded.shape == (8, 2)
        
        # Check specific encodings
        # Y=0, N=1, .=2, DK=2, nan=2, yes=0, no=1, ''=2
        expected_col1 = [0, 1, 2, 2, 2, 0, 1, 2]
        # 1=0, 0=1, .=2, unknown=2, None=2, True=0, False=1, missing=2
        expected_col2 = [0, 1, 2, 2, 2, 0, 1, 2]
        
        np.testing.assert_array_equal(encoded[:, 0], expected_col1)
        np.testing.assert_array_equal(encoded[:, 1], expected_col2)

    def test_predict(self, fitted_model, sample_multiclass_data):
        """Test prediction."""
        X, y = sample_multiclass_data
        
        # Make predictions
        predictions = fitted_model.predict(X[:10])
        
        # Check output
        assert len(predictions) == 10
        assert all(pred in fitted_model.classes_ for pred in predictions)
        assert isinstance(predictions, np.ndarray)

    def test_predict_proba(self, fitted_model, sample_multiclass_data):
        """Test probability prediction."""
        X, y = sample_multiclass_data
        
        # Get probabilities
        proba = fitted_model.predict_proba(X[:10])
        
        # Check shape and properties
        assert proba.shape == (10, 4)  # 10 samples, 4 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probabilities

    def test_predict_not_fitted(self, sample_va_data):
        """Test prediction on unfitted model."""
        X, y = sample_va_data
        model = CategoricalNBModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_predict_feature_mismatch(self, fitted_model):
        """Test prediction with mismatched features."""
        # Wrong number of features
        X_wrong = pd.DataFrame({
            'wrong_feature1': ['Y', 'N'],
            'wrong_feature2': ['Y', 'N']
        })
        
        with pytest.raises(ValueError, match="Feature names mismatch"):
            fitted_model.predict(X_wrong)

    def test_single_class_handling(self):
        """Test handling of single class gracefully."""
        X = pd.DataFrame({
            'feature1': ['Y', 'N', 'Y'],
            'feature2': ['N', 'Y', '.']
        })
        y = pd.Series(['single_cause'] * 3)
        
        model = CategoricalNBModel()
        model.fit(X, y)
        
        # Check single class attributes
        assert model._single_class is True
        assert model._single_class_label == 'single_cause'
        
        # Test predictions
        predictions = model.predict(X)
        assert all(pred == 'single_cause' for pred in predictions)
        
        # Test probabilities
        proba = model.predict_proba(X)
        assert proba.shape == (3, 1)
        assert np.allclose(proba, 1.0)

    def test_feature_importance_multiclass(self, fitted_model):
        """Test feature importance extraction for multiclass."""
        importance = fitted_model.get_feature_importance()
        
        # Check structure
        assert isinstance(importance, pd.DataFrame)
        assert list(importance.columns) == ["feature", "importance"]
        assert len(importance) == 5  # Number of features
        
        # Check sorting
        assert importance["importance"].is_monotonic_decreasing
        
        # Check values are non-negative (log probability differences should be >= 0)
        assert importance["importance"].min() >= 0

    def test_feature_importance_single_class(self):
        """Test feature importance for single class case."""
        X = pd.DataFrame({
            'feature1': ['Y', 'N'],
            'feature2': ['Y', 'N']
        })
        y = pd.Series(['single_cause'] * 2)
        
        model = CategoricalNBModel()
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        # All importance should be zero for single class
        assert len(importance) == 2
        assert all(imp == 0 for imp in importance["importance"])

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

    def test_cross_validation(self, sample_multiclass_data):
        """Test cross-validation functionality."""
        X, y = sample_multiclass_data
        model = CategoricalNBModel()
        
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

    def test_cross_validation_invalid_cv(self, sample_va_data):
        """Test cross-validation with invalid cv parameter."""
        X, y = sample_va_data
        model = CategoricalNBModel()
        
        with pytest.raises(ValueError, match="cv must be at least 2"):
            model.cross_validate(X, y, cv=1)

    def test_different_config_parameters(self, sample_va_data):
        """Test model with different configuration parameters."""
        X, y = sample_va_data
        
        # Test with different alpha
        config_alpha = CategoricalNBConfig(alpha=0.1)
        model_alpha = CategoricalNBModel(config=config_alpha)
        model_alpha.fit(X, y)
        assert model_alpha._is_fitted
        
        # Test without learning priors
        config_no_prior = CategoricalNBConfig(fit_prior=False)
        model_no_prior = CategoricalNBModel(config=config_no_prior)
        model_no_prior.fit(X, y)
        assert model_no_prior._is_fitted
        
        # Test with force_alpha
        config_force = CategoricalNBConfig(alpha=0.01, force_alpha=True)
        model_force = CategoricalNBModel(config=config_force)
        model_force.fit(X, y)
        assert model_force._is_fitted

    def test_sklearn_params_conversion(self):
        """Test conversion of config to sklearn parameters."""
        # Test basic conversion
        config = CategoricalNBConfig(alpha=0.5, fit_prior=False)
        model = CategoricalNBModel(config=config)
        params = model._get_sklearn_params()
        
        assert params["alpha"] == 0.5
        assert params["fit_prior"] is False
        assert params["force_alpha"] is False
        # class_prior should be included even if None
        assert "class_prior" in params
        assert params["class_prior"] is None

    def test_sample_weight_warning(self, sample_va_data):
        """Test that sample_weight parameter generates warning."""
        X, y = sample_va_data
        model = CategoricalNBModel()
        
        # Should complete successfully but log warning
        sample_weight = np.ones(len(y))
        model.fit(X, y, sample_weight=sample_weight)
        assert model._is_fitted

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very small dataset
        X_small = pd.DataFrame({
            'feature1': ['Y', 'N'],
            'feature2': ['N', 'Y']
        })
        y_small = pd.Series(['A', 'B'])
        
        model = CategoricalNBModel()
        model.fit(X_small, y_small)
        
        predictions = model.predict(X_small)
        assert len(predictions) == 2
        assert all(pred in ['A', 'B'] for pred in predictions)

    def test_reproducibility(self, sample_multiclass_data):
        """Test that results are reproducible with same random_state."""
        X, y = sample_multiclass_data
        
        # Train two models with same config
        config = CategoricalNBConfig(random_state=42)
        model1 = CategoricalNBModel(config=config)
        model2 = CategoricalNBModel(config=config)
        
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

    def test_comprehensive_categorical_mapping(self):
        """Test comprehensive categorical mapping handles various input formats."""
        model = CategoricalNBModel()
        
        # Test comprehensive mapping
        X = pd.DataFrame({
            'feature1': ['Y', 'YES', 'Yes', 'yes', 'y', 1, '1', True, 'True', 'true'],
            'feature2': ['N', 'NO', 'No', 'no', 'n', 0, '0', False, 'False', 'false'],
            'feature3': ['.', 'DK', 'dk', np.nan, None, '', ' ', 'missing', 'unknown', 'NA']
        })
        
        encoded = model._prepare_categorical_features(X)
        
        # All Y variants should map to 0
        assert all(encoded[:, 0] == 0)
        
        # All N variants should map to 1
        assert all(encoded[:, 1] == 1)
        
        # All missing variants should map to 2
        assert all(encoded[:, 2] == 2)