"""Tests for XGBoost model with medical prior integration."""

import numpy as np
import pytest

from baseline.models import XGBoostPriorConfig, XGBoostPriorEnhanced


class TestXGBoostPriorEnhanced:
    """Test suite for prior-enhanced XGBoost model."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample VA-like data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 15  # Simulated symptoms
        n_classes = 5  # Simulated causes
        
        # Generate binary symptom data
        X = np.random.binomial(1, 0.3, size=(n_samples, n_features))
        
        # Generate labels with some structure (certain symptoms more likely with certain causes)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Simple rule: if first 3 symptoms present, likely cause 0
            if X[i, :3].sum() >= 2:
                y[i] = 0
            # If symptoms 3-6 present, likely cause 1
            elif X[i, 3:6].sum() >= 2:
                y[i] = 1
            # Otherwise random
            else:
                y[i] = np.random.randint(0, n_classes)
                
        return X, y
    
    @pytest.fixture
    def basic_config(self):
        """Create basic configuration for testing."""
        return XGBoostPriorConfig(
            n_estimators=10,  # Small for fast tests
            max_depth=3,
            use_medical_priors=True,
            prior_method="both",
            lambda_prior=0.1
        )
    
    def test_initialization(self, basic_config):
        """Test model initialization."""
        model = XGBoostPriorEnhanced(basic_config)
        assert model.config.use_medical_priors is True
        assert model.config.prior_method == "both"
        assert model.config.lambda_prior == 0.1
        assert model.priors is None  # Not loaded until fit
        
    def test_initialization_without_config(self):
        """Test initialization with default config."""
        model = XGBoostPriorEnhanced()
        assert model.config.use_medical_priors is True
        assert model.config.prior_method == "both"
        
    def test_fit_with_priors(self, sample_data, basic_config):
        """Test fitting with medical priors."""
        X, y = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        
        # Fit the model
        model.fit(X, y)
        
        # Check that priors were loaded
        assert model.priors is not None
        assert model.prior_calculator is not None
        assert model.prior_constraints is not None
        assert model._fitted is True
        
    def test_fit_without_priors(self, sample_data):
        """Test fitting without medical priors."""
        X, y = sample_data
        config = XGBoostPriorConfig(
            n_estimators=10,
            use_medical_priors=False
        )
        model = XGBoostPriorEnhanced(config)
        
        # Fit the model
        model.fit(X, y)
        
        # Check that priors were not loaded
        assert model.priors is None
        assert model._fitted is True
        
    def test_predict_proba(self, sample_data, basic_config):
        """Test probability predictions."""
        X, y = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        model.fit(X, y)
        
        # Make predictions
        probs = model.predict_proba(X[:10])
        
        # Check output shape and properties
        assert probs.shape == (10, len(np.unique(y)))
        assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities
        
    def test_predict(self, sample_data, basic_config):
        """Test class predictions."""
        X, y = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        # Check output
        assert predictions.shape == (10,)
        assert np.all(np.isin(predictions, np.unique(y)))
        
    def test_feature_engineering_only(self, sample_data):
        """Test using only feature engineering method."""
        X, y = sample_data
        config = XGBoostPriorConfig(
            n_estimators=10,
            prior_method="feature_engineering"
        )
        model = XGBoostPriorEnhanced(config)
        model.fit(X, y)
        
        # Check that features were augmented
        assert len(model._prior_feature_names) > 0
        
        # Make predictions
        probs = model.predict_proba(X[:10])
        assert probs.shape[0] == 10
        
    def test_custom_objective_only(self, sample_data):
        """Test using only custom objective method."""
        X, y = sample_data
        config = XGBoostPriorConfig(
            n_estimators=10,
            prior_method="custom_objective"
        )
        model = XGBoostPriorEnhanced(config)
        model.fit(X, y)
        
        # Check that custom objective was used
        assert hasattr(model, 'custom_objective')
        assert model.custom_objective is not None
        
        # Make predictions
        probs = model.predict_proba(X[:10])
        assert probs.shape[0] == 10
        
    def test_prior_influence_report(self, sample_data, basic_config):
        """Test prior influence reporting."""
        X, y = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        model.fit(X, y)
        
        # Get influence report
        report = model.get_prior_influence_report()
        
        # Check report contents
        assert "kl_divergence" in report
        assert "correlation" in report
        assert "avg_contribution" in report
        assert "lambda_prior" in report
        assert report["lambda_prior"] == 0.1
        
    def test_feature_importance(self, sample_data, basic_config):
        """Test feature importance with prior features."""
        X, y = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        model.fit(X, y)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Check that we have importance for both original and prior features
        assert len(importance) > X.shape[1]  # More features than original
        
        # Check that prior features are included
        prior_features = [k for k in importance.keys() if k.startswith("prior_")]
        assert len(prior_features) > 0
        
    def test_csmf_accuracy(self, sample_data, basic_config):
        """Test CSMF accuracy calculation."""
        X, y = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        model.fit(X, y)
        
        # Calculate CSMF accuracy
        csmf_acc = model.calculate_csmf_accuracy(X, y)
        
        # Check valid range
        assert 0 <= csmf_acc <= 1
        
    def test_not_fitted_error(self, sample_data, basic_config):
        """Test error when predicting without fitting."""
        X, _ = sample_data
        model = XGBoostPriorEnhanced(basic_config)
        
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)
            
        with pytest.raises(ValueError, match="fitted"):
            model.predict_proba(X)
            
    def test_feature_selection_config(self, sample_data):
        """Test selective feature inclusion."""
        X, y = sample_data
        
        # Test with only likelihood features
        config = XGBoostPriorConfig(
            n_estimators=10,
            prior_method="feature_engineering",
            include_likelihood_features=True,
            include_log_odds_features=False,
            include_rank_features=False,
            include_cause_prior_features=False,
            include_plausibility_features=False
        )
        model = XGBoostPriorEnhanced(config)
        model.fit(X, y)
        
        # Check that only likelihood features were added
        likelihood_features = [f for f in model._prior_feature_names if "likelihood" in f]
        assert len(likelihood_features) > 0
        assert len(likelihood_features) == len(model._prior_feature_names)
        
    def test_lambda_scheduling(self, sample_data):
        """Test different lambda schedules."""
        X, y = sample_data
        
        for schedule in ["constant", "linear_decay", "exponential_decay"]:
            config = XGBoostPriorConfig(
                n_estimators=10,
                lambda_schedule=schedule,
                lambda_prior=0.5,
                lambda_min=0.1
            )
            model = XGBoostPriorEnhanced(config)
            model.fit(X, y)
            
            # Should fit without errors
            assert model._fitted is True
            
    def test_sample_weights(self, sample_data, basic_config):
        """Test fitting with sample weights."""
        X, y = sample_data
        weights = np.random.rand(len(y))
        
        model = XGBoostPriorEnhanced(basic_config)
        model.fit(X, y, sample_weight=weights)
        
        # Should handle weights properly
        assert model._fitted is True
        
    def test_repr(self, basic_config):
        """Test string representation."""
        model = XGBoostPriorEnhanced(basic_config)
        repr_str = repr(model)
        
        assert "XGBoostPriorEnhanced" in repr_str
        assert "method=both" in repr_str
        assert "λ=0.1" in repr_str
        
        # Test with priors disabled
        config_no_priors = XGBoostPriorConfig(use_medical_priors=False)
        model_no_priors = XGBoostPriorEnhanced(config_no_priors)
        repr_str_no_priors = repr(model_no_priors)
        
        assert "priors=disabled" in repr_str_no_priors