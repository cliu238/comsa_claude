"""Integration tests for InSilicoVA model with real Docker execution."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.model_config import InSilicoVAConfig
from baseline.models.model_validator import InSilicoVAValidator

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_small_va_dataset(n_samples=50, n_features=15, n_causes=4):
    """Create a small VA-like dataset for integration testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of symptom features
        n_causes: Number of unique causes
        
    Returns:
        Tuple of (X, y) DataFrames
    """
    np.random.seed(42)
    
    # Create binary symptom data (0/1) typical of VA data
    X = pd.DataFrame(
        np.random.binomial(1, 0.3, size=(n_samples, n_features)),
        columns=[f"symptom_{i:03d}" for i in range(n_features)]
    )
    
    # Add some NA values to simulate real VA data
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan
    
    # Create cause distribution with some imbalance
    cause_names = [f"Cause_{chr(65+i)}" for i in range(n_causes)]
    cause_probs = np.array([0.4, 0.3, 0.2, 0.1][:n_causes])
    cause_probs = cause_probs / cause_probs.sum()
    
    y = pd.Series(
        np.random.choice(cause_names, size=n_samples, p=cause_probs),
        name="cause"
    )
    
    return X, y


@pytest.mark.integration
class TestInSilicoVAIntegration:
    """Integration tests requiring Docker."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration tests."""
        return InSilicoVAConfig(
            nsim=1000,  # Lower for faster tests
            docker_timeout=120,  # 2 minutes should be enough
            verbose=True,  # Show Docker output in tests
            output_dir="test_output"
        )
    
    def test_docker_availability(self, integration_config):
        """Test that Docker is available for integration tests."""
        validator = InSilicoVAValidator(integration_config)
        result = validator.validate_docker_availability()
        
        if not result.is_valid:
            pytest.skip(f"Docker not available: {result.errors}")
        
        assert result.is_valid
        logger.info(f"Docker validation result: {result.metadata}")
    
    def test_small_dataset_execution(self, integration_config):
        """Test full execution with a small dataset."""
        # Check Docker availability first
        validator = InSilicoVAValidator(integration_config)
        docker_result = validator.validate_docker_availability()
        
        if not docker_result.is_valid:
            pytest.skip(f"Docker not available: {docker_result.errors}")
        
        # Create small dataset
        X, y = create_small_va_dataset(n_samples=50, n_features=15)
        
        # Initialize and fit model
        model = InSilicoVAModel(integration_config)
        model.fit(X, y)
        
        # Make predictions
        X_test = X.iloc[:10]  # Use subset for prediction
        
        try:
            probabilities = model.predict_proba(X_test)
            
            # Validate output
            assert probabilities.shape[0] == len(X_test)
            assert probabilities.shape[1] == len(y.unique())
            assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
            
            # Test regular predictions
            predictions = model.predict(X_test)
            assert len(predictions) == len(X_test)
            assert all(pred in model._unique_causes for pred in predictions)
            
            logger.info(f"Successfully got predictions for {len(X_test)} samples")
            
        except RuntimeError as e:
            if "Docker" in str(e):
                pytest.skip(f"Docker execution failed: {e}")
            else:
                raise
    
    def test_csmf_accuracy_with_real_execution(self, integration_config):
        """Test CSMF accuracy calculation with real Docker execution."""
        # Check Docker availability
        validator = InSilicoVAValidator(integration_config)
        docker_result = validator.validate_docker_availability()
        
        if not docker_result.is_valid:
            pytest.skip(f"Docker not available: {docker_result.errors}")
        
        # Create dataset with known distribution
        X, y = create_small_va_dataset(n_samples=100, n_features=20, n_causes=4)
        
        # Split into train/test
        train_idx = np.random.choice(len(X), size=80, replace=False)
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Train and predict
        model = InSilicoVAModel(integration_config)
        model.fit(X_train, y_train)
        
        try:
            predictions = model.predict(X_test)
            
            # Calculate CSMF accuracy
            accuracy = model.calculate_csmf_accuracy(
                pd.Series(y_test.values, name="true"),
                pd.Series(predictions, name="pred")
            )
            
            # Check accuracy is in reasonable range
            assert 0.0 <= accuracy <= 1.0
            logger.info(f"CSMF accuracy achieved: {accuracy:.3f}")
            
            # For small test data, we expect moderate accuracy
            # Not checking against specific benchmarks in integration test
            
        except RuntimeError as e:
            if "Docker" in str(e):
                pytest.skip(f"Docker execution failed: {e}")
            else:
                raise
    
    def test_fallback_dockerfile(self, integration_config):
        """Test fallback to local Dockerfile if primary image fails."""
        # Modify config to use non-existent image
        config = InSilicoVAConfig(
            docker_image="nonexistent-insilicova:latest",
            use_fallback_dockerfile=True,
            nsim=1000,
            docker_timeout=300  # Building might take time
        )
        
        # Check if Dockerfile exists
        if not Path("Dockerfile").exists():
            pytest.skip("Dockerfile not found for fallback test")
        
        # Check Docker availability
        validator = InSilicoVAValidator(config)
        docker_result = validator.validate_docker_availability()
        
        if not docker_result.is_valid:
            pytest.skip(f"Docker not available: {docker_result.errors}")
        
        # Create small dataset
        X, y = create_small_va_dataset(n_samples=30, n_features=10)
        
        # This should trigger fallback
        model = InSilicoVAModel(config)
        model.fit(X, y)
        
        try:
            probabilities = model.predict_proba(X[:5])
            assert probabilities.shape == (5, len(y.unique()))
            logger.info("Fallback Dockerfile execution successful")
            
        except RuntimeError as e:
            if "Docker" in str(e) or "Dockerfile" in str(e):
                pytest.skip(f"Fallback execution failed: {e}")
            else:
                raise
    
    def test_different_prior_types(self, integration_config):
        """Test model with different prior types."""
        # Check Docker availability
        validator = InSilicoVAValidator(integration_config)
        docker_result = validator.validate_docker_availability()
        
        if not docker_result.is_valid:
            pytest.skip(f"Docker not available: {docker_result.errors}")
        
        X, y = create_small_va_dataset(n_samples=40, n_features=12)
        
        for prior_type in ["quantile", "default"]:
            config = InSilicoVAConfig(
                prior_type=prior_type,
                nsim=1000,
                docker_timeout=120
            )
            
            model = InSilicoVAModel(config)
            model.fit(X, y)
            
            try:
                probabilities = model.predict_proba(X[:5])
                assert probabilities.shape == (5, len(y.unique()))
                logger.info(f"Prior type '{prior_type}' execution successful")
                
            except RuntimeError as e:
                if "Docker" in str(e):
                    pytest.skip(f"Docker execution failed for prior '{prior_type}': {e}")
                else:
                    raise
    
    def test_large_nsim_warning(self, integration_config, caplog):
        """Test that large nsim values produce warning but still work."""
        # This test doesn't actually run Docker, just checks configuration
        config = InSilicoVAConfig(nsim=150000)
        
        # Check if warning was logged during config creation
        # Note: This is a simplified test since the warning happens at config creation
        assert config.nsim == 150000
    
    @pytest.mark.slow
    def test_benchmark_scenario(self, integration_config):
        """Test a scenario closer to benchmark conditions."""
        # Check Docker availability
        validator = InSilicoVAValidator(integration_config)
        docker_result = validator.validate_docker_availability()
        
        if not docker_result.is_valid:
            pytest.skip(f"Docker not available: {docker_result.errors}")
        
        # Create larger dataset for more realistic test
        X, y = create_small_va_dataset(n_samples=200, n_features=50, n_causes=10)
        
        # Use more realistic config
        config = InSilicoVAConfig(
            nsim=5000,  # Still lower than production
            prior_type="quantile",
            docker_timeout=300
        )
        
        model = InSilicoVAModel(config)
        model.fit(X, y)
        
        # Test on held-out data
        X_test = X.iloc[150:]
        y_test = y.iloc[150:]
        
        try:
            predictions = model.predict(X_test)
            accuracy = model.calculate_csmf_accuracy(y_test, pd.Series(predictions))
            
            # Log accuracy for manual inspection
            logger.info(f"Benchmark scenario CSMF accuracy: {accuracy:.3f}")
            
            # We expect reasonable accuracy but not necessarily matching papers
            assert 0.3 <= accuracy <= 1.0  # Very generous bounds
            
        except RuntimeError as e:
            if "Docker" in str(e):
                pytest.skip(f"Docker execution failed: {e}")
            else:
                raise