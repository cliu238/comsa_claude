"""Tests for InSilicoVA model implementation."""

import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.model_config import InSilicoVAConfig


class TestInSilicoVAModel:
    """Test cases for InSilicoVA model."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return InSilicoVAConfig(
            nsim=1000,  # Lower for faster tests
            docker_timeout=60,  # Minimum allowed timeout
            verbose=False
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create feature data
        X = pd.DataFrame(
            np.random.randint(0, 2, size=(n_samples, n_features)),
            columns=[f"symptom_{i}" for i in range(n_features)]
        )
        
        # Create target with 4 causes
        causes = ["Cause_A", "Cause_B", "Cause_C", "Cause_D"]
        y = pd.Series(np.random.choice(causes, size=n_samples))
        
        return X, y
    
    @pytest.fixture
    def mock_docker_validation(self):
        """Mock successful Docker validation."""
        with patch.object(
            InSilicoVAModel, 
            '_InSilicoVAModel__init__',
            return_value=None
        ):
            yield
    
    def test_model_initialization(self, mock_config):
        """Test model creates with valid config."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            
            assert model.config == mock_config
            assert not model.is_fitted
            assert model.train_data is None
            assert model._unique_causes is None
            assert model._feature_columns is None
    
    def test_model_initialization_default_config(self):
        """Test model creates with default config."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel()
            
            assert isinstance(model.config, InSilicoVAConfig)
            assert not model.is_fitted
    
    def test_fit_with_valid_data(self, mock_config, sample_data):
        """Test model fits with valid training data."""
        X, y = sample_data
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            
            # Fit the model
            result = model.fit(X, y)
            
            # Check model state
            assert model.is_fitted
            assert model.train_data is not None
            assert len(model.train_data) == len(X)
            assert mock_config.cause_column in model.train_data.columns
            assert model._unique_causes == ["Cause_A", "Cause_B", "Cause_C", "Cause_D"]
            assert model._feature_columns == X.columns.tolist()
            assert result is model  # Check method chaining
    
    def test_fit_with_invalid_data(self, mock_config):
        """Test fit raises error with invalid data."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            
            # Empty data
            X_empty = pd.DataFrame()
            y_empty = pd.Series(dtype=str)
            
            with pytest.raises(ValueError, match="Invalid training data"):
                model.fit(X_empty, y_empty)
            
            # Mismatched shapes
            X_mismatch = pd.DataFrame(np.random.rand(10, 5))
            y_mismatch = pd.Series(["A"] * 5)
            
            with pytest.raises(ValueError, match="Invalid training data"):
                model.fit(X_mismatch, y_mismatch)
    
    def test_predict_without_fit(self, mock_config):
        """Test predict raises error when not fitted."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            X = pd.DataFrame(np.random.rand(5, 10))
            
            with pytest.raises(ValueError, match="Model must be fitted"):
                model.predict(X)
    
    def test_predict_proba_without_fit(self, mock_config):
        """Test predict_proba raises error when not fitted."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            X = pd.DataFrame(np.random.rand(5, 10))
            
            with pytest.raises(ValueError, match="Model must be fitted"):
                model.predict_proba(X)
    
    @patch("baseline.models.insilico_model.InSilicoVAModel._execute_insilico")
    def test_predict_proba_with_mock_docker(self, mock_execute, mock_config, sample_data):
        """Test predict_proba with mocked Docker execution."""
        X, y = sample_data
        X_test = X.iloc[:10]  # Use subset for testing
        
        # Create mock probability output
        mock_probs = pd.DataFrame({
            "Cause_A": ([0.4, 0.1, 0.3, 0.2] * 3)[:10],
            "Cause_B": ([0.3, 0.4, 0.2, 0.3] * 3)[:10],
            "Cause_C": ([0.2, 0.3, 0.3, 0.3] * 3)[:10],
            "Cause_D": ([0.1, 0.2, 0.2, 0.2] * 3)[:10],
        })
        mock_execute.return_value = mock_probs
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            model.fit(X, y)
            
            # Get predictions
            probs = model.predict_proba(X_test)
            
            # Check output shape and properties
            assert probs.shape == (10, 4)  # 10 samples, 4 classes
            assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
            assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities
    
    @patch("baseline.models.insilico_model.InSilicoVAModel._execute_insilico")
    def test_predict_with_mock_docker(self, mock_execute, mock_config, sample_data):
        """Test predict with mocked Docker execution."""
        X, y = sample_data
        X_test = X.iloc[:10]
        
        # Create mock probability output
        mock_probs = pd.DataFrame({
            "Cause_A": ([0.7, 0.1, 0.1, 0.2] * 3)[:10],
            "Cause_B": ([0.1, 0.7, 0.2, 0.3] * 3)[:10],
            "Cause_C": ([0.1, 0.1, 0.6, 0.3] * 3)[:10],
            "Cause_D": ([0.1, 0.1, 0.1, 0.2] * 3)[:10],
        })
        mock_execute.return_value = mock_probs
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            model.fit(X, y)
            
            # Get predictions
            predictions = model.predict(X_test)
            
            # Check output
            assert len(predictions) == 10
            assert all(pred in model._unique_causes for pred in predictions)
            
            # Check that predictions match highest probability
            # Note: When probabilities are tied, argmax returns the first index
            expected = ["Cause_A", "Cause_B", "Cause_C", "Cause_B"] + \
                      ["Cause_A", "Cause_B", "Cause_C", "Cause_B"] + \
                      ["Cause_A", "Cause_B"]
            assert list(predictions) == expected
    
    def test_csmf_accuracy_calculation(self, mock_config):
        """Test CSMF accuracy calculation matches expected formula."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            
            # Perfect prediction
            y_true = pd.Series(["A", "A", "B", "B", "C", "C"])
            y_pred = pd.Series(["A", "A", "B", "B", "C", "C"])
            
            accuracy = model.calculate_csmf_accuracy(y_true, y_pred)
            assert accuracy == 1.0
            
            # Some misclassification
            y_true = pd.Series(["A", "A", "B", "B", "C", "C"])
            y_pred = pd.Series(["A", "B", "B", "B", "C", "C"])
            
            accuracy = model.calculate_csmf_accuracy(y_true, y_pred)
            assert 0.0 <= accuracy <= 1.0
            assert accuracy < 1.0  # Not perfect due to misclassification
            
            # Calculate expected accuracy manually
            # True CSMF: A=2/6, B=2/6, C=2/6 = [0.333, 0.333, 0.333]
            # Pred CSMF: A=1/6, B=3/6, C=2/6 = [0.167, 0.500, 0.333]
            # Diff: |0.167-0.333| + |0.500-0.333| + |0.333-0.333| = 0.166 + 0.167 + 0 = 0.333
            # Denominator: 2 * (1 - 0.333) = 1.334
            # Accuracy: 1 - 0.333/1.334 = 0.75
            assert abs(accuracy - 0.75) < 0.01
    
    def test_generate_r_script(self, mock_config):
        """Test R script generation."""
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            r_script = model._generate_r_script()
            
            # Check key components are in script
            assert "library(openVA)" in r_script
            assert f"set.seed({mock_config.random_seed})" in r_script
            assert "codeVA(" in r_script
            assert f'model = "InSilicoVA"' in r_script
            assert f"Nsim = {mock_config.nsim}" in r_script
            assert f"jump.scale = {mock_config.jump_scale}" in r_script
            assert 'data.type = "customize"' in r_script
            assert "write.csv(results$indiv.prob" in r_script
    
    @patch("subprocess.run")
    def test_run_docker_command_success(self, mock_run, mock_config):
        """Test successful Docker command execution."""
        # Mock successful subprocess run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "R output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            
            # Create mock output file
            with patch("os.path.exists", return_value=True):
                with patch("pandas.read_csv") as mock_read:
                    mock_probs = pd.DataFrame({"Cause_A": [0.5], "Cause_B": [0.5]})
                    mock_read.return_value = mock_probs
                    
                    result = model._run_docker_command("/tmp/test")
                    
                    assert result is not None
                    assert isinstance(result, pd.DataFrame)
    
    @patch("subprocess.run")
    def test_run_docker_command_failure(self, mock_run, mock_config):
        """Test Docker command failure handling."""
        # Mock failed subprocess run
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_run.return_value = mock_result
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            result = model._run_docker_command("/tmp/test")
            
            assert result is None
    
    @patch("subprocess.run")
    def test_run_docker_command_timeout(self, mock_run, mock_config):
        """Test Docker command timeout handling."""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("docker", mock_config.docker_timeout)
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            result = model._run_docker_command("/tmp/test")
            
            assert result is None
    
    def test_format_probabilities(self, mock_config, sample_data):
        """Test probability formatting to numpy array."""
        X, y = sample_data
        
        with patch("baseline.models.model_validator.InSilicoVAValidator.validate_docker_availability") as mock_docker:
            mock_docker.return_value = Mock(is_valid=True, errors=[])
            
            model = InSilicoVAModel(mock_config)
            model.fit(X, y)
            
            # Create mock probability DataFrame
            probs_df = pd.DataFrame({
                "Cause_A": [0.4, 0.1],
                "Cause_B": [0.3, 0.4],
                "Cause_C": [0.2, 0.3],
                "Cause_D": [0.1, 0.2],
            })
            
            probs_array = model._format_probabilities(probs_df)
            
            # Check shape and properties
            assert probs_array.shape == (2, 4)
            assert np.allclose(probs_array.sum(axis=1), 1.0)
            
            # Check order matches unique causes
            np.testing.assert_array_almost_equal(probs_array[0], [0.4, 0.3, 0.2, 0.1])  # First row
    
    @pytest.mark.benchmark
    def test_benchmark_accuracy(self):
        """Test model achieves reasonable CSMF accuracy compared to published benchmarks."""
        # This would be an integration test with real data
        # Placeholder for benchmark testing
        pass
    
    @pytest.mark.benchmark  
    def test_table3_specific_scenarios(self):
        """Test specific Table 3 scenarios with generous tolerance."""
        # This would be an integration test with real data
        # Placeholder for Table 3 scenario testing
        pass