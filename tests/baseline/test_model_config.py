"""Tests for InSilicoVA model configuration."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.models.model_config import InSilicoVAConfig


class TestInSilicoVAConfig:
    """Test cases for InSilicoVA configuration."""
    
    def test_default_config_creation(self):
        """Test that default configuration is created successfully."""
        config = InSilicoVAConfig()
        
        # Check default values
        assert config.nsim == 10000
        assert config.jump_scale == 0.05
        assert config.auto_length is False
        assert config.convert_type == "fixed"
        assert config.prior_type == "quantile"
        assert config.docker_image == "insilicova-arm64:latest"
        assert config.docker_platform == "linux/arm64"
        assert config.docker_timeout == 3600
        assert config.cause_column == "gs_text34"
        assert config.phmrc_type == "adult"
        assert config.use_hce is True
        assert config.random_seed == 42
        assert config.output_dir == "temp"
        assert config.verbose is True
    
    def test_custom_config_creation(self):
        """Test creating configuration with custom values."""
        config = InSilicoVAConfig(
            nsim=5000,
            jump_scale=0.1,
            prior_type="default",
            docker_platform="linux/amd64",
            phmrc_type="child"
        )
        
        assert config.nsim == 5000
        assert config.jump_scale == 0.1
        assert config.prior_type == "default"
        assert config.docker_platform == "linux/amd64"
        assert config.phmrc_type == "child"
    
    def test_docker_platform_validation(self):
        """Test Docker platform validation."""
        # Valid platforms
        config1 = InSilicoVAConfig(docker_platform="linux/arm64")
        assert config1.docker_platform == "linux/arm64"
        
        config2 = InSilicoVAConfig(docker_platform="linux/amd64")
        assert config2.docker_platform == "linux/amd64"
        
        # Invalid platform
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(docker_platform="windows/amd64")
        
        # Check that error mentions the invalid platform
        error_str = str(exc_info.value)
        assert "docker_platform" in error_str
        assert "windows/amd64" in error_str or "literal_error" in error_str
    
    def test_jump_scale_validation(self):
        """Test jump scale parameter validation."""
        # Valid values
        config1 = InSilicoVAConfig(jump_scale=0.001)
        assert config1.jump_scale == 0.001
        
        config2 = InSilicoVAConfig(jump_scale=1.0)
        assert config2.jump_scale == 1.0
        
        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(jump_scale=0)
        
        assert "greater than 0" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(jump_scale=1.5)
        
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_nsim_validation(self):
        """Test nsim parameter validation with warnings."""
        # Valid normal range
        config1 = InSilicoVAConfig(nsim=10000)
        assert config1.nsim == 10000
        
        # Very low value (should log warning but still valid)
        config2 = InSilicoVAConfig(nsim=1000)
        assert config2.nsim == 1000
        
        # Very high value (should log warning but still valid)
        config3 = InSilicoVAConfig(nsim=150000)
        assert config3.nsim == 150000
        
        # Too low value
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(nsim=500)
        
        assert "greater than or equal to 1000" in str(exc_info.value)
    
    def test_docker_timeout_validation(self):
        """Test Docker timeout validation."""
        # Valid values
        config1 = InSilicoVAConfig(docker_timeout=60)
        assert config1.docker_timeout == 60
        
        config2 = InSilicoVAConfig(docker_timeout=7200)
        assert config2.docker_timeout == 7200
        
        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(docker_timeout=30)
        
        assert "greater than or equal to 60" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(docker_timeout=10000)
        
        assert "less than or equal to 7200" in str(exc_info.value)
    
    def test_docker_image_validation(self):
        """Test Docker image validation."""
        # Valid image names
        config1 = InSilicoVAConfig(docker_image="insilicova:latest")
        assert config1.docker_image == "insilicova:latest"
        
        # SHA256 format
        sha = "sha256:61df64731dec9b9e188b2b5a78000dd81e63caf78957257872b9ac1ba947efa4"
        config2 = InSilicoVAConfig(docker_image=sha)
        assert config2.docker_image == sha
        
        # Invalid SHA256 format
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(docker_image="sha256:invalid")
        
        assert "Invalid SHA256 hash format" in str(exc_info.value)
        
        # Empty image
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(docker_image="")
        
        assert "Docker image cannot be empty" in str(exc_info.value)
    
    def test_literal_field_validation(self):
        """Test literal field validations."""
        # Valid convert_type
        config1 = InSilicoVAConfig(convert_type="fixed")
        assert config1.convert_type == "fixed"
        
        config2 = InSilicoVAConfig(convert_type="adaptive")
        assert config2.convert_type == "adaptive"
        
        # Invalid convert_type
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(convert_type="dynamic")
        
        # Valid prior_type
        config3 = InSilicoVAConfig(prior_type="quantile")
        assert config3.prior_type == "quantile"
        
        config4 = InSilicoVAConfig(prior_type="default")
        assert config4.prior_type == "default"
        
        # Invalid prior_type
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(prior_type="uniform")
        
        # Valid phmrc_type
        config5 = InSilicoVAConfig(phmrc_type="adult")
        assert config5.phmrc_type == "adult"
        
        config6 = InSilicoVAConfig(phmrc_type="child")
        assert config6.phmrc_type == "child"
        
        config7 = InSilicoVAConfig(phmrc_type="neonate")
        assert config7.phmrc_type == "neonate"
        
        # Invalid phmrc_type
        with pytest.raises(ValidationError) as exc_info:
            InSilicoVAConfig(phmrc_type="infant")
    
    def test_get_r_script_params(self):
        """Test R script parameter generation."""
        config = InSilicoVAConfig(
            nsim=5000,
            jump_scale=0.1,
            auto_length=True,
            convert_type="adaptive",
            cause_column="cause",
            phmrc_type="child",
            random_seed=123,
            use_hce=False
        )
        
        params = config.get_r_script_params()
        
        assert params["nsim"] == 5000
        assert params["jump_scale"] == 0.1
        assert params["auto_length"] == "TRUE"  # Converted to R boolean
        assert params["convert_type"] == "adaptive"
        assert params["cause_column"] == "cause"
        assert params["phmrc_type"] == "child"
        assert params["random_seed"] == 123
        assert params["use_hce"] is False
    
    def test_config_field_descriptions(self):
        """Test that all fields have descriptions."""
        # This ensures documentation is maintained
        config = InSilicoVAConfig()
        
        # Check key fields have descriptions
        assert InSilicoVAConfig.model_fields["nsim"].description is not None
        assert InSilicoVAConfig.model_fields["docker_image"].description is not None
        assert InSilicoVAConfig.model_fields["cause_column"].description is not None