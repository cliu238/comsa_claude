"""Configuration for InSilicoVA model.

This module provides configuration management for the InSilicoVA model,
including Docker settings, model parameters, and execution options.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class InSilicoVAConfig(BaseModel):
    """Configuration for InSilicoVA model parameters.
    
    This configuration class manages all settings for the InSilicoVA model,
    including Docker configuration, model hyperparameters, and data processing options.
    """
    
    # Core InSilicoVA parameters
    nsim: int = Field(
        default=10000, 
        ge=1000, 
        description="Number of MCMC simulations"
    )
    jump_scale: float = Field(
        default=0.05, 
        gt=0, 
        le=1.0,
        description="Jump scale parameter for MCMC"
    )
    auto_length: bool = Field(
        default=False, 
        description="Auto length parameter for symptom duration"
    )
    convert_type: Literal["fixed", "adaptive"] = Field(
        default="fixed", 
        description="Convert type parameter for data transformation"
    )
    prior_type: Literal["quantile", "default"] = Field(
        default="quantile", 
        description="Prior type: 'quantile' or 'default'"
    )
    
    # Docker configuration
    docker_image: str = Field(
        default="insilicova-arm64:latest", 
        description="Docker image name or SHA"
    )
    docker_platform: Literal["linux/arm64", "linux/amd64"] = Field(
        default="linux/arm64", 
        description="Docker platform specification"
    )
    docker_timeout: int = Field(
        default=3600, 
        ge=60, 
        le=7200,
        description="Docker execution timeout in seconds"
    )
    use_fallback_dockerfile: bool = Field(
        default=False,
        description="Use local Dockerfile if primary image fails"
    )
    
    # Data configuration
    cause_column: str = Field(
        default="va34", 
        description="Column name containing cause of death labels (numeric codes for InSilicoVA)"
    )
    phmrc_type: Literal["adult", "child", "neonate"] = Field(
        default="adult", 
        description="PHMRC data type"
    )
    use_hce: bool = Field(
        default=True, 
        description="Use Historical Cause-Specific Elements"
    )
    
    # Execution parameters
    random_seed: int = Field(
        default=42, 
        ge=0, 
        description="Random seed for reproducibility"
    )
    output_dir: str = Field(
        default="temp", 
        description="Directory for temporary files during execution"
    )
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging of Docker execution"
    )
    
    @field_validator("docker_platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate Docker platform is supported.
        
        Args:
            v: Platform string to validate
            
        Returns:
            Validated platform string
            
        Raises:
            ValueError: If platform is not supported
        """
        valid_platforms = ["linux/arm64", "linux/amd64"]
        if v not in valid_platforms:
            raise ValueError(f"Platform must be one of {valid_platforms}, got {v}")
        return v
    
    @field_validator("jump_scale")
    @classmethod
    def validate_jump_scale(cls, v: float) -> float:
        """Validate jump scale is in reasonable range.
        
        Args:
            v: Jump scale value
            
        Returns:
            Validated jump scale
            
        Raises:
            ValueError: If jump scale is out of range
        """
        if not 0.001 <= v <= 1.0:
            raise ValueError(f"Jump scale should be between 0.001 and 1.0, got {v}")
        return v
    
    @field_validator("nsim")
    @classmethod
    def validate_nsim(cls, v: int) -> int:
        """Validate number of simulations is reasonable.
        
        Args:
            v: Number of simulations
            
        Returns:
            Validated nsim value
            
        Raises:
            ValueError: If nsim is too small or too large
        """
        if v < 1000:
            logger.warning(f"nsim={v} is very low, results may be unreliable")
        elif v > 100000:
            logger.warning(f"nsim={v} is very high, execution will be slow")
        return v
    
    @field_validator("docker_image")
    @classmethod
    def validate_docker_image(cls, v: str) -> str:
        """Validate Docker image specification.
        
        Args:
            v: Docker image name or SHA
            
        Returns:
            Validated Docker image
            
        Raises:
            ValueError: If image specification is invalid
        """
        if not v:
            raise ValueError("Docker image cannot be empty")
        
        # Check if it's a SHA256 hash
        if v.startswith("sha256:") and len(v) != 71:  # sha256: + 64 chars
            raise ValueError(f"Invalid SHA256 hash format: {v}")
            
        return v
    
    def get_r_script_params(self) -> dict:
        """Get parameters formatted for R script generation.
        
        Returns:
            Dictionary of parameters for R script
        """
        return {
            "nsim": self.nsim,
            "jump_scale": self.jump_scale,
            "auto_length": str(self.auto_length).upper(),
            "convert_type": self.convert_type,
            "cause_column": self.cause_column,
            "phmrc_type": self.phmrc_type,
            "random_seed": self.random_seed,
            "use_hce": self.use_hce,
        }