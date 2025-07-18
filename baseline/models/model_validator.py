"""Validation utilities for InSilicoVA model.

This module provides validation utilities for ensuring data compatibility
and Docker availability for the InSilicoVA model.
"""

import logging
import shutil
import subprocess
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel

from baseline.models.model_config import InSilicoVAConfig

logger = logging.getLogger(__name__)


class ModelValidationResult(BaseModel):
    """Result of model validation.
    
    Contains validation status, warnings, errors, and metadata
    about the validation process.
    """
    
    is_valid: bool
    warnings: List[str] = []
    errors: List[str] = []
    metadata: Dict[str, Any] = {}


class InSilicoVAValidator:
    """Validator for InSilicoVA model requirements.
    
    This class provides methods to validate Docker availability,
    data format compatibility, and other requirements for running
    the InSilicoVA model.
    """
    
    def __init__(self, config: InSilicoVAConfig):
        """Initialize the validator.
        
        Args:
            config: InSilicoVA model configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_docker_availability(self) -> ModelValidationResult:
        """Validate that Docker is available and configured properly.
        
        Returns:
            ModelValidationResult with Docker validation status
        """
        result = ModelValidationResult(is_valid=True)
        
        # Check if Docker is installed
        if not shutil.which("docker"):
            result.is_valid = False
            result.errors.append("Docker is not installed or not in PATH")
            return result
        
        # Check if Docker daemon is running
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            result.metadata["docker_available"] = True
            self.logger.info("Docker is available and running")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            result.is_valid = False
            result.errors.append(f"Docker daemon is not running: {str(e)}")
            result.metadata["docker_available"] = False
            return result
        
        # Check if the configured image exists
        try:
            image_check = subprocess.run(
                ["docker", "image", "inspect", self.config.docker_image],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if image_check.returncode == 0:
                result.metadata["image_exists"] = True
                self.logger.info(f"Docker image {self.config.docker_image} found")
            else:
                result.warnings.append(
                    f"Docker image {self.config.docker_image} not found locally. "
                    "Will attempt to pull when needed."
                )
                result.metadata["image_exists"] = False
                
                # If primary image doesn't exist and fallback is disabled, it's an error
                if not self.config.use_fallback_dockerfile:
                    result.warnings.append(
                        "Consider enabling use_fallback_dockerfile in config "
                        "to build from local Dockerfile if image pull fails"
                    )
                    
        except subprocess.TimeoutExpired:
            result.warnings.append("Docker image inspection timed out")
            result.metadata["image_exists"] = "unknown"
        
        return result
    
    def validate_training_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> ModelValidationResult:
        """Validate training data for InSilicoVA compatibility.
        
        Args:
            X: Feature DataFrame
            y: Target Series with cause labels
            
        Returns:
            ModelValidationResult with data validation status
        """
        result = ModelValidationResult(is_valid=True)
        
        # Check for empty data
        if X.empty or y.empty:
            result.is_valid = False
            result.errors.append("Training data is empty")
            return result
        
        # Check shapes match
        if len(X) != len(y):
            result.is_valid = False
            result.errors.append(
                f"Feature and target shapes don't match: {len(X)} vs {len(y)}"
            )
            return result
        
        # Check for minimum samples
        min_samples = 10  # InSilicoVA needs reasonable sample size
        if len(X) < min_samples:
            result.is_valid = False
            result.errors.append(
                f"Insufficient training samples: {len(X)} < {min_samples}"
            )
            return result
        
        # Check for minimum unique causes
        unique_causes = y.nunique()
        min_causes = 2
        if unique_causes < min_causes:
            result.is_valid = False
            result.errors.append(
                f"Insufficient unique causes: {unique_causes} < {min_causes}"
            )
            return result
        
        # Check for NA values in target
        if y.isna().any():
            result.is_valid = False
            result.errors.append("Target variable contains NA values")
            return result
        
        # Check for non-string causes
        if not all(isinstance(val, (str, int)) for val in y.unique()):
            result.warnings.append(
                "Target contains non-string/int values. Will convert to string."
            )
        
        # Check for very imbalanced data
        cause_counts = y.value_counts()
        min_count = cause_counts.min()
        max_count = cause_counts.max()
        
        if min_count < 3:
            result.warnings.append(
                f"Some causes have very few samples (min: {min_count}). "
                "Results may be unreliable."
            )
        
        if max_count / min_count > 100:
            result.warnings.append(
                f"Highly imbalanced data detected (ratio: {max_count/min_count:.1f}). "
                "Consider stratified sampling."
            )
        
        # Store metadata
        result.metadata.update({
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_unique_causes": unique_causes,
            "cause_distribution": cause_counts.to_dict(),
            "min_cause_count": int(min_count),
            "max_cause_count": int(max_count),
        })
        
        return result
    
    def validate_prediction_data(self, X: pd.DataFrame) -> ModelValidationResult:
        """Validate prediction data for InSilicoVA compatibility.
        
        Args:
            X: Feature DataFrame for prediction
            
        Returns:
            ModelValidationResult with data validation status
        """
        result = ModelValidationResult(is_valid=True)
        
        # Check for empty data
        if X.empty:
            result.is_valid = False
            result.errors.append("Prediction data is empty")
            return result
        
        # Check for minimum samples (InSilicoVA may have issues with single samples)
        if len(X) < 1:
            result.is_valid = False
            result.errors.append("No samples provided for prediction")
            return result
        
        # Check for all-NA columns
        all_na_cols = X.columns[X.isna().all()].tolist()
        if all_na_cols:
            result.warnings.append(
                f"Columns with all NA values: {all_na_cols}. "
                "These will be handled by InSilicoVA."
            )
        
        # Store metadata
        result.metadata.update({
            "n_samples": len(X),
            "n_features": X.shape[1],
            "all_na_columns": all_na_cols,
        })
        
        return result
    
    def validate_column_compatibility(
        self, 
        train_columns: List[str], 
        test_columns: List[str]
    ) -> ModelValidationResult:
        """Validate that train and test data have compatible columns.
        
        Args:
            train_columns: Column names from training data
            test_columns: Column names from test data
            
        Returns:
            ModelValidationResult with column compatibility status
        """
        result = ModelValidationResult(is_valid=True)
        
        train_set = set(train_columns)
        test_set = set(test_columns)
        
        # Check for missing columns in test data
        missing_in_test = train_set - test_set
        if missing_in_test:
            result.is_valid = False
            result.errors.append(
                f"Columns missing in test data: {sorted(missing_in_test)}"
            )
        
        # Check for extra columns in test data
        extra_in_test = test_set - train_set
        if extra_in_test:
            result.warnings.append(
                f"Extra columns in test data will be ignored: {sorted(extra_in_test)}"
            )
        
        # Store metadata
        result.metadata.update({
            "train_columns": len(train_columns),
            "test_columns": len(test_columns),
            "missing_in_test": sorted(missing_in_test) if missing_in_test else [],
            "extra_in_test": sorted(extra_in_test) if extra_in_test else [],
        })
        
        return result