"""InSilicoVA model implementation with sklearn-like interface.

This module provides the main InSilicoVA model implementation that executes
the InSilicoVA algorithm via Docker containers and provides predictions
for VA cause-of-death classification.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from baseline.models.model_config import InSilicoVAConfig
from baseline.models.model_validator import InSilicoVAValidator

logger = logging.getLogger(__name__)


class InSilicoVAModel(BaseEstimator):
    """InSilicoVA model with sklearn-like interface.
    
    This model implements the InSilicoVA algorithm for verbal autopsy
    cause-of-death prediction using Docker containers to handle the
    complex R/Java dependencies.
    """
    
    def __init__(self, config: Optional[InSilicoVAConfig] = None):
        """Initialize the InSilicoVA model.
        
        Args:
            config: Model configuration. If None, uses default configuration.
        """
        self.config = config or InSilicoVAConfig()
        self.logger = logging.getLogger(__name__)
        self.validator = InSilicoVAValidator(self.config)
        
        # Model state
        self.is_fitted = False
        self.train_data: Optional[pd.DataFrame] = None
        self._unique_causes: Optional[list[str]] = None
        self._feature_columns: Optional[list[str]] = None
        
        # Validate Docker availability on initialization
        docker_result = self.validator.validate_docker_availability()
        if not docker_result.is_valid:
            self.logger.warning(
                f"Docker validation failed: {docker_result.errors}. "
                "Model may not work properly."
            )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "InSilicoVAModel":
        """Fit the model using training data.
        
        Args:
            X: Feature DataFrame with VA symptoms
            y: Target Series with cause of death labels
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If training data is invalid
        """
        self.logger.info(f"Fitting InSilicoVA model with {len(X)} samples")
        
        # Validate training data
        validation_result = self.validator.validate_training_data(X, y)
        if not validation_result.is_valid:
            raise ValueError(
                f"Invalid training data: {'; '.join(validation_result.errors)}"
            )
        
        # Log warnings if any
        for warning in validation_result.warnings:
            self.logger.warning(warning)
        
        # Store training data for Docker execution
        self.train_data = X.copy()
        self.train_data[self.config.cause_column] = y.astype(str)
        self._unique_causes = sorted(y.astype(str).unique())
        self._feature_columns = X.columns.tolist()
        self.is_fitted = True
        
        self.logger.info(
            f"Model fitted with {len(self._unique_causes)} unique causes: "
            f"{self._unique_causes[:5]}{'...' if len(self._unique_causes) > 5 else ''}"
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using fitted model.
        
        Args:
            X: Feature DataFrame for prediction
            
        Returns:
            Array of predicted cause labels
            
        Raises:
            ValueError: If model is not fitted or prediction fails
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get probability predictions
        proba = self.predict_proba(X)
        
        # Convert to class predictions (highest probability)
        if self._unique_causes is None:
            raise RuntimeError("Model not fitted - unique causes not available")
            
        predicted_indices = np.argmax(proba, axis=1)
        predictions = np.array([self._unique_causes[idx] for idx in predicted_indices])
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions using fitted model.
        
        Args:
            X: Feature DataFrame for prediction
            
        Returns:
            Array of shape (n_samples, n_classes) with probability predictions
            
        Raises:
            ValueError: If model is not fitted or prediction fails
            RuntimeError: If Docker execution fails
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate prediction data
        validation_result = self.validator.validate_prediction_data(X)
        if not validation_result.is_valid:
            raise ValueError(
                f"Invalid prediction data: {'; '.join(validation_result.errors)}"
            )
        
        # Validate column compatibility
        if self._feature_columns is None:
            raise RuntimeError("Model not fitted - feature columns not available")
            
        col_validation = self.validator.validate_column_compatibility(
            self._feature_columns, X.columns.tolist()
        )
        if not col_validation.is_valid:
            raise ValueError(
                f"Column mismatch: {'; '.join(col_validation.errors)}"
            )
        
        # Ensure columns are in same order as training
        X_aligned = X[self._feature_columns].copy()
        
        self.logger.info(f"Running InSilicoVA prediction for {len(X)} samples")
        
        # Execute InSilicoVA via Docker
        with tempfile.TemporaryDirectory() as temp_dir:
            probs_df = self._execute_insilico(X_aligned, str(temp_dir))
        
        # Convert to numpy array with consistent ordering
        probs_array = self._format_probabilities(probs_df)
        
        return probs_array
    
    def calculate_csmf_accuracy(
        self, 
        y_true: pd.Series, 
        y_pred: pd.Series
    ) -> float:
        """Calculate CSMF accuracy following openVA implementation.
        
        CSMF (Cause-Specific Mortality Fraction) accuracy measures how well
        the predicted distribution of causes matches the true distribution.
        
        Formula: 1 - sum(|pred_fraction - true_fraction|) / (2 * (1 - min(true_fraction)))
        
        Args:
            y_true: True cause labels
            y_pred: Predicted cause labels
            
        Returns:
            CSMF accuracy score between 0 and 1
        """
        # Ensure both series have the same data type (convert to string for consistency)
        y_true_str = y_true.astype(str)
        y_pred_str = y_pred.astype(str)
        
        # Calculate true and predicted CSMFs
        csmf_true = y_true_str.value_counts(normalize=True).sort_index()
        csmf_pred = y_pred_str.value_counts(normalize=True).sort_index()
        
        # Align indices to handle missing causes
        all_causes = sorted(set(csmf_true.index) | set(csmf_pred.index))
        csmf_true_aligned = csmf_true.reindex(all_causes, fill_value=0)
        csmf_pred_aligned = csmf_pred.reindex(all_causes, fill_value=0)
        
        # Calculate CSMF accuracy
        numerator = np.sum(np.abs(csmf_pred_aligned - csmf_true_aligned))
        denominator = 2 * (1 - np.min(csmf_true_aligned))
        
        accuracy = 1 - (numerator / denominator)
        
        self.logger.info(f"CSMF accuracy: {accuracy:.3f}")
        
        return float(accuracy)
    
    def _execute_insilico(
        self, 
        X: pd.DataFrame, 
        temp_dir: str
    ) -> pd.DataFrame:
        """Execute InSilicoVA via Docker container.
        
        Args:
            X: Feature DataFrame for prediction
            temp_dir: Temporary directory for file I/O
            
        Returns:
            DataFrame with probability predictions
            
        Raises:
            RuntimeError: If Docker execution fails
        """
        # Save data files
        train_file = os.path.join(temp_dir, "train_data.csv")
        test_file = os.path.join(temp_dir, "test_data.csv")
        
        # Save training data with proper handling of empty strings
        if self.train_data is None:
            raise RuntimeError("Training data is not available")
        
        # Replace NaN with empty strings to preserve OpenVA encoding
        train_data_clean = self.train_data.fillna("")
        train_data_clean.to_csv(train_file, index=False)
        
        # Create test data with empty cause column
        test_data = X.copy()
        test_data[self.config.cause_column] = ""
        # Replace NaN with empty strings to preserve OpenVA encoding
        test_data_clean = test_data.fillna("")
        test_data_clean.to_csv(test_file, index=False)
        
        # Generate R script
        r_script = self._generate_r_script()
        r_script_path = os.path.join(temp_dir, "run_insilico.R")
        with open(r_script_path, "w") as f:
            f.write(r_script)
        
        # Execute Docker
        try:
            probs_df = self._run_docker_command(temp_dir)
            if probs_df is None and self.config.use_fallback_dockerfile:
                self.logger.info("Primary Docker image failed, trying fallback Dockerfile")
                probs_df = self._try_fallback_docker(temp_dir)
        except Exception as e:
            self.logger.error(f"Docker execution failed: {str(e)}")
            raise RuntimeError(f"InSilicoVA Docker execution failed: {str(e)}")
        
        if probs_df is None:
            raise RuntimeError("Failed to get predictions from InSilicoVA")
        
        return probs_df
    
    def _run_docker_command(self, temp_dir: str) -> Optional[pd.DataFrame]:
        """Run Docker command for InSilicoVA execution.
        
        Args:
            temp_dir: Directory containing input/output files
            
        Returns:
            DataFrame with probabilities or None if failed
        """
        cmd = [
            "docker", "run", "--rm",
            "--platform", self.config.docker_platform,
            "-v", f"{os.path.abspath(temp_dir)}:/data",
            self.config.docker_image,
            "R", "-f", "/data/run_insilico.R"
        ]
        
        self.logger.debug(f"Running Docker command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.docker_timeout
            )
            
            if self.config.verbose:
                if result.stdout:
                    self.logger.info(f"Docker stdout:\n{result.stdout}")
                if result.stderr:
                    self.logger.warning(f"Docker stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                self.logger.error(
                    f"Docker command failed with return code {result.returncode}"
                )
                return None
            
            # Read results
            probs_file = os.path.join(temp_dir, "insilico_probs.csv")
            if not os.path.exists(probs_file):
                self.logger.error(f"Output file not found: {probs_file}")
                return None
            
            probs_df = pd.read_csv(probs_file, index_col=0)
            self.logger.info(
                f"Successfully loaded predictions: {probs_df.shape}"
            )
            
            return probs_df
            
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"Docker execution timed out after {self.config.docker_timeout}s"
            )
            return None
        except Exception as e:
            self.logger.error(f"Docker execution error: {str(e)}")
            return None
    
    def _try_fallback_docker(self, temp_dir: str) -> Optional[pd.DataFrame]:
        """Try building and running from local Dockerfile.
        
        Args:
            temp_dir: Directory containing input/output files
            
        Returns:
            DataFrame with probabilities or None if failed
        """
        # Check if Dockerfile exists
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            self.logger.error("Dockerfile not found for fallback")
            return None
        
        # Build image
        build_tag = "insilicova-local:latest"
        build_cmd = ["docker", "build", "-f", "Dockerfile", "-t", build_tag, "."]
        
        self.logger.info("Building InSilicoVA image from Dockerfile")
        try:
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for build
            )
            
            if build_result.returncode != 0:
                self.logger.error(
                    f"Docker build failed: {build_result.stderr}"
                )
                return None
            
            # Update config to use local image and retry
            original_image = self.config.docker_image
            self.config.docker_image = build_tag
            
            result = self._run_docker_command(temp_dir)
            
            # Restore original image
            self.config.docker_image = original_image
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback Docker build/run failed: {str(e)}")
            return None
    
    def _generate_r_script(self) -> str:
        """Generate R script for InSilicoVA execution.
        
        Returns:
            R script as string
        """
        params = self.config.get_r_script_params()
        
        r_script = f"""
# InSilicoVA R Script
library(openVA)

# Set random seed for reproducibility
set.seed({params['random_seed']})

# Read data
train_data <- read.csv("/data/train_data.csv", stringsAsFactors = FALSE)
test_data <- read.csv("/data/test_data.csv", stringsAsFactors = FALSE)

# Add ID columns using row numbers
train_data <- cbind(ID = seq_len(nrow(train_data)), train_data)
test_data <- cbind(ID = seq_len(nrow(test_data)), test_data)

# Convert NA to empty strings
train_data[is.na(train_data)] <- ""
test_data[is.na(test_data)] <- ""

# Convert all columns to character
train_data[] <- lapply(train_data, as.character)
test_data[] <- lapply(test_data, as.character)

# Run InSilicoVA
tryCatch({{
    results <- codeVA(
        data = test_data,
        data.type = "customize",
        model = "InSilicoVA",
        data.train = train_data,
        causes.train = "{params['cause_column']}",
        phmrc.type = "{params['phmrc_type']}",
        jump.scale = {params['jump_scale']},
        convert.type = "{params['convert_type']}",
        Nsim = {params['nsim']},
        auto.length = {params['auto_length']},
        seed = {params['random_seed']}
    )
    
    # Save individual probabilities
    if (!is.null(results) && !is.null(results$indiv.prob)) {{
        write.csv(results$indiv.prob, "/data/insilico_probs.csv")
        cat("InSilicoVA completed successfully\\n")
    }} else {{
        stop("InSilicoVA returned NULL results")
    }}
}}, error = function(e) {{
    cat("Error in InSilicoVA execution:\\n")
    cat(as.character(e), "\\n")
    quit(status = 1)
}})
"""
        
        return r_script
    
    def _format_probabilities(self, probs_df: pd.DataFrame) -> np.ndarray:
        """Format probability DataFrame to numpy array with consistent ordering.
        
        Args:
            probs_df: DataFrame with probability predictions
            
        Returns:
            Numpy array of shape (n_samples, n_classes)
        """
        # Ensure all training causes are represented
        if self._unique_causes is None:
            raise RuntimeError("Model not fitted - unique causes not available")
        
        probs_array: np.ndarray = np.zeros((len(probs_df), len(self._unique_causes)))
        
        for i, cause in enumerate(self._unique_causes):
            if cause in probs_df.columns:
                probs_array[:, i] = probs_df[cause].values
        
        # Normalize if needed (should already sum to 1)
        row_sums = probs_array.sum(axis=1, keepdims=True)
        probs_array = np.divide(
            probs_array, 
            row_sums, 
            out=probs_array, 
            where=row_sums > 0
        )
        
        return probs_array