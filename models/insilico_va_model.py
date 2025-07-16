"""InSilicoVA model implementation for replicating Table 3 results."""

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseModel


class InSilicoVAModel(BaseModel):
    """InSilicoVA model implementation using Docker container for Table 3 replication."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the InSilicoVA model.

        Args:
            model_params: Dictionary of model parameters including:
                - random_seed: Random seed for reproducibility (default: 42)
                - docker_image: Name of InSilicoVA Docker image (default: "insilicova-arm64:latest")
                - docker_platform: Docker platform (default: "linux/arm64")
                - nsim: Number of MCMC simulations (default: 10000)
                - jump_scale: Jump scale parameter (default: 0.05)
                - auto_length: Auto length parameter (default: False)
                - convert_type: Convert type parameter (default: "fixed")
                - phmrc_type: PHMRC data type (default: "adult")
                - cause_col: Cause column name (default: "gs_text34")
                - use_hce: Use Historical Cause-Specific Elements (default: True)
        """
        default_params = {
            "random_seed": 42,
            "docker_image": "insilicova-arm64:latest",
            "docker_platform": "linux/arm64",
            "nsim": 10000,
            "jump_scale": 0.05,
            "auto_length": False,
            "convert_type": "fixed",
            "phmrc_type": "adult",
            "cause_col": "gs_text34",
            "use_hce": True,
        }

        if model_params:
            default_params.update(model_params)

        super().__init__(default_params)

        # Store training data for Docker execution
        self.train_data = None
        self.is_fitted = False

    def _initialize_model(self) -> None:
        """Initialize the model (no sklearn model needed for InSilicoVA)."""
        # InSilicoVA doesn't use sklearn, so we don't initialize self.model
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model by storing training data.

        Args:
            X: Training features
            y: Training labels (cause of death)
        """
        # Combine features and labels for InSilicoVA
        self.train_data = X.copy()
        self.train_data[self.model_params["cause_col"]] = y
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using InSilicoVA.

        Args:
            X: Features to predict on

        Returns:
            Array of predicted causes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Get probability predictions
        probs = self.predict_proba(X)

        # Convert probabilities to class predictions
        # Get the class with highest probability for each sample
        predicted_indices = np.argmax(probs, axis=1)

        # Get unique causes from training data to map indices back to causes
        unique_causes = sorted(self.train_data[self.model_params["cause_col"]].unique())
        predictions = [unique_causes[idx] for idx in predicted_indices]

        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions using InSilicoVA.

        Args:
            X: Features to predict on

        Returns:
            Array of probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Run InSilicoVA via Docker
        probs_df = self._run_insilico_va(X)

        if probs_df is None:
            raise RuntimeError("InSilicoVA prediction failed")

        # Convert probabilities DataFrame to numpy array
        # Ensure consistent ordering of causes
        unique_causes = sorted(self.train_data[self.model_params["cause_col"]].unique())

        # Reorder columns to match unique_causes order
        probs_array = np.zeros((len(X), len(unique_causes)))
        for i, cause in enumerate(unique_causes):
            if cause in probs_df.columns:
                probs_array[:, i] = probs_df[cause].values

        return probs_array

    def _run_insilico_va(self, test_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Run InSilicoVA using Docker container.

        Args:
            test_data: Test data for prediction

        Returns:
            DataFrame with probability predictions or None if failed
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Prepare data files
                train_file = os.path.join(temp_dir, "train_data.csv")
                test_file = os.path.join(temp_dir, "test_data.csv")

                # Save training data
                self.train_data.to_csv(train_file, index=False)

                # Create test data with empty cause column
                test_data_copy = test_data.copy()
                test_data_copy[self.model_params["cause_col"]] = ""
                test_data_copy.to_csv(test_file, index=False)

                # Create R script
                r_script = self._create_r_script()
                r_script_path = os.path.join(temp_dir, "run_insilico.R")
                with open(r_script_path, "w") as f:
                    f.write(r_script)

                # Run Docker command with dplyr auto-install (from PR #5)
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--platform",
                    self.model_params["docker_platform"],
                    "-v",
                    f"{temp_dir}:/data",
                    self.model_params["docker_image"],
                    "R",
                    "-e",
                    'if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos="http://cran.rstudio.com/"); source("/data/run_insilico.R")',
                ]

                logging.info(f"Running InSilicoVA with command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=3600
                )

                if result.returncode != 0:
                    logging.error(
                        f"InSilicoVA failed with return code {result.returncode}"
                    )
                    logging.error(f"stdout: {result.stdout}")
                    logging.error(f"stderr: {result.stderr}")
                    return None

                # Read results
                probs_file = os.path.join(temp_dir, "insilico_probs.csv")
                if not os.path.exists(probs_file):
                    logging.error(f"Output file {probs_file} not found")
                    return None

                probs_df = pd.read_csv(probs_file, index_col=0)
                return probs_df

            except Exception as e:
                logging.error(f"Error running InSilicoVA: {e}")
                return None

    def _create_r_script(self) -> str:
        """Create R script for InSilicoVA execution.

        Returns:
            R script as string
        """
        use_hce = self.model_params["use_hce"]

        if use_hce:
            # Use PHMRC data type with HCE
            r_script = f"""
# InSilicoVA R Script with HCE
library(openVA)

# Set random seed
set.seed({self.model_params["random_seed"]})

# Read data
train_data <- read.csv("/data/train_data.csv", stringsAsFactors = FALSE)
test_data <- read.csv("/data/test_data.csv", stringsAsFactors = FALSE)

# Clean data - replace NA with empty strings using base R
train_data[is.na(train_data)] <- ""
test_data[is.na(test_data)] <- ""

# Convert all columns to character (equivalent to dplyr::across)
train_data[] <- lapply(train_data, as.character)
test_data[] <- lapply(test_data, as.character)

# Run InSilicoVA with HCE
insilico_results <- codeVA(
    data = test_data,
    data.type = "PHMRC",
    model = "InSilicoVA",
    data.train = train_data,
    causes.train = "{self.model_params["cause_col"]}",
    phmrc.type = "{self.model_params["phmrc_type"]}",
    jump.scale = {self.model_params["jump_scale"]},
    convert.type = "{self.model_params["convert_type"]}",
    Nsim = {self.model_params["nsim"]},
    auto.length = {str(self.model_params["auto_length"]).upper()},
    seed = {self.model_params["random_seed"]}
)

# Save results
if (!is.null(insilico_results) && !is.null(insilico_results$indiv.prob)) {{
    write.csv(insilico_results$indiv.prob, "/data/insilico_probs.csv")
    cat("InSilicoVA completed successfully\\n")
}} else {{
    cat("InSilicoVA failed - no results generated\\n")
    quit(status = 1)
}}
"""
        else:
            # Use customized data type without HCE
            r_script = f"""
# InSilicoVA R Script without HCE
library(openVA)
library(dplyr)

# Set random seed
set.seed({self.model_params["random_seed"]})

# Read data
train_data <- read.csv("/data/train_data.csv")
test_data <- read.csv("/data/test_data.csv")

# Convert data using ConvertData.phmrc
binary <- ConvertData.phmrc(train_data, test_data, 
                           phmrc.type = "{self.model_params["phmrc_type"]}", 
                           cause = "{self.model_params["cause_col"]}")

# Remove reduced variables (a1_01_1 to a1_01_14) as in paper
reduced <- paste0("a1_01_", 1:14)
binary$output <- binary$output[, -which(colnames(binary$output) %in% reduced)]
binary$output.test <- binary$output.test[, -which(colnames(binary$output.test) %in% reduced)]

# Run InSilicoVA
insilico_results <- codeVA(
    data = binary$output.test,
    data.type = "customize",
    model = "InSilicoVA",
    data.train = binary$output,
    causes.train = "Cause",
    jump.scale = {self.model_params["jump_scale"]},
    convert.type = "{self.model_params["convert_type"]}",
    Nsim = {self.model_params["nsim"]},
    auto.length = {str(self.model_params["auto_length"]).upper()},
    seed = {self.model_params["random_seed"]}
)

# Save results
if (!is.null(insilico_results) && !is.null(insilico_results$indiv.prob)) {{
    write.csv(insilico_results$indiv.prob, "/data/insilico_probs.csv")
    cat("InSilicoVA completed successfully\\n")
}} else {{
    cat("InSilicoVA failed - no results generated\\n")
    quit(status = 1)
}}
"""

        return r_script

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()

    def set_params(self, **params: Any) -> None:
        """Set model parameters."""
        self.model_params.update(params)
