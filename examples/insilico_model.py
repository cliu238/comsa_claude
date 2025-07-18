"""InSilicoVA model implementation."""

import logging
import os
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils.data_utils import extract_xy_str_from_subset


class InSilicoVAModel:
    """InSilicoVA model implementation using Docker container."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the InSilicoVA model.

        Args:
            model_params: Dictionary of model parameters including:
                - random_seed: Random seed for reproducibility (default: 42)
                - docker_image: Name of InsilicoVA Docker image to use (default: "insilicova-arm64")
                - docker_platform: Docker platform to use (default: "linux/arm64"; alternative: "linux/amd64")
                - output_dir: Directory to store temporary files (default: "temp")
                - cause_col_name: Name of the cause column (default: "va34")
                - phmrc_data_type: PHMRC data type (default: "adult")
                - jump_scale: Jump scale parameter (default: 0.05)
                - nsim: Number of simulations (default: 10000)
                - convert_type: Convert type parameter (default: "fixed")
                - auto_length: Auto length parameter (default: False)
        """
        default_params = {
            "random_seed": 42,
            "output_dir": "temp",
            "cause_col_name": "va34",
            "phmrc_data_type": "adult",
            "jump_scale": 0.05,
            "nsim": 10000,
            "convert_type": "fixed",
            "auto_length": False,
            "docker_image": "insilicova-arm64",
            "docker_platform": "linux/arm64",
        }
        if model_params:
            default_params.update(model_params)
        self.model_params = default_params

        # Create output directory if it doesn't exist
        os.makedirs(self.model_params["output_dir"], exist_ok=True)

    def predict(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
    ) -> np.ndarray:
        """Run InSilicoVA experiment and get predictions.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features to predict on

        Returns:
            Array of predicted causes
        """
        # Combine training data
        train_data = pd.concat([X_train, y_train], axis=1)

        # Create test data by combining features with empty labels
        test_data = X_test.copy()
        test_data[self.model_params["cause_col_name"]] = ""

        # Run InSilicoVA experiment
        probs = run_insilico_experiment(
            train_data=train_data,
            test_data=test_data,
            **self.model_params,
        )

        if probs is None:
            raise RuntimeError("InSilicoVA experiment failed")

        # Convert probabilities to predictions by taking the most likely cause

        return probs


def run_insilico_experiment(
    train_data,
    test_data,
    output_dir="temp",
    docker_image="insilicova-arm64",
    docker_platform="linux/arm64",
    phmrc_data_type="adult",
    cause_col_name="va34",
    nsim=10000,
    jump_scale=0.05,
    convert_type="fixed",
    auto_length=False,
    random_seed=42,
    **kwargs,
):
    """
    Run InSilicoVA experiment using Docker

    Args:
        train_data: Training data subset
        test_data: Test data subset
        output_dir: Directory to store outputs
        random_seed: Random seed for InSilicoVA model (default: 42)

    Returns:
        pd.DataFrame: Probability predictions or None if experiment fails
    """
    # Create temporary files for train and test data
    train_file = os.path.join(output_dir, "train_data.csv")
    test_file = os.path.join(output_dir, "test_data.csv")

    # Convert Subset objects directly to DataFrames using the simplified function
    train_df = extract_xy_str_from_subset(train_data)
    test_df = extract_xy_str_from_subset(test_data)

    # Check for empty dataframes which can happen with specific site splits
    if train_df.empty or test_df.empty:
        logging.warning(
            f"Skipping InSilicoVA run due to empty train ({train_df.empty}) or test ({test_df.empty}) data."
        )
        return None

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Modify the R script to use our specific files and set random seed
    # * Using repr() below adds quotes to strings
    # * R accepts TRUE, T, FALSE, F as boolean literals
    r_script = f"""
# Script to run codeVA with specific parameters and data

# Load necessary library
library(openVA)
library(dplyr)

# --- Configuration ---
# Data Files
train_data_file <- "/data/train_data.csv"
test_data_file  <- "/data/test_data.csv"

# codeVA Parameters
cause_col_name  <- {repr(cause_col_name)}
phmrc_data_type <- {repr(phmrc_data_type)}
jump_scale_val  <- {jump_scale}
convert_type_val<- {repr(convert_type)}
nsim_val        <- {nsim}
auto_length_val <- {str(auto_length).upper()}
model_val       <- "InSilicoVA"
data_type_val   <- "customize"
random_seed_val <- {random_seed}

# Set random seed for reproducibility
set.seed(random_seed_val)

# Read Data
train_data <- read.csv(train_data_file)
test_data <- read.csv(test_data_file)

# Add ID column at the beginning of both datasets
train_data <- train_data %>%
    mutate(ID = row_number()) %>%
    select(ID, everything())

test_data <- test_data %>%
    mutate(ID = row_number()) %>%
    select(ID, everything())

# Check for NA values and replace with empty strings
train_data <- train_data %>% 
    mutate(across(everything(), ~ifelse(is.na(.), "", as.character(.))))

test_data <- test_data %>% 
    mutate(across(everything(), ~ifelse(is.na(.), "", as.character(.))))

# Run codeVA
insilico_results <- codeVA(
    data = test_data,
    data.type = data_type_val,
    model = model_val,
    data.train = train_data,
    causes.train = cause_col_name,
    phmrc.type = phmrc_data_type,
    jump.scale = jump_scale_val,
    convert.type = convert_type_val,
    Nsim = nsim_val,
    auto.length = auto_length_val,
    seed = random_seed_val
)

# Save results
if (!is.null(insilico_results) && !is.null(insilico_results$indiv.prob)) {{
    write.csv(insilico_results$indiv.prob, "/data/insilico_probs.csv")
}}
"""

    # Save R script
    r_script_path = os.path.join(output_dir, "run_insilico.R")
    with open(r_script_path, "w") as f:
        f.write(r_script)

    # Run Docker command
    output_dir = os.path.abspath(output_dir)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "-v",
        f"{output_dir}:/data",
        # Make sure R image has dplyr installed already; old images do not have it
        docker_image,
        "R",
        "-f",
        "/data/run_insilico.R",
    ]

    try:
        # Run the Docker command and capture output
        result = subprocess.run(cmd, check=True, text=True)
        logging.info("Docker command output:")
        logging.info(result.stdout)
        if result.stderr:
            logging.warning("Docker command stderr:")
            logging.warning(result.stderr)

        # Check if the output file exists
        probs_file = os.path.join(output_dir, "insilico_probs.csv")
        if not os.path.exists(probs_file):
            logging.error(
                f"Output file {probs_file} not found after running InSilicoVA"
            )
            return None

        # Read and return results
        probs = pd.read_csv(probs_file, index_col=0)
        return probs

    except subprocess.CalledProcessError as e:
        logging.error(f"Docker command failed with error: {e}")
        logging.error(f"Command output: {e.output}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error running InSilicoVA: {e}")
        return None
