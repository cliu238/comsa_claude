"""Data validation stage implementation."""

import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

import va_data.phmrc_plugins  # 确保 transform 注册
from va_data.phmrc_data import PHMRCData


def validate_data(
    data_path: str,
    label_column: str,
    drop_columns: Optional[List[str]] = None,
    is_va_data: bool = False,
    openva_encoding: bool = False,
    target_format: str = "numeric",
    table3_compatible: bool = False,
) -> pd.DataFrame:
    """
    Validate and load input data.

    Args:
        data_path: Path to the input data file
        label_column: Name of the label column
        drop_columns: Optional list of columns to drop
        is_va_data: Whether the data is verbal autopsy data
        openva_encoding: Whether to use OpenVA encoding for InSilicoVA
        target_format: "numeric" for va34 or "text" for gs_text34
        table3_compatible: Whether to use Table 3 compatible preprocessing

    Returns:
        Validated DataFrame

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If validation fails
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Handle VA data validation
    if is_va_data:
        try:
            va_data = PHMRCData(data_path)
            df = va_data.validate()
            
            # Apply different processing based on mode
            if table3_compatible:
                # Use Table 3 compatible preprocessing (similar to replicate_table3_results.py)
                df = _prepare_table3_compatible_data(df, target_format)
            else:
                # Apply OpenVA transformation for pipeline use
                df = va_data.xform("openva")
                
                if openva_encoding:
                    repldict = {1: "Y", 0: "", 2: "."}
                    cols = df.columns.difference(["site", "va34", "cod5"])
                    df = df.astype({c: object for c in cols})
                    with pd.option_context("future.no_silent_downcasting", True):
                        df = df.replace({c: repldict for c in cols})
                else:
                    # Convert categorical data to numeric for ML models
                    df = _convert_categorical_to_numeric(df, label_column)
                        
            if drop_columns:
                df = df.drop(columns=drop_columns, errors="ignore")
            return df
        except Exception as e:
            raise ValueError(f"Failed to validate VA data: {str(e)}")

    # Regular data validation
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")

    # Ensure label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")

    # Drop specified columns if any
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    return df


def get_data_statistics(data: pd.DataFrame) -> Dict:
    """
    Get statistics about the dataset.

    Args:
        data: Input DataFrame

    Returns:
        Dictionary containing dataset statistics
    """
    return {
        "n_samples": len(data),
        "n_features": len(data.columns),
        "column_names": list(data.columns),
        "feature_types": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "class_distribution": data.iloc[:, -1].value_counts().to_dict(),
    }


def check_data_quality(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Check data quality metrics.

    Args:
        data: Input DataFrame

    Returns:
        Dictionary containing data quality checks
    """
    # Check for duplicates in any column
    has_duplicates = False
    for col in data.columns:
        if data[col].duplicated().any():
            has_duplicates = True
            break

    return {
        "has_missing_values": data.isnull().any().any(),
        "has_duplicates": has_duplicates,
        "has_infinite_values": data.isin([float("inf"), float("-inf")]).any().any(),
        "has_constant_features": (data.nunique() == 1).any(),
    }


def _prepare_table3_compatible_data(df: pd.DataFrame, target_format: str) -> pd.DataFrame:
    """
    Prepare PHMRC data in Table 3 compatible format.
    
    This function replicates the data preparation approach used in 
    replicate_table3_results.py for consistency with the published results.
    
    Args:
        df: Validated PHMRC DataFrame
        target_format: "numeric" for va34 or "text" for gs_text34
    
    Returns:
        Processed DataFrame with Table 3 compatible format
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Add gs_text34 column by mapping va34 values to cause names
    if "gs_text34" not in data.columns and "va34" in data.columns:
        # Import cause mapping from PHMRCData
        cod34_mapping = PHMRCData.cod34
        data["gs_text34"] = data["va34"].map(cod34_mapping)
    
    # Define columns to exclude (similar to replicate_table3_results.py)
    exclude_cols = [
        "site", "module", "gs_code34", "gs_text34", "va34", 
        "gs_code46", "gs_text46", "va46", "gs_code55", "gs_text55", "va55",
        "gs_comorbid1", "gs_comorbid2", "gs_level", "newid", "cod5"
    ]
    
    # Select target column based on format
    if target_format == "text":
        target_col = "gs_text34"
    else:
        target_col = "va34"
    
    # Keep site, target, and all feature columns
    keep_cols = ["site", target_col]
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    keep_cols.extend(feature_cols)
    
    # Filter to keep only relevant columns
    data = data[keep_cols]
    
    # Convert string categorical data to numeric format for ML models
    data = _convert_categorical_to_numeric(data, target_col)
    
    return data


def _convert_categorical_to_numeric(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert string categorical data to numeric format for ML compatibility.
    
    Args:
        df: DataFrame with potentially string categorical data
        target_col: Name of the target column to preserve
    
    Returns:
        DataFrame with numeric features
    """
    data = df.copy()
    
    # Get feature columns (exclude site and target)
    feature_cols = [col for col in data.columns if col not in ["site", target_col]]
    
    # Convert string responses to numeric
    for col in feature_cols:
        if data[col].dtype == 'object':
            # Handle common VA data patterns
            data[col] = data[col].astype(str)  # Ensure string type
            
            # Common Yes/No mappings
            with pd.option_context("future.no_silent_downcasting", True):
                data[col] = data[col].replace({
                    'Yes': 1, 'No': 0, 'Y': 1, 'N': 0,
                    'yes': 1, 'no': 0, 'y': 1, 'n': 0,
                    '1': 1, '0': 0, 'True': 1, 'False': 0,
                    'true': 1, 'false': 0
                })
                
                # Handle missing values and empty strings
                data[col] = data[col].replace({
                    '': 0, 'nan': 0, 'NaN': 0, 'None': 0,
                    'NA': 0, 'na': 0, '.': 0, 'null': 0
                })
            
            # Convert to numeric, coercing any remaining strings to NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Fill remaining NaN values with 0 (conservative approach)
            data[col] = data[col].fillna(0)
            
            # Ensure integer type for binary features
            if data[col].nunique() <= 10:  # Likely categorical
                data[col] = data[col].astype(int)
    
    # Handle missing values in target column
    if target_col in data.columns:
        if data[target_col].dtype == 'object' and target_col != "gs_text34":
            # Only convert numeric targets, keep text targets as strings
            data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        
        # Remove rows with missing target values
        data = data.dropna(subset=[target_col])
    
    return data
