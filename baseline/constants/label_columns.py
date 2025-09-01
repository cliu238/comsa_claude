"""Centralized definition of label-equivalent columns for VA data.

This module provides a single source of truth for all columns that contain
cause-of-death information or other label-equivalent data that must be
excluded from features to prevent data leakage.

CRITICAL: This list must be kept in sync across all data processing functions
to ensure consistent feature spaces for all models.
"""

from typing import List, Optional

# Complete list of all label-equivalent columns that must be excluded from features
# NOTE: "cause" is NOT included here because it's the unified target column that 
# gets created from other label columns (va34, cod5, etc.) and needs to be available
LABEL_EQUIVALENT_COLUMNS = [
    # Site and metadata
    "site",  # Site information (used for stratification, not features)
    "module",  # Module type
    "newid",  # ID column
    
    # 34-cause classification columns
    "gs_code34",  # Gold standard code for 34 causes
    "gs_text34",  # Gold standard text for 34 causes
    "va34",  # VA 34-cause classification
    
    # 46-cause classification columns
    "gs_code46",  # Gold standard code for 46 causes
    "gs_text46",  # Gold standard text for 46 causes
    "va46",  # VA 46-cause classification
    
    # 55-cause classification columns
    "gs_code55",  # Gold standard code for 55 causes
    "gs_text55",  # Gold standard text for 55 causes
    "va55",  # VA 55-cause classification
    
    # Comorbidity and additional information
    "gs_comorbid1",  # Primary comorbidity information
    "gs_comorbid2",  # Secondary comorbidity information
    "gs_level",  # Gold standard level
    
    # Broad cause groupings
    "cod5",  # 5-cause grouping (broad categories)
]


def get_label_columns_to_drop(
    columns: List[str],
    keep_target: Optional[str] = None,
) -> List[str]:
    """Get the list of label columns to drop from a given set of columns.
    
    This function returns only the label columns that actually exist in the
    provided column list, optionally keeping a specific target column.
    
    Args:
        columns: List of column names from the dataset
        keep_target: Optional target column to keep (e.g., 'cause', 'va34', 'cod5')
                    This column will be excluded from the drop list.
    
    Returns:
        List of label column names that should be dropped from features.
        
    Example:
        >>> df_columns = ['site', 'cause', 'feature1', 'feature2', 'va34']
        >>> get_label_columns_to_drop(df_columns, keep_target='cause')
        ['site', 'va34']  # 'cause' is kept as the target
    """
    # Get columns that exist in the dataset
    columns_to_drop = [col for col in LABEL_EQUIVALENT_COLUMNS if col in columns]
    
    # Remove the target column if specified
    if keep_target and keep_target in columns_to_drop:
        columns_to_drop.remove(keep_target)
    
    return columns_to_drop


def validate_no_label_leakage(
    feature_columns: List[str],
    raise_on_leakage: bool = True,
) -> List[str]:
    """Validate that no label-equivalent columns are present in features.
    
    Args:
        feature_columns: List of feature column names to validate
        raise_on_leakage: If True, raise ValueError on detecting leakage.
                         If False, return list of leaked columns.
    
    Returns:
        List of label columns found in features (empty if no leakage).
        
    Raises:
        ValueError: If raise_on_leakage=True and label columns are found.
    """
    leaked_columns = [col for col in feature_columns if col in LABEL_EQUIVALENT_COLUMNS]
    
    if leaked_columns and raise_on_leakage:
        raise ValueError(
            f"Data leakage detected! The following label columns are present "
            f"in features: {leaked_columns}. These must be excluded to prevent "
            f"information leakage."
        )
    
    return leaked_columns