"""Class distribution validation utilities for VA data splitting.

This module provides utilities for validating class distributions and handling
small classes in the context of train/test splitting operations.
"""

import logging
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of class distribution validation.
    
    Attributes:
        is_valid: Whether the validation passed
        warnings: List of warning messages
        errors: List of error messages
        class_distribution: Dictionary mapping class labels to sample counts
    """
    
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    class_distribution: Dict[str, int]


class ClassValidator:
    """Utility for validating class distributions and handling small classes.
    
    This class provides methods to validate class distributions in datasets
    and identify potential issues with small classes that might cause problems
    during stratified splitting operations.
    """
    
    def __init__(self, min_samples_per_class: int = 5):
        """Initialize the class validator.
        
        Args:
            min_samples_per_class: Minimum number of samples required per class
        """
        self.min_samples_per_class = min_samples_per_class
        self.logger = logging.getLogger(__name__)
    
    def validate_class_distribution(self, y: pd.Series) -> ValidationResult:
        """Validate class distribution in target variable.
        
        Args:
            y: Target variable series
            
        Returns:
            ValidationResult with validation status and details
        """
        # Count class occurrences
        class_counts = y.value_counts()
        
        # Convert to regular Python types for JSON serialization
        class_distribution = {str(k): int(v) for k, v in class_counts.items()}
        
        # Initialize result containers
        warnings = []
        errors = []
        
        # Check for small classes
        small_classes = class_counts[class_counts < self.min_samples_per_class]
        
        if len(small_classes) > 0:
            # Handle single-instance classes (critical for stratified splitting)
            single_instance = class_counts[class_counts == 1]
            
            if len(single_instance) > 0:
                single_classes = single_instance.index.tolist()
                errors.append(
                    f"Classes with single instance found: {single_classes}. "
                    f"These will cause stratified splitting to fail."
                )
                self.logger.error(f"Single instance classes detected: {single_classes}")
            
            # Handle other small classes
            other_small = small_classes[small_classes > 1]
            if len(other_small) > 0:
                small_class_info = {str(k): int(v) for k, v in other_small.items()}
                warnings.append(
                    f"Classes with fewer than {self.min_samples_per_class} samples: "
                    f"{small_class_info}. Consider using non-stratified splitting."
                )
                self.logger.warning(f"Small classes detected: {small_class_info}")
        
        # Log overall distribution statistics
        self.logger.info(f"Class distribution: {len(class_counts)} classes, "
                        f"total samples: {len(y)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            class_distribution=class_distribution
        )
    
    def get_stratifiable_classes(self, y: pd.Series, min_samples: int = 2) -> pd.Series:
        """Get classes that have sufficient samples for stratified splitting.
        
        Args:
            y: Target variable series
            min_samples: Minimum samples required per class for stratification
            
        Returns:
            Series with only classes that have sufficient samples
        """
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        
        # Filter to only include valid classes
        filtered_y = y[y.isin(valid_classes)]
        
        self.logger.info(f"Filtered from {len(class_counts)} to {len(valid_classes)} "
                        f"classes with >= {min_samples} samples")
        
        return filtered_y
    
    def suggest_handling_strategy(self, y: pd.Series) -> str:
        """Suggest appropriate handling strategy for small classes.
        
        Args:
            y: Target variable series
            
        Returns:
            Suggested handling strategy
        """
        validation_result = self.validate_class_distribution(y)
        
        if not validation_result.is_valid:
            # Single instance classes - cannot use stratified splitting
            return "non_stratified"
        
        if validation_result.warnings:
            # Some small classes but no single instances
            return "exclude_small_classes"
        
        # All classes have sufficient samples
        return "stratified"