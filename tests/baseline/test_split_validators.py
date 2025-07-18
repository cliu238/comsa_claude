"""Unit tests for split validation utilities."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.config.data_config import DataConfig
from baseline.utils.class_validator import ClassValidator, ValidationResult
from baseline.utils.split_validator import SplitValidator


class TestClassValidator:
    """Test cases for ClassValidator class."""
    
    @pytest.fixture
    def class_validator(self):
        """Create a class validator for testing."""
        return ClassValidator(min_samples_per_class=5)
    
    @pytest.fixture
    def balanced_data(self):
        """Create balanced class data for testing."""
        return pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced class data for testing."""
        return pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])  # Class 4 has only 1 sample
    
    @pytest.fixture
    def single_instance_data(self):
        """Create data with single instance classes."""
        return pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5])  # Classes 3, 4, 5 have 1 sample each
    
    def test_initialization(self):
        """Test ClassValidator initialization."""
        validator = ClassValidator(min_samples_per_class=10)
        assert validator.min_samples_per_class == 10
    
    def test_validate_balanced_distribution(self, class_validator, balanced_data):
        """Test validation of balanced class distribution."""
        result = class_validator.validate_class_distribution(balanced_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.class_distribution == {"1": 6, "2": 6, "3": 6}
    
    def test_validate_imbalanced_distribution(self, class_validator, imbalanced_data):
        """Test validation of imbalanced class distribution."""
        result = class_validator.validate_class_distribution(imbalanced_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False  # Should fail due to single instance class
        assert len(result.errors) > 0
        assert "single instance" in result.errors[0].lower()
        assert "4" in result.errors[0]  # Class 4 should be mentioned
    
    def test_validate_single_instance_classes(self, class_validator, single_instance_data):
        """Test validation with multiple single instance classes."""
        result = class_validator.validate_class_distribution(single_instance_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "single instance" in result.errors[0].lower()
        # Should mention classes 3, 4, 5
        for cls in ["3", "4", "5"]:
            assert cls in result.errors[0]
    
    def test_validate_small_classes_warning(self, class_validator):
        """Test validation with small but not single instance classes."""
        # Classes with 2-4 samples (below threshold of 5)
        data = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4])
        
        result = class_validator.validate_class_distribution(data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True  # Valid since no single instance classes
        assert len(result.warnings) > 0
        assert "fewer than 5 samples" in result.warnings[0]
    
    def test_get_stratifiable_classes(self, class_validator, imbalanced_data):
        """Test getting classes suitable for stratification."""
        filtered_data = class_validator.get_stratifiable_classes(imbalanced_data, min_samples=2)
        
        # Should exclude class 4 (single instance)
        unique_classes = set(filtered_data.unique())
        assert 4 not in unique_classes
        assert 1 in unique_classes
        assert 2 in unique_classes
        assert 3 in unique_classes
    
    def test_get_stratifiable_classes_higher_threshold(self, class_validator, imbalanced_data):
        """Test getting classes with higher sample threshold."""
        filtered_data = class_validator.get_stratifiable_classes(imbalanced_data, min_samples=5)
        
        # Should only include class 1 (has 8 samples)
        unique_classes = set(filtered_data.unique())
        assert unique_classes == {1}
    
    def test_suggest_handling_strategy_balanced(self, class_validator, balanced_data):
        """Test strategy suggestion for balanced data."""
        strategy = class_validator.suggest_handling_strategy(balanced_data)
        assert strategy == "stratified"
    
    def test_suggest_handling_strategy_single_instance(self, class_validator, single_instance_data):
        """Test strategy suggestion for single instance classes."""
        strategy = class_validator.suggest_handling_strategy(single_instance_data)
        assert strategy == "non_stratified"
    
    def test_suggest_handling_strategy_small_classes(self, class_validator):
        """Test strategy suggestion for small classes."""
        # Small classes but no single instances
        data = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        
        strategy = class_validator.suggest_handling_strategy(data)
        assert strategy == "exclude_small_classes"
    
    def test_empty_data(self, class_validator):
        """Test handling of empty data."""
        empty_data = pd.Series([], dtype=int)
        
        result = class_validator.validate_class_distribution(empty_data)
        assert isinstance(result, ValidationResult)
        assert result.class_distribution == {}
    
    def test_different_min_samples_threshold(self):
        """Test validator with different minimum samples threshold."""
        validator = ClassValidator(min_samples_per_class=10)
        data = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])  # 6 samples each
        
        result = validator.validate_class_distribution(data)
        
        # Should warn about classes with < 10 samples
        assert len(result.warnings) > 0
        assert "fewer than 10 samples" in result.warnings[0]


class TestSplitValidator:
    """Test cases for SplitValidator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock(spec=DataConfig)
        config.split_strategy = "train_test"
        config.test_size = 0.3
        config.site_column = "site"
        config.label_column = "va34"
        config.train_sites = None
        config.test_sites = None
        return config
    
    @pytest.fixture
    def valid_data(self):
        """Create valid data for testing."""
        return pd.DataFrame({
            "site": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "va34": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            "symptom1": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "symptom2": ["Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N"]
        })
    
    @pytest.fixture
    def split_validator(self, mock_config):
        """Create a split validator for testing."""
        return SplitValidator(mock_config)
    
    def test_initialization(self, mock_config):
        """Test SplitValidator initialization."""
        validator = SplitValidator(mock_config)
        assert validator.config == mock_config
    
    def test_validate_data_train_test_success(self, split_validator, valid_data):
        """Test successful validation for train_test strategy."""
        # Should not raise any exception
        split_validator.validate_data_for_splitting(valid_data)
    
    def test_validate_data_missing_label_column(self, split_validator, valid_data):
        """Test validation failure for missing label column."""
        data_no_label = valid_data.drop(columns=["va34"])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            split_validator.validate_data_for_splitting(data_no_label)
    
    def test_validate_data_missing_site_column(self, split_validator, valid_data):
        """Test validation failure for missing site column in site-based strategies."""
        split_validator.config.split_strategy = "cross_site"
        data_no_site = valid_data.drop(columns=["site"])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            split_validator.validate_data_for_splitting(data_no_site)
    
    def test_validate_data_insufficient_samples(self, split_validator):
        """Test validation failure for insufficient data."""
        insufficient_data = pd.DataFrame({
            "site": ["A", "A"],
            "va34": [1, 2],
            "symptom1": ["Yes", "No"]
        })
        
        with pytest.raises(ValueError, match="Insufficient data for splitting"):
            split_validator.validate_data_for_splitting(insufficient_data)
    
    def test_validate_data_single_site_cross_site(self, split_validator, valid_data):
        """Test validation failure for single site in cross-site strategy."""
        split_validator.config.split_strategy = "cross_site"
        # Create larger single site data to pass minimum sample validation
        single_site_data = pd.DataFrame({
            "site": ["A"] * 15,
            "va34": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            "symptom1": ["Yes", "No"] * 7 + ["Yes"],
            "symptom2": ["Y", "N"] * 7 + ["Y"]
        })
        
        with pytest.raises(ValueError, match="Need at least 2 sites"):
            split_validator.validate_data_for_splitting(single_site_data)
    
    def test_validate_cross_site_specific_sites(self, split_validator, valid_data):
        """Test validation with specific train/test sites."""
        split_validator.config.split_strategy = "cross_site"
        split_validator.config.train_sites = ["A", "B"]
        split_validator.config.test_sites = ["C"]
        
        # Should not raise any exception
        split_validator.validate_data_for_splitting(valid_data)
    
    def test_validate_cross_site_invalid_train_sites(self, split_validator, valid_data):
        """Test validation failure for invalid train sites."""
        split_validator.config.split_strategy = "cross_site"
        split_validator.config.train_sites = ["X", "Y"]  # Sites that don't exist
        split_validator.config.test_sites = ["A"]
        
        with pytest.raises(ValueError, match="Train sites not found"):
            split_validator.validate_data_for_splitting(valid_data)
    
    def test_validate_cross_site_invalid_test_sites(self, split_validator, valid_data):
        """Test validation failure for invalid test sites."""
        split_validator.config.split_strategy = "cross_site"
        split_validator.config.train_sites = ["A"]
        split_validator.config.test_sites = ["X", "Y"]  # Sites that don't exist
        
        with pytest.raises(ValueError, match="Test sites not found"):
            split_validator.validate_data_for_splitting(valid_data)
    
    def test_validate_cross_site_overlapping_sites(self, split_validator, valid_data):
        """Test validation failure for overlapping train/test sites."""
        split_validator.config.split_strategy = "cross_site"
        split_validator.config.train_sites = ["A", "B"]
        split_validator.config.test_sites = ["B", "C"]  # B overlaps
        
        with pytest.raises(ValueError, match="Train and test sites overlap"):
            split_validator.validate_data_for_splitting(valid_data)
    
    def test_get_available_sites(self, split_validator, valid_data):
        """Test getting available sites from data."""
        sites = split_validator.get_available_sites(valid_data)
        
        assert isinstance(sites, list)
        assert set(sites) == {"A", "B", "C"}
        assert sites == sorted(sites)  # Should be sorted
    
    def test_get_available_sites_no_site_column(self, split_validator, valid_data):
        """Test getting available sites when site column is missing."""
        data_no_site = valid_data.drop(columns=["site"])
        sites = split_validator.get_available_sites(data_no_site)
        
        assert sites == []
    
    def test_get_available_sites_with_nulls(self, split_validator, valid_data):
        """Test getting available sites with null values."""
        data_with_nulls = valid_data.copy()
        data_with_nulls.loc[0, "site"] = None
        
        sites = split_validator.get_available_sites(data_with_nulls)
        
        # Should exclude null values
        assert None not in sites
        assert set(sites) == {"A", "B", "C"}
    
    def test_suggest_site_split(self, split_validator, valid_data):
        """Test site split suggestion."""
        train_sites, test_sites = split_validator.suggest_site_split(valid_data)
        
        assert isinstance(train_sites, list)
        assert isinstance(test_sites, list)
        assert len(train_sites) > 0
        assert len(test_sites) > 0
        assert set(train_sites).isdisjoint(set(test_sites))  # No overlap
        assert set(train_sites + test_sites) == {"A", "B", "C"}
    
    def test_suggest_site_split_insufficient_sites(self, split_validator, valid_data):
        """Test site split suggestion with insufficient sites."""
        single_site_data = valid_data[valid_data["site"] == "A"]
        train_sites, test_sites = split_validator.suggest_site_split(single_site_data)
        
        assert train_sites == []
        assert test_sites == []
    
    def test_suggest_site_split_different_test_size(self, split_validator, valid_data):
        """Test site split suggestion with different test sizes."""
        split_validator.config.test_size = 0.5
        train_sites, test_sites = split_validator.suggest_site_split(valid_data)
        
        # With 3 sites and 0.5 test size, should have roughly equal split
        assert len(test_sites) >= 1
        assert len(train_sites) >= 1
    
    def test_validate_site_sample_sizes(self, split_validator):
        """Test validation of site sample sizes."""
        # Create data with small site but enough total samples
        data_small_site = pd.DataFrame({
            "site": ["A"] * 8 + ["B"] * 8 + ["C"] * 1,  # Site C has only 1 sample, 17 total
            "va34": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            "symptom1": ["Yes", "No"] * 8 + ["Yes"],
            "symptom2": ["Y", "N"] * 8 + ["Y"]
        })
        
        split_validator.config.split_strategy = "cross_site"
        
        # Should not raise error but may log warnings
        split_validator.validate_data_for_splitting(data_small_site)
    
    def test_stratified_site_validation(self, split_validator, valid_data):
        """Test validation for stratified site strategy."""
        split_validator.config.split_strategy = "stratified_site"
        
        # Should not raise any exception
        split_validator.validate_data_for_splitting(valid_data)
    
    def test_validate_required_columns_site_based(self, split_validator, valid_data):
        """Test that site-based strategies require site column."""
        split_validator.config.split_strategy = "stratified_site"
        
        # Should require both label and site columns
        required_columns = ["va34", "site"]
        
        for col in required_columns:
            data_missing_col = valid_data.drop(columns=[col])
            with pytest.raises(ValueError, match="Missing required columns"):
                split_validator.validate_data_for_splitting(data_missing_col)
    
    def test_validate_sufficient_data_different_test_sizes(self, split_validator):
        """Test data sufficiency validation with different test sizes."""
        # Create minimal data
        minimal_data = pd.DataFrame({
            "site": ["A"] * 15,
            "va34": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 3, 3, 3, 3, 3],
            "symptom1": ["Yes", "No"] * 7 + ["Yes"],
            "symptom2": ["Y", "N"] * 7 + ["Y"]
        })
        
        # Should pass with small test size
        split_validator.config.test_size = 0.1
        split_validator.validate_data_for_splitting(minimal_data)
        
        # Should fail with large test size - use very small data
        very_small_data = pd.DataFrame({
            "site": ["A"] * 5,
            "va34": [1, 1, 2, 2, 1],
            "symptom1": ["Yes", "No", "Yes", "No", "Yes"]
        })
        split_validator.config.test_size = 0.9
        with pytest.raises(ValueError, match="Insufficient data for splitting"):
            split_validator.validate_data_for_splitting(very_small_data)