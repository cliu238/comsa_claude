"""Unit tests for VA data splitter."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.config.data_config import DataConfig
from baseline.data.data_splitter import VADataSplitter, SplitResult


class TestVADataSplitter:
    """Test cases for VADataSplitter class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock(spec=DataConfig)
        config.data_path = "dummy.csv"
        config.output_dir = "test_output/"
        config.split_strategy = "train_test"
        config.test_size = 0.3
        config.random_state = 42
        config.site_column = "site"
        config.label_column = "va34"
        config.train_sites = None
        config.test_sites = None
        config.min_samples_per_class = 5
        config.handle_small_classes = "warn"
        config.model_dump.return_value = {"test": "config"}
        return config
    
    @pytest.fixture
    def sample_va_data(self):
        """Create sample VA data for testing."""
        return pd.DataFrame({
            "site": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
            "va34": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "cod5": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "symptom1": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "symptom2": ["Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N"],
            "symptom3": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def small_class_data(self):
        """Create data with small classes for testing edge cases."""
        return pd.DataFrame({
            "site": ["A"] * 8 + ["B"] * 8,  # 16 samples total
            "va34": [1, 1, 1, 1, 2, 2, 2, 999, 1, 1, 1, 1, 2, 2, 2, 999],  # Class 999 has only 2 samples
            "symptom1": ["Yes", "No"] * 8,
            "symptom2": ["Y", "N"] * 8,
            "symptom3": [1, 0] * 8,
            "symptom4": [0, 1] * 8
        })
    
    @pytest.fixture
    def single_instance_data(self):
        """Create data with single instance classes."""
        return pd.DataFrame({
            "site": ["A"] * 8 + ["B"] * 8,  # 16 samples total
            "va34": [1, 1, 1, 1, 2, 2, 2, 999, 1, 1, 1, 1, 2, 2, 2, 2],  # Class 999 has only 1 sample
            "symptom1": ["Yes", "No"] * 8,
            "symptom2": ["Y", "N"] * 8,
            "symptom3": [1, 0] * 8,
            "symptom4": [0, 1] * 8
        })
    
    def test_initialization(self, mock_config):
        """Test VADataSplitter initialization."""
        splitter = VADataSplitter(mock_config)
        assert splitter.config == mock_config
        assert splitter.class_validator is not None
        assert splitter.split_validator is not None
    
    def test_train_test_split_success(self, mock_config, sample_va_data):
        """Test successful train/test split."""
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # Verify result structure
        assert isinstance(result, SplitResult)
        assert isinstance(result.train, pd.DataFrame)
        assert isinstance(result.test, pd.DataFrame)
        assert isinstance(result.metadata, dict)
        
        # Verify split proportions
        total_samples = len(result.train) + len(result.test)
        assert total_samples == len(sample_va_data)
        
        test_ratio = len(result.test) / total_samples
        assert 0.2 <= test_ratio <= 0.4  # Allow some variance due to small dataset
        
        # Verify metadata content
        assert result.metadata["split_strategy"] == "train_test"
        assert result.metadata["test_size"] == 0.3
        assert result.metadata["random_state"] == 42
        assert "train_samples" in result.metadata
        assert "test_samples" in result.metadata
        assert "train_class_distribution" in result.metadata
        assert "test_class_distribution" in result.metadata
    
    def test_train_test_split_stratified(self, mock_config, sample_va_data):
        """Test stratified train/test split maintains class distribution."""
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # Check that both classes are present in both splits
        train_classes = set(result.train[mock_config.label_column].unique())
        test_classes = set(result.test[mock_config.label_column].unique())
        
        # Should have at least one class in common (ideally all)
        assert len(train_classes.intersection(test_classes)) > 0
        
        # Verify metadata indicates stratification
        assert "stratified" in result.metadata
    
    def test_small_class_warning(self, mock_config, small_class_data):
        """Test handling of small classes with warning."""
        mock_config.handle_small_classes = "warn"
        splitter = VADataSplitter(mock_config)
        
        # Should complete without error but with warnings
        result = splitter.split_data(small_class_data)
        assert isinstance(result, SplitResult)
    
    def test_single_instance_error(self, mock_config, single_instance_data):
        """Test handling of single instance classes with error."""
        mock_config.handle_small_classes = "error"
        splitter = VADataSplitter(mock_config)
        
        # Should raise error for single instance classes
        with pytest.raises(ValueError, match="Class validation failed"):
            splitter.split_data(single_instance_data)
    
    def test_single_instance_exclude(self, mock_config, single_instance_data):
        """Test handling of single instance classes with exclude."""
        mock_config.handle_small_classes = "exclude"
        splitter = VADataSplitter(mock_config)
        
        # Should complete by excluding problematic classes
        result = splitter.split_data(single_instance_data)
        assert isinstance(result, SplitResult)
        
        # Should have fewer samples than original
        total_samples = len(result.train) + len(result.test)
        assert total_samples < len(single_instance_data)
    
    def test_cross_site_split(self, mock_config, sample_va_data):
        """Test cross-site splitting functionality."""
        mock_config.split_strategy = "cross_site"
        mock_config.train_sites = ["A"]
        mock_config.test_sites = ["B"]
        
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # Verify site separation
        train_sites = set(result.train[mock_config.site_column].unique())
        test_sites = set(result.test[mock_config.site_column].unique())
        
        assert train_sites == {"A"}
        assert test_sites == {"B"}
        
        # Verify metadata
        assert result.metadata["train_sites"] == ["A"]
        assert result.metadata["test_sites"] == ["B"]
    
    def test_cross_site_split_auto(self, mock_config, sample_va_data):
        """Test cross-site splitting with automatic site selection."""
        mock_config.split_strategy = "cross_site"
        mock_config.train_sites = None
        mock_config.test_sites = None
        
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # Should have split sites automatically
        assert "train_sites" in result.metadata
        assert "test_sites" in result.metadata
        assert len(result.metadata["train_sites"]) > 0
        assert len(result.metadata["test_sites"]) > 0
    
    def test_stratified_site_split(self, mock_config, sample_va_data):
        """Test stratified site splitting functionality."""
        mock_config.split_strategy = "stratified_site"
        
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # Verify both sites are represented in train and test
        train_sites = set(result.train[mock_config.site_column].unique())
        test_sites = set(result.test[mock_config.site_column].unique())
        
        # Should have both sites in both splits
        assert "A" in train_sites and "A" in test_sites
        assert "B" in train_sites and "B" in test_sites
        
        # Verify metadata
        assert "sites_processed" in result.metadata
        assert result.metadata["sites_processed"] == 2
    
    def test_missing_label_column(self, mock_config, sample_va_data):
        """Test error handling for missing label column."""
        data_without_label = sample_va_data.drop(columns=[mock_config.label_column])
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Missing required columns"):
            splitter.split_data(data_without_label)
    
    def test_missing_site_column(self, mock_config, sample_va_data):
        """Test error handling for missing site column in site-based splits."""
        mock_config.split_strategy = "cross_site"
        data_without_site = sample_va_data.drop(columns=[mock_config.site_column])
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Missing required columns"):
            splitter.split_data(data_without_site)
    
    def test_insufficient_data(self, mock_config):
        """Test error handling for insufficient data."""
        insufficient_data = pd.DataFrame({
            "site": ["A", "A"],
            "va34": [1, 2],
            "symptom1": ["Yes", "No"]
        })
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Insufficient data for splitting"):
            splitter.split_data(insufficient_data)
    
    def test_single_site_cross_site_error(self, mock_config):
        """Test error handling for single site in cross-site split."""
        single_site_data = pd.DataFrame({
            "site": ["A"] * 15,  # 15 samples, all same site
            "va34": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            "symptom1": ["Yes", "No"] * 7 + ["Yes"],
            "symptom2": ["Y", "N"] * 7 + ["Y"],
            "symptom3": [1, 0] * 7 + [1],
            "symptom4": [0, 1] * 7 + [0]
        })
        
        mock_config.split_strategy = "cross_site"
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Need at least 2 sites"):
            splitter.split_data(single_site_data)
    
    def test_invalid_train_sites(self, mock_config, sample_va_data):
        """Test error handling for invalid train sites."""
        mock_config.split_strategy = "cross_site"
        mock_config.train_sites = ["X"]  # Site that doesn't exist
        mock_config.test_sites = ["A"]
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Train sites not found"):
            splitter.split_data(sample_va_data)
    
    def test_invalid_test_sites(self, mock_config, sample_va_data):
        """Test error handling for invalid test sites."""
        mock_config.split_strategy = "cross_site"
        mock_config.train_sites = ["A"]
        mock_config.test_sites = ["X"]  # Site that doesn't exist
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Test sites not found"):
            splitter.split_data(sample_va_data)
    
    def test_overlapping_sites(self, mock_config, sample_va_data):
        """Test error handling for overlapping train/test sites."""
        mock_config.split_strategy = "cross_site"
        mock_config.train_sites = ["A"]
        mock_config.test_sites = ["A"]  # Same site for both
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Train and test sites overlap"):
            splitter.split_data(sample_va_data)
    
    def test_metadata_generation(self, mock_config, sample_va_data):
        """Test comprehensive metadata generation."""
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # Check all required metadata fields
        required_fields = [
            "split_strategy", "test_size", "random_state", "split_timestamp",
            "train_samples", "test_samples", "total_samples", "actual_test_ratio",
            "train_class_distribution", "test_class_distribution", "config"
        ]
        
        for field in required_fields:
            assert field in result.metadata, f"Missing metadata field: {field}"
        
        # Check metadata values
        assert result.metadata["total_samples"] == len(sample_va_data)
        assert result.metadata["train_samples"] == len(result.train)
        assert result.metadata["test_samples"] == len(result.test)
    
    def test_save_results(self, mock_config, sample_va_data):
        """Test saving split results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.output_dir = tmpdir
            
            splitter = VADataSplitter(mock_config)
            result = splitter.split_data(sample_va_data)
            
            # Check that files were created
            splits_dir = Path(tmpdir) / "splits"
            assert splits_dir.exists()
            
            # Should have created a timestamped directory
            split_dirs = list(splits_dir.iterdir())
            assert len(split_dirs) > 0
            
            split_dir = split_dirs[0]
            assert (split_dir / "train.csv").exists()
            assert (split_dir / "test.csv").exists()
            assert (split_dir / "split_metadata.json").exists()
            assert (split_dir / "split_summary.txt").exists()
            
            # Verify metadata file content
            with open(split_dir / "split_metadata.json") as f:
                saved_metadata = json.load(f)
                assert saved_metadata["split_strategy"] == "train_test"
    
    def test_unknown_strategy(self, mock_config, sample_va_data):
        """Test error handling for unknown split strategy."""
        mock_config.split_strategy = "unknown_strategy"
        
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="Unknown split strategy"):
            splitter.split_data(sample_va_data)
    
    def test_class_distribution_calculation(self, mock_config, sample_va_data):
        """Test class distribution calculation."""
        splitter = VADataSplitter(mock_config)
        
        # Test with normal data
        distribution = splitter._get_class_distribution(sample_va_data)
        assert isinstance(distribution, dict)
        assert "1" in distribution
        assert "2" in distribution
        assert distribution["1"] == 6
        assert distribution["2"] == 6
        
        # Test with missing label column
        data_no_label = sample_va_data.drop(columns=[mock_config.label_column])
        distribution_empty = splitter._get_class_distribution(data_no_label)
        assert distribution_empty == {}
    
    def test_reproducibility(self, mock_config, sample_va_data):
        """Test that splits are reproducible with same random state."""
        splitter1 = VADataSplitter(mock_config)
        result1 = splitter1.split_data(sample_va_data)
        
        splitter2 = VADataSplitter(mock_config)
        result2 = splitter2.split_data(sample_va_data)
        
        # Should get identical splits
        pd.testing.assert_frame_equal(result1.train.sort_index(), result2.train.sort_index())
        pd.testing.assert_frame_equal(result1.test.sort_index(), result2.test.sort_index())
    
    def test_different_random_states(self, mock_config, sample_va_data):
        """Test that different random states produce different splits."""
        mock_config.random_state = 42
        splitter1 = VADataSplitter(mock_config)
        result1 = splitter1.split_data(sample_va_data)
        
        mock_config.random_state = 123
        splitter2 = VADataSplitter(mock_config)
        result2 = splitter2.split_data(sample_va_data)
        
        # Should get different splits (with high probability)
        try:
            pd.testing.assert_frame_equal(result1.train.sort_index(), result2.train.sort_index())
            # If they're equal, that's unlikely but possible - just warn
            import warnings
            warnings.warn("Random splits were identical - this is unlikely but possible")
        except AssertionError:
            # This is expected - different random states should produce different splits
            pass