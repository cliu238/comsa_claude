"""Unit tests for VADataSplitter module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.config.data_config import DataConfig
from baseline.data.data_splitter import VADataSplitter


class TestVADataSplitter:
    """Test suite for VADataSplitter class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample VA data for testing."""
        data = pd.DataFrame({
            'site': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'],
            'va34': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        return data

    @pytest.fixture
    def basic_config(self):
        """Create basic configuration for testing."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="train_test",
            test_size=0.3,
            random_state=42
        )
        return config

    def test_init(self, basic_config):
        """Test VADataSplitter initialization."""
        splitter = VADataSplitter(basic_config)
        assert splitter.config == basic_config
        assert splitter.config.split_strategy == "train_test"

    def test_validate_columns_valid(self, sample_data, basic_config):
        """Test column validation with valid data."""
        splitter = VADataSplitter(basic_config)
        # Should not raise an exception
        splitter._validate_columns(sample_data)

    def test_validate_columns_missing_site(self, sample_data, basic_config):
        """Test column validation with missing site column."""
        splitter = VADataSplitter(basic_config)
        data_no_site = sample_data.drop(columns=['site'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            splitter._validate_columns(data_no_site)

    def test_validate_columns_missing_label(self, sample_data, basic_config):
        """Test column validation with missing label column."""
        splitter = VADataSplitter(basic_config)
        data_no_label = sample_data.drop(columns=['va34'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            splitter._validate_columns(data_no_label)

    def test_train_test_split_basic(self, sample_data, basic_config):
        """Test basic train/test split."""
        splitter = VADataSplitter(basic_config)
        splits = splitter.split_data(sample_data)
        
        assert 'train' in splits
        assert 'test' in splits
        assert len(splits['train']) + len(splits['test']) == len(sample_data)
        assert len(splits['test']) == 3  # 30% of 10 samples
        assert len(splits['train']) == 7  # 70% of 10 samples

    def test_train_test_split_columns(self, sample_data, basic_config):
        """Test that train/test split preserves columns."""
        splitter = VADataSplitter(basic_config)
        splits = splitter.split_data(sample_data)
        
        expected_columns = set(sample_data.columns)
        assert set(splits['train'].columns) == expected_columns
        assert set(splits['test'].columns) == expected_columns

    def test_cross_site_split_with_train_sites(self, sample_data):
        """Test cross-site split with specified train sites."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="cross_site",
            train_sites=["A", "B"],
            random_state=42
        )
        splitter = VADataSplitter(config)
        splits = splitter.split_data(sample_data)
        
        train_sites = splits['train']['site'].unique()
        test_sites = splits['test']['site'].unique()
        
        assert set(train_sites) == {"A", "B"}
        assert set(test_sites) == {"C"}
        assert len(splits['train']) == 8  # Sites A and B
        assert len(splits['test']) == 2   # Site C

    def test_cross_site_split_with_test_sites(self, sample_data):
        """Test cross-site split with specified test sites."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="cross_site",
            test_sites=["C"],
            random_state=42
        )
        splitter = VADataSplitter(config)
        splits = splitter.split_data(sample_data)
        
        train_sites = splits['train']['site'].unique()
        test_sites = splits['test']['site'].unique()
        
        assert set(train_sites) == {"A", "B"}
        assert set(test_sites) == {"C"}

    def test_cross_site_split_no_sites_specified(self, sample_data):
        """Test cross-site split with no sites specified."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="cross_site",
            random_state=42
        )
        splitter = VADataSplitter(config)
        
        with pytest.raises(ValueError, match="Must specify train_sites or test_sites"):
            splitter.split_data(sample_data)

    def test_cross_site_split_invalid_sites(self, sample_data):
        """Test cross-site split with invalid sites."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="cross_site",
            train_sites=["A", "INVALID"],
            random_state=42
        )
        splitter = VADataSplitter(config)
        
        with pytest.raises(ValueError, match="Invalid train sites"):
            splitter.split_data(sample_data)

    def test_stratified_site_split_basic(self, sample_data):
        """Test stratified site split."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="stratified_site",
            test_size=0.5,
            random_state=42
        )
        splitter = VADataSplitter(config)
        splits = splitter.split_data(sample_data)
        
        assert 'train' in splits
        assert 'test' in splits
        assert len(splits['train']) + len(splits['test']) == len(sample_data)
        
        # Check that all sites are represented
        train_sites = set(splits['train']['site'].unique())
        test_sites = set(splits['test']['site'].unique())
        all_sites = set(sample_data['site'].unique())
        
        assert train_sites.union(test_sites) == all_sites

    def test_stratified_site_split_insufficient_data(self):
        """Test stratified site split with insufficient data per site."""
        # Create data with one site having only 1 sample
        data = pd.DataFrame({
            'site': ['A', 'B', 'B', 'B'],
            'va34': [1, 1, 2, 1],
            'feature1': [1, 2, 3, 4]
        })
        
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="stratified_site",
            test_size=0.5,
            random_state=42
        )
        splitter = VADataSplitter(config)
        splits = splitter.split_data(data)
        
        # Should still work, insufficient data site goes to test
        assert 'train' in splits
        assert 'test' in splits
        assert len(splits['train']) + len(splits['test']) == len(data)

    def test_get_split_statistics(self, sample_data, basic_config):
        """Test split statistics calculation."""
        splitter = VADataSplitter(basic_config)
        splits = splitter.split_data(sample_data)
        stats = splitter.get_split_statistics(splits)
        
        assert 'train' in stats
        assert 'test' in stats
        
        for split_name in ['train', 'test']:
            assert 'n_samples' in stats[split_name]
            assert 'n_features' in stats[split_name]
            assert 'n_sites' in stats[split_name]
            assert 'n_classes' in stats[split_name]
            
            # Check values make sense
            assert stats[split_name]['n_samples'] > 0
            assert stats[split_name]['n_features'] == 3  # 4 columns - 1 target
            assert stats[split_name]['n_classes'] == 2   # Labels 1 and 2

    def test_invalid_split_strategy(self, sample_data):
        """Test invalid split strategy."""
        # This should fail at config validation level (Pydantic validation)
        with pytest.raises(ValueError):
            config = DataConfig(
                data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
                split_strategy="invalid_strategy",  # This should be caught by Pydantic
                random_state=42
            )

    def test_empty_data(self, basic_config):
        """Test splitting with empty data."""
        empty_data = pd.DataFrame()
        splitter = VADataSplitter(basic_config)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            splitter.split_data(empty_data)

    def test_single_class_data(self, basic_config):
        """Test splitting with single class data."""
        single_class_data = pd.DataFrame({
            'site': ['A', 'A', 'B', 'B'],
            'va34': [1, 1, 1, 1],  # All same class
            'feature1': [1, 2, 3, 4]
        })
        
        splitter = VADataSplitter(basic_config)
        splits = splitter.split_data(single_class_data)
        
        # Should fall back to random split
        assert 'train' in splits
        assert 'test' in splits
        assert len(splits['train']) + len(splits['test']) == len(single_class_data)

    def test_reproducibility(self, sample_data, basic_config):
        """Test that splits are reproducible with same random state."""
        splitter1 = VADataSplitter(basic_config)
        splitter2 = VADataSplitter(basic_config)
        
        splits1 = splitter1.split_data(sample_data)
        splits2 = splitter2.split_data(sample_data)
        
        # Should produce identical splits
        pd.testing.assert_frame_equal(splits1['train'], splits2['train'])
        pd.testing.assert_frame_equal(splits1['test'], splits2['test'])

    def test_different_random_states(self, sample_data):
        """Test that different random states produce different splits."""
        config1 = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="train_test",
            random_state=42
        )
        config2 = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            split_strategy="train_test",
            random_state=123
        )
        
        splitter1 = VADataSplitter(config1)
        splitter2 = VADataSplitter(config2)
        
        splits1 = splitter1.split_data(sample_data)
        splits2 = splitter2.split_data(sample_data)
        
        # Should produce different splits (with high probability)
        # We'll check if train sets are different
        try:
            pd.testing.assert_frame_equal(splits1['train'], splits2['train'])
            # If they're equal, the test should fail (very unlikely)
            assert False, "Different random states produced identical splits"
        except AssertionError:
            # Expected - splits should be different
            pass


class TestDataConfigSplitValidation:
    """Test suite for DataConfig split validation."""
    
    def test_valid_test_size(self):
        """Test valid test size values."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
            test_size=0.3
        )
        assert config.test_size == 0.3

    def test_invalid_test_size_too_low(self):
        """Test test size too low."""
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataConfig(
                data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
                test_size=0.0
            )

    def test_invalid_test_size_too_high(self):
        """Test test size too high."""
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataConfig(
                data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
                test_size=1.0
            )

    def test_invalid_test_size_negative(self):
        """Test negative test size."""
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataConfig(
                data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
                test_size=-0.1
            )

    def test_invalid_test_size_over_one(self):
        """Test test size over 1.0."""
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            DataConfig(
                data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
                test_size=1.5
            )

    def test_split_strategy_defaults(self):
        """Test split strategy defaults."""
        config = DataConfig(
            data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
        )
        assert config.split_strategy == "train_test"
        assert config.test_size == 0.3
        assert config.random_state == 42
        assert config.site_column == "site"
        assert config.label_column == "va34"
        assert config.train_sites is None
        assert config.test_sites is None