"""Unit tests for VA data loader and preprocessor."""

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
from baseline.data.data_loader_preprocessor import VADataProcessor


class TestDataConfig:
    """Test cases for DataConfig model."""
    
    def test_data_config_creation(self):
        """Test creating a valid DataConfig."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            
        # Create empty CSV for testing
        pd.DataFrame().to_csv(tmp_path, index=False)
        
        try:
            config = DataConfig(
                data_path=tmp_path,
                output_dir="test_output/",
                openva_encoding=False
            )
            assert config.data_path == tmp_path
            assert config.output_dir == "test_output/"
            assert config.openva_encoding is False
            assert config.stratify_by_site is True
        finally:
            Path(tmp_path).unlink()
            
    def test_data_config_invalid_path(self):
        """Test DataConfig with non-existent file."""
        with pytest.raises(ValueError, match="Data file not found"):
            DataConfig(data_path="non_existent_file.csv")
            
    def test_data_config_invalid_extension(self):
        """Test DataConfig with non-CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            with pytest.raises(ValueError, match="Data file must be a CSV file"):
                DataConfig(data_path=tmp_path)
        finally:
            Path(tmp_path).unlink()
            
    def test_output_path_generation(self):
        """Test output path generation."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            
        pd.DataFrame().to_csv(tmp_path, index=False)
        
        try:
            config = DataConfig(data_path=tmp_path)
            output_path = config.get_output_path("adult", "numeric")
            
            assert "adult_numeric_" in str(output_path)
            assert output_path.suffix == ".csv"
            assert "processed_data" in str(output_path)
        finally:
            Path(tmp_path).unlink()


class TestVADataProcessor:
    """Test cases for VADataProcessor class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock(spec=DataConfig)
        config.data_path = "data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
        config.output_dir = "test_output/"
        config.openva_encoding = False
        config.drop_columns = []
        config.stratify_by_site = True
        config.get_output_path.return_value = Path("test_output/adult_numeric_test.csv")
        return config
    
    @pytest.fixture
    def sample_va_data(self):
        """Create sample VA data for testing."""
        return pd.DataFrame({
            "site": ["A", "A", "B", "B"],
            "va34": [1, 2, 1, 3],
            "cod5": [1, 1, 2, 2],
            "symptom1": ["Yes", "No", "Yes", "No"],
            "symptom2": ["Y", "N", "Y", "N"],
            "symptom3": [1, 0, 1, 0]
        })
    
    @patch("baseline.data.data_loader_preprocessor.PHMRCData")
    def test_load_and_process_numeric(self, mock_phmrc_data, mock_config, sample_va_data):
        """Test loading and processing data for ML models (numeric output)."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_instance.validate.return_value = sample_va_data
        mock_instance.xform.return_value = sample_va_data
        mock_phmrc_data.return_value = mock_instance
        
        # Create processor and process data
        processor = VADataProcessor(mock_config)
        df = processor.load_and_process()
        
        # Verify PHMRCData was called correctly
        mock_phmrc_data.assert_called_once_with(mock_config.data_path)
        mock_instance.validate.assert_called_once_with(nullable=False, drop=[])
        mock_instance.xform.assert_called_once_with("openva")
        
        # Check numeric conversion
        assert df["symptom1"].dtype in ['int64', 'float64']
        assert df["symptom2"].dtype in ['int64', 'float64']
        
    @patch("baseline.data.data_loader_preprocessor.PHMRCData")
    def test_load_and_process_openva(self, mock_phmrc_data, mock_config, sample_va_data):
        """Test loading and processing data for InSilicoVA (OpenVA encoding)."""
        # Configure for OpenVA encoding
        mock_config.openva_encoding = True
        mock_config.get_output_path.return_value = Path("test_output/adult_openva_test.csv")
        
        # Setup mocks
        mock_instance = MagicMock()
        mock_instance.validate.return_value = sample_va_data
        mock_instance.xform.return_value = sample_va_data.copy()
        mock_phmrc_data.return_value = mock_instance
        
        # Create processor and process data
        processor = VADataProcessor(mock_config)
        df = processor.load_and_process()
        
        # Check OpenVA encoding applied
        # The mock doesn't actually apply the encoding, so we verify the method was called
        assert mock_instance.xform.called
        
    def test_extract_dataset_name(self, mock_config):
        """Test dataset name extraction from file path."""
        processor = VADataProcessor(mock_config)
        
        # Test adult dataset
        mock_config.data_path = "IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
        assert processor._extract_dataset_name() == "adult"
        
        # Test child dataset
        mock_config.data_path = "IHME_PHMRC_VA_DATA_CHILD_Y2013M09D11_0.csv"
        assert processor._extract_dataset_name() == "child"
        
        # Test neonate dataset
        mock_config.data_path = "IHME_PHMRC_VA_DATA_NEONATE_Y2013M09D11_0.csv"
        assert processor._extract_dataset_name() == "neonate"
        
        # Test unknown dataset
        mock_config.data_path = "some_other_file.csv"
        assert processor._extract_dataset_name() == "unknown"
        
    def test_convert_categorical_to_numeric(self, mock_config, sample_va_data):
        """Test categorical to numeric conversion."""
        processor = VADataProcessor(mock_config)
        
        # Apply conversion
        df_numeric = processor._convert_categorical_to_numeric(sample_va_data)
        
        # Check conversions
        assert all(df_numeric["symptom1"] == [1, 0, 1, 0])
        assert all(df_numeric["symptom2"] == [1, 0, 1, 0])
        assert df_numeric["symptom1"].dtype in ['int64', 'float64']
        assert df_numeric["symptom2"].dtype in ['int64', 'float64']
        
    def test_apply_openva_encoding(self, mock_config, sample_va_data):
        """Test OpenVA encoding application."""
        processor = VADataProcessor(mock_config)
        
        # Prepare data with numeric values
        df = sample_va_data.copy()
        df["symptom1"] = [1, 0, 1, 0]
        df["symptom2"] = [1, 0, 2, 0]
        
        # Apply encoding
        df_encoded = processor._apply_openva_encoding(df)
        
        # Check encoding applied correctly
        assert all(df_encoded["symptom1"] == ["Y", "", "Y", ""])
        assert all(df_encoded["symptom2"] == ["Y", "", ".", ""])
        
    @patch("baseline.data.data_loader_preprocessor.PHMRCData")
    def test_data_validation_error(self, mock_phmrc_data, mock_config):
        """Test handling of data validation errors."""
        # Setup mock to raise exception
        mock_phmrc_data.side_effect = Exception("Validation failed")
        
        # Create processor and verify error handling
        processor = VADataProcessor(mock_config)
        with pytest.raises(ValueError, match="Data validation failed"):
            processor.load_and_process()
            
    @patch("baseline.data.data_loader_preprocessor.PHMRCData")
    def test_missing_file_error(self, mock_phmrc_data, mock_config):
        """Test handling of missing file errors."""
        # Setup mock to raise FileNotFoundError
        mock_phmrc_data.side_effect = FileNotFoundError("File not found")
        
        # Create processor and verify error handling
        processor = VADataProcessor(mock_config)
        with pytest.raises(ValueError, match="Data validation failed"):
            processor.load_and_process()
            
    def test_save_results(self, mock_config, sample_va_data):
        """Test saving results and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config to use temp directory
            mock_config.output_dir = tmpdir
            mock_config.get_output_path.return_value = Path(tmpdir) / "test_output.csv"
            
            processor = VADataProcessor(mock_config)
            output_path = processor._save_results(sample_va_data, "adult", "numeric")
            
            # Check file was created
            assert output_path.exists()
            
            # Check metadata was created
            metadata_path = output_path.with_suffix(".metadata.json")
            assert metadata_path.exists()
            
            # Verify metadata content
            with open(metadata_path) as f:
                metadata = json.load(f)
                assert metadata["dataset_name"] == "adult"
                assert metadata["encoding_type"] == "numeric"
                assert metadata["data_shape"] == list(sample_va_data.shape)
                
    @patch("baseline.data.data_loader_preprocessor.PHMRCData")
    def test_site_stratification(self, mock_phmrc_data, mock_config, sample_va_data):
        """Test processing with site stratification."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_instance.validate.return_value = sample_va_data
        mock_instance.xform.side_effect = lambda _, df=None: df if df is not None else sample_va_data
        mock_phmrc_data.return_value = mock_instance
        
        # Create processor and process with stratification
        processor = VADataProcessor(mock_config)
        results = processor.process_with_site_stratification()
        
        # Check results contain data for each site
        assert "A" in results
        assert "B" in results
        assert len(results["A"]) == 2
        assert len(results["B"]) == 2