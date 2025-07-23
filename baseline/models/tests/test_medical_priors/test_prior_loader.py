"""Tests for medical prior data loader."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import pandas as pd

from baseline.models.medical_priors import PriorLoader, MedicalPriors


class TestPriorLoader:
    """Test suite for PriorLoader."""
    
    def test_initialization(self):
        """Test PriorLoader initialization."""
        loader = PriorLoader()
        assert loader.prior_data_path is None
        assert loader._priors is None
        
        # With path
        loader_with_path = PriorLoader(Path("/some/path"))
        assert loader_with_path.prior_data_path == Path("/some/path")
        
    def test_load_simulated_priors(self):
        """Test loading simulated priors."""
        loader = PriorLoader()
        priors = loader.load_priors()
        
        # Check structure
        assert isinstance(priors, MedicalPriors)
        assert len(priors.symptom_names) > 0
        assert len(priors.cause_names) > 0
        assert len(priors.conditional_probs) > 0
        assert len(priors.cause_priors) > 0
        assert len(priors.implausible_patterns) > 0
        
        # Check conditional matrix
        assert priors.conditional_matrix is not None
        assert priors.conditional_matrix.shape == (
            len(priors.symptom_names),
            len(priors.cause_names)
        )
        
    def test_prior_properties(self):
        """Test properties of loaded priors."""
        loader = PriorLoader()
        priors = loader.load_priors()
        
        # Check cause priors sum to 1
        total = sum(priors.cause_priors.values())
        assert np.isclose(total, 1.0, rtol=1e-5)
        
        # Check conditional probabilities are in valid range
        for prob in priors.conditional_probs.values():
            assert 0 <= prob <= 1
            
        # Check all symptoms and causes have entries
        for symptom in priors.symptom_names:
            for cause in priors.cause_names:
                assert (symptom, cause) in priors.conditional_probs
                
    def test_medical_knowledge_encoding(self):
        """Test that medical knowledge is properly encoded."""
        loader = PriorLoader()
        priors = loader.load_priors()
        
        # Test some medical associations
        # Fever should be highly associated with infectious diseases
        assert priors.conditional_probs[("fever", "malaria")] > 0.5
        assert priors.conditional_probs[("fever", "tuberculosis")] > 0.5
        assert priors.conditional_probs[("fever", "pneumonia")] > 0.5
        
        # Injury should be associated with trauma
        assert priors.conditional_probs[("injury", "road_traffic")] > 0.8
        assert priors.conditional_probs[("injury", "homicide")] > 0.8
        
        # Check implausible patterns
        implausible_pairs = [(s, c) for s, c in priors.implausible_patterns]
        assert ("injury", "diabetes") in implausible_pairs
        assert ("injury", "tuberculosis") in implausible_pairs
        
    def test_save_priors(self):
        """Test saving priors to files."""
        loader = PriorLoader()
        priors = loader.load_priors()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            loader.save_priors(priors, output_path)
            
            # Check files were created
            assert (output_path / "conditional_probs.csv").exists()
            assert (output_path / "cause_priors.csv").exists()
            assert (output_path / "implausible_patterns.csv").exists()
            
            # Check content
            cond_df = pd.read_csv(output_path / "conditional_probs.csv")
            assert len(cond_df) == len(priors.conditional_probs)
            assert set(cond_df.columns) == {"symptom", "cause", "probability"}
            
            cause_df = pd.read_csv(output_path / "cause_priors.csv")
            assert len(cause_df) == len(priors.cause_priors)
            assert set(cause_df.columns) == {"cause", "prior_probability"}
            
            imp_df = pd.read_csv(output_path / "implausible_patterns.csv")
            assert len(imp_df) == len(priors.implausible_patterns)
            assert set(imp_df.columns) == {"symptom", "cause"}
            
    def test_caching(self):
        """Test that priors are cached after first load."""
        loader = PriorLoader()
        
        # First load
        priors1 = loader.load_priors()
        
        # Second load should return same object
        priors2 = loader.load_priors()
        
        assert priors1 is priors2
        
    def test_conditional_matrix_consistency(self):
        """Test that conditional matrix matches dictionary."""
        loader = PriorLoader()
        priors = loader.load_priors()
        
        # Check random entries
        for _ in range(10):
            i = np.random.randint(0, len(priors.symptom_names))
            j = np.random.randint(0, len(priors.cause_names))
            
            symptom = priors.symptom_names[i]
            cause = priors.cause_names[j]
            
            dict_value = priors.conditional_probs[(symptom, cause)]
            matrix_value = priors.conditional_matrix[i, j]
            
            assert np.isclose(dict_value, matrix_value)