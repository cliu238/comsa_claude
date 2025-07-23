"""Tests for prior probability calculator."""

import numpy as np
import pytest

from baseline.models.medical_priors import PriorLoader, PriorCalculator


class TestPriorCalculator:
    """Test suite for PriorCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator with test priors."""
        loader = PriorLoader()
        priors = loader.load_priors()
        return PriorCalculator(priors)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample symptom features."""
        np.random.seed(42)
        # 10 samples, 15 features (matching symptom count)
        return np.random.binomial(1, 0.3, size=(10, 15))
    
    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.priors is not None
        assert calculator._n_symptoms == len(calculator.priors.symptom_names)
        assert calculator._n_causes == len(calculator.priors.cause_names)
        assert len(calculator._symptom_to_idx) == calculator._n_symptoms
        assert len(calculator._cause_to_idx) == calculator._n_causes
        
    def test_calculate_symptom_likelihood(self, calculator, sample_features):
        """Test symptom likelihood calculation."""
        likelihood = calculator._calculate_symptom_likelihood(sample_features)
        
        # Check shape
        assert likelihood.shape == (sample_features.shape[0], calculator._n_causes)
        
        # Check properties
        assert np.all(likelihood >= 0)  # Non-negative
        assert np.all(likelihood <= 1)  # Normalized
        
        # Check normalization
        row_sums = likelihood.sum(axis=1)
        assert np.allclose(row_sums, 1.0, rtol=1e-5)
        
    def test_calculate_prior_gradient(self, calculator, sample_features):
        """Test prior gradient calculation."""
        n_samples = sample_features.shape[0]
        n_causes = calculator._n_causes
        
        # Create mock predictions
        predictions = np.random.dirichlet(np.ones(n_causes), size=n_samples)
        
        # Calculate gradient
        gradient = calculator.calculate_prior_gradient(
            predictions, sample_features, lambda_prior=0.1
        )
        
        # Check shape
        assert gradient.shape == predictions.shape
        
        # Gradient should push predictions toward priors
        # So it should have both positive and negative values
        assert np.any(gradient > 0)
        assert np.any(gradient < 0)
        
    def test_calculate_prior_features(self, calculator, sample_features):
        """Test prior feature calculation."""
        prior_features = calculator.calculate_prior_features(sample_features)
        
        # Check shape
        n_samples = sample_features.shape[0]
        n_causes = calculator._n_causes
        
        # Expected features: likelihood + log_odds + ranks + cause_priors + plausibility
        expected_features = n_causes * 5
        assert prior_features.shape == (n_samples, expected_features)
        
        # Check that features are finite
        assert np.all(np.isfinite(prior_features))
        
    def test_feature_names(self, calculator):
        """Test feature name generation."""
        feature_names = calculator.get_feature_names()
        
        # Check count
        n_causes = calculator._n_causes
        expected_count = n_causes * 5  # 5 types of features
        assert len(feature_names) == expected_count
        
        # Check naming patterns
        likelihood_names = [n for n in feature_names if "prior_likelihood_" in n]
        log_odds_names = [n for n in feature_names if "prior_log_odds_" in n]
        rank_names = [n for n in feature_names if "prior_rank_" in n]
        cause_prior_names = [n for n in feature_names if "cause_prior_" in n]
        plausibility_names = [n for n in feature_names if "plausibility_" in n]
        
        assert len(likelihood_names) == n_causes
        assert len(log_odds_names) == n_causes
        assert len(rank_names) == n_causes
        assert len(cause_prior_names) == n_causes
        assert len(plausibility_names) == n_causes
        
    def test_plausibility_scores(self, calculator, sample_features):
        """Test medical plausibility scoring."""
        # Create features with known implausible patterns
        features = np.zeros((2, 15))
        
        # Get indices for injury and diabetes
        injury_idx = calculator._symptom_to_idx.get("injury", 0)
        diabetes_idx = calculator._cause_to_idx.get("diabetes", 0)
        
        # First sample: has injury symptom
        features[0, injury_idx] = 1
        
        # Calculate plausibility
        plausibility = calculator._calculate_plausibility_scores(features)
        
        # Check that diabetes has low plausibility for injury
        if injury_idx < features.shape[1] and diabetes_idx < calculator._n_causes:
            assert plausibility[0, diabetes_idx] < plausibility[1, diabetes_idx]
            
    def test_symptom_cause_associations(self, calculator):
        """Test that known associations are captured."""
        # Create specific symptom patterns
        features = np.zeros((3, 15))
        
        # Pattern 1: Fever symptoms
        fever_idx = calculator._symptom_to_idx.get("fever", 0)
        if fever_idx < features.shape[1]:
            features[0, fever_idx] = 1
            
        # Pattern 2: Injury symptoms
        injury_idx = calculator._symptom_to_idx.get("injury", 0)
        if injury_idx < features.shape[1]:
            features[1, injury_idx] = 1
            
        # Calculate likelihood
        likelihood = calculator._calculate_symptom_likelihood(features)
        
        # Check associations
        malaria_idx = calculator._cause_to_idx.get("malaria", 0)
        road_traffic_idx = calculator._cause_to_idx.get("road_traffic", 0)
        
        if fever_idx < features.shape[1] and malaria_idx < calculator._n_causes:
            # Fever should increase malaria likelihood
            assert likelihood[0, malaria_idx] > likelihood[2, malaria_idx]
            
        if injury_idx < features.shape[1] and road_traffic_idx < calculator._n_causes:
            # Injury should increase road traffic likelihood
            assert likelihood[1, road_traffic_idx] > likelihood[2, road_traffic_idx]
            
    def test_edge_cases(self, calculator):
        """Test edge cases."""
        # Empty features (no symptoms)
        empty_features = np.zeros((5, 15))
        likelihood = calculator._calculate_symptom_likelihood(empty_features)
        assert np.all(np.isfinite(likelihood))
        assert np.allclose(likelihood.sum(axis=1), 1.0)
        
        # All symptoms present
        all_features = np.ones((5, 15))
        likelihood = calculator._calculate_symptom_likelihood(all_features)
        assert np.all(np.isfinite(likelihood))
        assert np.allclose(likelihood.sum(axis=1), 1.0)
        
        # Single sample
        single_feature = np.random.binomial(1, 0.5, size=(1, 15))
        prior_features = calculator.calculate_prior_features(single_feature)
        assert prior_features.shape[0] == 1
        assert np.all(np.isfinite(prior_features))