"""Calculator for prior-based probabilities and features.

This module provides functions to calculate prior probabilities and
create features based on medical knowledge.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax

from .prior_loader import MedicalPriors

logger = logging.getLogger(__name__)


class PriorCalculator:
    """Calculates prior-based probabilities and features."""
    
    def __init__(self, priors: MedicalPriors):
        """Initialize calculator with medical priors.
        
        Args:
            priors: MedicalPriors object containing prior probabilities
        """
        self.priors = priors
        self._symptom_to_idx = {s: i for i, s in enumerate(priors.symptom_names)}
        self._cause_to_idx = {c: i for i, c in enumerate(priors.cause_names)}
        self._n_symptoms = len(priors.symptom_names)
        self._n_causes = len(priors.cause_names)
        
    def calculate_prior_gradient(
        self, 
        predictions: np.ndarray,
        features: np.ndarray,
        lambda_prior: float = 0.1
    ) -> np.ndarray:
        """Calculate gradient contribution from prior knowledge.
        
        Args:
            predictions: Current prediction probabilities (n_samples, n_classes)
            features: Input features (n_samples, n_features)
            lambda_prior: Weight for prior term
            
        Returns:
            Gradient adjustment from priors (n_samples, n_classes)
        """
        n_samples, n_classes = predictions.shape
        prior_grad = np.zeros_like(predictions)
        
        # Convert cause priors to array
        cause_prior_array = np.array([
            self.priors.cause_priors.get(cause, 1/n_classes)
            for cause in self.priors.cause_names[:n_classes]
        ])
        
        # Calculate symptom-based likelihood for each cause
        symptom_likelihood = self._calculate_symptom_likelihood(features)
        
        # Prior gradient pushes predictions toward prior-weighted likelihoods
        for i in range(n_samples):
            # Combine symptom likelihood with cause priors
            prior_probs = symptom_likelihood[i] * cause_prior_array
            prior_probs = prior_probs / (prior_probs.sum() + 1e-8)
            
            # Gradient points from current predictions to prior predictions
            prior_grad[i] = lambda_prior * (prior_probs - predictions[i])
            
        return prior_grad
    
    def calculate_prior_features(self, features: np.ndarray) -> np.ndarray:
        """Create prior-based features from symptoms.
        
        Args:
            features: Input features (n_samples, n_features)
            
        Returns:
            Prior-based features to augment the original features
        """
        n_samples = features.shape[0]
        
        # Calculate symptom likelihood for each cause
        symptom_likelihood = self._calculate_symptom_likelihood(features)
        
        # Create features:
        # 1. Raw symptom likelihoods
        # 2. Log odds relative to uniform
        # 3. Rank features
        # 4. Prior cause probabilities
        
        prior_features = []
        
        # 1. Symptom likelihoods (normalized)
        prior_features.append(symptom_likelihood)
        
        # 2. Log odds relative to uniform
        uniform_prob = 1.0 / self._n_causes
        log_odds = np.log(symptom_likelihood + 1e-8) - np.log(uniform_prob)
        prior_features.append(log_odds)
        
        # 3. Rank features (which causes are most likely)
        ranks = np.zeros_like(symptom_likelihood)
        for i in range(n_samples):
            # Rank in descending order (highest prob = rank 1)
            sorted_indices = np.argsort(-symptom_likelihood[i])
            ranks[i, sorted_indices] = np.arange(1, self._n_causes + 1)
        prior_features.append(ranks / self._n_causes)  # Normalize ranks
        
        # 4. Prior cause probabilities (repeated for each sample)
        cause_prior_array = np.array([
            self.priors.cause_priors.get(cause, 1/self._n_causes)
            for cause in self.priors.cause_names[:self._n_causes]
        ])
        cause_prior_features = np.tile(cause_prior_array, (n_samples, 1))
        prior_features.append(cause_prior_features)
        
        # 5. Medical plausibility scores
        plausibility = self._calculate_plausibility_scores(features)
        prior_features.append(plausibility)
        
        # Concatenate all prior features
        return np.hstack(prior_features)
    
    def _calculate_symptom_likelihood(self, features: np.ndarray) -> np.ndarray:
        """Calculate likelihood of each cause given observed symptoms.
        
        Args:
            features: Binary symptom indicators (n_samples, n_features)
            
        Returns:
            Likelihood scores for each cause (n_samples, n_causes)
        """
        n_samples = features.shape[0]
        likelihood = np.ones((n_samples, self._n_causes))
        
        # Use the precomputed conditional matrix for efficiency
        # For each sample, calculate P(symptoms | cause) for all causes
        for i in range(n_samples):
            for j, cause in enumerate(self.priors.cause_names[:self._n_causes]):
                # Calculate likelihood using present symptoms
                for k in range(min(features.shape[1], self._n_symptoms)):
                    if features[i, k] > 0.5:  # Symptom is present
                        # P(symptom=1 | cause)
                        prob = self.priors.conditional_matrix[k, j]
                        likelihood[i, j] *= max(prob, 0.01)  # Avoid zeros
                    else:  # Symptom is absent
                        # P(symptom=0 | cause) = 1 - P(symptom=1 | cause)
                        prob = 1 - self.priors.conditional_matrix[k, j]
                        likelihood[i, j] *= max(prob, 0.01)
        
        # Normalize to get relative likelihoods
        # Use log-sum-exp trick for numerical stability
        log_likelihood = np.log(likelihood + 1e-10)
        max_log = np.max(log_likelihood, axis=1, keepdims=True)
        exp_diff = np.exp(log_likelihood - max_log)
        row_sums = exp_diff.sum(axis=1, keepdims=True)
        likelihood = exp_diff / row_sums
        
        return likelihood
    
    def _calculate_plausibility_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate medical plausibility scores for each cause.
        
        Args:
            features: Binary symptom indicators (n_samples, n_features)
            
        Returns:
            Plausibility scores (n_samples, n_causes)
        """
        n_samples = features.shape[0]
        plausibility = np.ones((n_samples, self._n_causes))
        
        # Penalize implausible patterns
        for symptom, cause in self.priors.implausible_patterns:
            if symptom in self._symptom_to_idx and cause in self._cause_to_idx:
                symptom_idx = self._symptom_to_idx[symptom]
                cause_idx = self._cause_to_idx[cause]
                
                if symptom_idx < features.shape[1] and cause_idx < self._n_causes:
                    # Reduce plausibility when implausible pattern is present
                    mask = features[:, symptom_idx] > 0.5
                    plausibility[mask, cause_idx] *= 0.1
        
        return plausibility
    
    def get_feature_names(self) -> List[str]:
        """Get names for all prior-based features.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Symptom likelihood features
        for cause in self.priors.cause_names[:self._n_causes]:
            feature_names.append(f"prior_likelihood_{cause}")
            
        # Log odds features
        for cause in self.priors.cause_names[:self._n_causes]:
            feature_names.append(f"prior_log_odds_{cause}")
            
        # Rank features
        for cause in self.priors.cause_names[:self._n_causes]:
            feature_names.append(f"prior_rank_{cause}")
            
        # Cause prior features
        for cause in self.priors.cause_names[:self._n_causes]:
            feature_names.append(f"cause_prior_{cause}")
            
        # Plausibility features
        for cause in self.priors.cause_names[:self._n_causes]:
            feature_names.append(f"plausibility_{cause}")
            
        return feature_names