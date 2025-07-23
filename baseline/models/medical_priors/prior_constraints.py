"""Prior-based constraints for XGBoost custom objectives.

This module implements the custom objective function that incorporates
medical priors into XGBoost training.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.special import softmax

from .prior_calculator import PriorCalculator
from .prior_loader import MedicalPriors

logger = logging.getLogger(__name__)


class PriorConstraints:
    """Implements prior-informed custom objective for XGBoost."""
    
    def __init__(self, priors: MedicalPriors, lambda_prior: float = 0.1):
        """Initialize constraints with medical priors.
        
        Args:
            priors: MedicalPriors object containing prior probabilities
            lambda_prior: Weight for prior term in objective
        """
        self.priors = priors
        self.lambda_prior = lambda_prior
        self.calculator = PriorCalculator(priors)
        self._features = None  # Will be set during training
        
    def set_features(self, features: np.ndarray) -> None:
        """Set features for current training batch.
        
        Args:
            features: Training features (n_samples, n_features)
        """
        self._features = features
        
    def prior_informed_objective(
        self, 
        y_pred: np.ndarray, 
        dtrain
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Custom objective function incorporating medical priors.
        
        This is designed to be used with XGBoost's custom objective interface.
        
        Args:
            y_pred: Current predictions (raw scores before softmax)
            dtrain: XGBoost DMatrix containing labels
            
        Returns:
            Tuple of (gradient, hessian) arrays
        """
        # Get true labels
        y_true = dtrain.get_label()
        n_samples = len(y_true)
        
        # Reshape predictions for multi-class
        # XGBoost gives us flat array, reshape to (n_samples, n_classes)
        n_classes = len(self.priors.cause_names)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((n_samples, n_classes))
        
        # Convert to probabilities
        probs = softmax(y_pred, axis=1)
        
        # Calculate standard gradient for cross-entropy
        # Create one-hot encoding of true labels
        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(n_samples), y_true.astype(int)] = 1
        
        # Standard gradient: p - y
        grad_data = probs - y_one_hot
        
        # Calculate prior gradient if features are available
        if self._features is not None:
            grad_prior = self.calculator.calculate_prior_gradient(
                probs, self._features, self.lambda_prior
            )
            grad = grad_data + grad_prior
        else:
            grad = grad_data
            logger.warning("Features not set, using standard gradient only")
        
        # Calculate Hessian (second derivative)
        # For softmax: H_ij = p_i * (delta_ij - p_j)
        # We use a diagonal approximation for efficiency
        hess = probs * (1 - probs) + 1e-6  # Add small epsilon for stability
        
        # Flatten back to 1D as XGBoost expects
        grad = grad.flatten()
        hess = hess.flatten()
        
        return grad, hess
    
    def create_fobj(self):
        """Create objective function callable for XGBoost.
        
        Returns:
            Callable objective function
        """
        def fobj(y_pred, dtrain):
            return self.prior_informed_objective(y_pred, dtrain)
        return fobj
    
    def evaluate_prior_influence(
        self, 
        predictions: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the influence of priors on predictions.
        
        Args:
            predictions: Model predictions (probabilities)
            features: Input features
            
        Returns:
            Dictionary with prior influence metrics
        """
        # Calculate what predictions would be with only priors
        prior_only = self.calculator._calculate_symptom_likelihood(features)
        
        # Calculate KL divergence between predictions and priors
        kl_div = np.mean([
            np.sum(pred * np.log(pred / (prior + 1e-8) + 1e-8))
            for pred, prior in zip(predictions, prior_only)
        ])
        
        # Calculate correlation between predictions and priors
        correlation = np.corrcoef(
            predictions.flatten(),
            prior_only.flatten()
        )[0, 1]
        
        # Calculate average prior contribution
        # This estimates how much the prior term affected the final predictions
        prior_grad = self.calculator.calculate_prior_gradient(
            predictions, features, self.lambda_prior
        )
        avg_contribution = np.abs(prior_grad).mean() / np.abs(predictions).mean()
        
        return {
            "kl_divergence": float(kl_div),
            "correlation": float(correlation),
            "avg_contribution": float(avg_contribution),
            "lambda_prior": self.lambda_prior
        }