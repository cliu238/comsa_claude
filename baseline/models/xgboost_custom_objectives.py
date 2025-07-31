"""Custom objective functions for XGBoost optimized for VA data.

This module provides custom objective functions that directly optimize
for CSMF accuracy and other VA-specific metrics.
"""

import numpy as np
import xgboost as xgb
from typing import Tuple


def csmf_weighted_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cause_weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Custom objective function that weights errors by cause-specific mortality fractions.
    
    This objective gives more weight to errors in predicting common causes,
    which directly impacts CSMF accuracy. It helps the model focus on getting
    the overall distribution right rather than just individual accuracy.
    
    Args:
        y_true: True labels (encoded as integers)
        y_pred: Predicted probabilities (raw predictions from XGBoost)
        cause_weights: Optional weights for each cause based on prevalence
        
    Returns:
        Tuple of (gradient, hessian)
    """
    # Convert raw predictions to probabilities using softmax
    # XGBoost provides raw scores, we need probabilities
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    # Calculate true cause fractions if weights not provided
    if cause_weights is None:
        n_classes = probs.shape[1]
        cause_counts = np.bincount(y_true.astype(int), minlength=n_classes)
        cause_weights = cause_counts / len(y_true)
        # Add small epsilon to avoid division by zero
        cause_weights = np.maximum(cause_weights, 1e-6)
    
    # Create weight matrix for each sample based on true cause
    sample_weights = cause_weights[y_true.astype(int)]
    
    # Calculate gradient and hessian with cause-specific weighting
    # This is based on multinomial log loss but with cause weights
    n_samples = len(y_true)
    grad = np.zeros_like(probs)
    hess = np.zeros_like(probs)
    
    for i in range(n_samples):
        true_class = int(y_true[i])
        
        # Gradient for true class
        grad[i, true_class] = (probs[i, true_class] - 1.0) * sample_weights[i]
        
        # Gradient for other classes
        for j in range(probs.shape[1]):
            if j != true_class:
                grad[i, j] = probs[i, j] * sample_weights[i]
        
        # Hessian (second derivative)
        for j in range(probs.shape[1]):
            hess[i, j] = probs[i, j] * (1.0 - probs[i, j]) * sample_weights[i]
    
    # Flatten for XGBoost
    return grad.flatten(), hess.flatten()


def balanced_focal_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Focal loss objective for handling class imbalance in VA data.
    
    Focal loss down-weights easy examples and focuses on hard cases,
    which is particularly useful for rare causes that are hard to predict.
    
    Args:
        y_true: True labels (encoded as integers)
        y_pred: Predicted probabilities (raw predictions)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class balancing parameter
        
    Returns:
        Tuple of (gradient, hessian)
    """
    # Convert to probabilities
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    n_samples = len(y_true)
    n_classes = probs.shape[1]
    
    # Calculate class weights based on frequency
    class_counts = np.bincount(y_true.astype(int), minlength=n_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    
    grad = np.zeros_like(probs)
    hess = np.zeros_like(probs)
    
    for i in range(n_samples):
        true_class = int(y_true[i])
        pt = probs[i, true_class]
        
        # Focal loss modulation
        focal_weight = (1 - pt) ** gamma
        
        # Class weight
        alpha_t = alpha * class_weights[true_class]
        
        # Gradient for true class
        grad[i, true_class] = alpha_t * focal_weight * (
            gamma * pt * np.log(pt + 1e-8) + pt - 1
        )
        
        # Gradient for other classes
        for j in range(n_classes):
            if j != true_class:
                grad[i, j] = alpha_t * focal_weight * probs[i, j]
        
        # Approximate hessian
        for j in range(n_classes):
            hess[i, j] = abs(grad[i, j]) * 0.1  # Simplified hessian
    
    return grad.flatten(), hess.flatten()


def domain_adversarial_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domain_labels: np.ndarray,
    lambda_domain: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Domain adversarial objective for better cross-site generalization.
    
    This objective includes a domain confusion term that encourages the model
    to learn features that are predictive of causes but invariant to sites.
    
    Args:
        y_true: True cause labels
        y_pred: Predicted probabilities
        domain_labels: Site/domain labels for each sample
        lambda_domain: Weight for domain confusion term
        
    Returns:
        Tuple of (gradient, hessian)
    """
    # Standard multinomial gradient
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    
    n_samples = len(y_true)
    n_classes = probs.shape[1]
    n_domains = len(np.unique(domain_labels))
    
    # Calculate domain statistics
    domain_probs = np.zeros((n_domains, n_classes))
    for d in range(n_domains):
        mask = domain_labels == d
        if mask.any():
            domain_probs[d] = probs[mask].mean(axis=0)
    
    # Calculate domain confusion gradient
    # We want to maximize confusion between domains
    domain_grad = np.zeros_like(probs)
    
    for i in range(n_samples):
        domain = domain_labels[i]
        # Gradient pushes predictions away from domain-specific patterns
        domain_diff = probs[i] - domain_probs[domain]
        domain_grad[i] = -lambda_domain * domain_diff
    
    # Combine standard gradient with domain confusion
    grad = np.zeros_like(probs)
    for i in range(n_samples):
        true_class = int(y_true[i])
        
        # Standard gradient
        grad[i, true_class] = probs[i, true_class] - 1.0
        for j in range(n_classes):
            if j != true_class:
                grad[i, j] = probs[i, j]
        
        # Add domain confusion term
        grad[i] += domain_grad[i]
    
    # Simplified hessian
    hess = np.ones_like(grad) * 0.1
    
    return grad.flatten(), hess.flatten()


class CSMFOptimizedObjective:
    """Wrapper class for using custom objectives with XGBoost.
    
    This class provides a clean interface for integrating custom objectives
    into the XGBoost training pipeline.
    """
    
    def __init__(self, objective_type: str = "csmf_weighted", **kwargs):
        """Initialize custom objective.
        
        Args:
            objective_type: Type of objective ("csmf_weighted", "focal", "domain_adversarial")
            **kwargs: Additional parameters for the objective function
        """
        self.objective_type = objective_type
        self.kwargs = kwargs
        
        # Map objective types to functions
        self.objective_map = {
            "csmf_weighted": csmf_weighted_objective,
            "focal": balanced_focal_objective,
            "domain_adversarial": domain_adversarial_objective,
        }
        
        if objective_type not in self.objective_map:
            raise ValueError(f"Unknown objective type: {objective_type}")
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Call the objective function.
        
        Args:
            y_true: True labels
            y_pred: Predicted values (raw scores from XGBoost)
            
        Returns:
            Tuple of (gradient, hessian)
        """
        objective_func = self.objective_map[self.objective_type]
        
        # Handle XGBoost's format (y_pred is flattened for multiclass)
        n_samples = len(y_true)
        n_classes = len(np.unique(y_true))
        
        # Reshape predictions if needed
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(n_samples, n_classes)
        
        return objective_func(y_true, y_pred, **self.kwargs)
    
    def get_xgb_objective(self):
        """Get objective function compatible with XGBoost's API.
        
        Returns:
            Callable objective function for XGBoost
        """
        def objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            """XGBoost-compatible objective function."""
            labels = dtrain.get_label()
            return self(labels, preds)
        
        return objective


def create_csmf_eval_metric():
    """Create CSMF accuracy evaluation metric for XGBoost.
    
    Returns:
        Callable evaluation function for XGBoost
    """
    def csmf_eval(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        """Evaluate CSMF accuracy during training."""
        labels = dtrain.get_label()
        n_samples = len(labels)
        n_classes = len(np.unique(labels))
        
        # Reshape predictions if needed
        if preds.ndim == 1:
            preds = preds.reshape(n_samples, n_classes)
        
        # Convert to probabilities
        exp_pred = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Get predicted classes
        pred_classes = np.argmax(probs, axis=1)
        
        # Calculate CSMF accuracy
        true_fractions = np.bincount(labels.astype(int), minlength=n_classes) / n_samples
        pred_fractions = np.bincount(pred_classes, minlength=n_classes) / n_samples
        
        diff = np.abs(true_fractions - pred_fractions).sum()
        min_frac = true_fractions[true_fractions > 0].min()
        
        csmf_acc = 1 - diff / (2 * (1 - min_frac))
        
        # XGBoost expects (name, value) tuple where higher is better
        return "csmf_acc", max(0, csmf_acc)
    
    return csmf_eval