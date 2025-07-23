"""Example usage of XGBoost with medical prior integration.

This script demonstrates how to use the prior-enhanced XGBoost model
for improved cross-site generalization in VA cause-of-death prediction.
"""

import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from baseline.models import XGBoostModel, XGBoostPriorEnhanced
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_prior_config import XGBoostPriorConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_va_data(data_path: Path = None):
    """Load and preprocess VA data.
    
    Args:
        data_path: Path to VA data CSV file
        
    Returns:
        Tuple of (X, y) features and labels
    """
    if data_path is None:
        # Use example data if no path provided
        logger.info("No data path provided, generating synthetic VA data")
        np.random.seed(42)
        n_samples = 500
        n_features = 50  # Number of symptoms
        n_classes = 10   # Number of causes
        
        # Generate synthetic VA-like data
        X = np.random.binomial(1, 0.2, size=(n_samples, n_features))
        
        # Create structured labels (certain symptoms correlate with causes)
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Simple rules to create structure
            if X[i, :5].sum() >= 3:
                y[i] = 0  # Cause 0 associated with first 5 symptoms
            elif X[i, 5:10].sum() >= 3:
                y[i] = 1  # Cause 1 associated with symptoms 5-10
            elif X[i, 10:15].sum() >= 3:
                y[i] = 2
            else:
                y[i] = np.random.randint(0, n_classes)
        
        # Convert to DataFrame/Series for compatibility
        feature_names = [f"symptom_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="cause")
        
        return X_df, y_series
    else:
        # Load real VA data
        logger.info(f"Loading VA data from {data_path}")
        # This would load and preprocess real VA data
        # For now, return synthetic data
        return load_va_data(None)


def compare_models(X_train, X_test, y_train, y_test):
    """Compare vanilla XGBoost with prior-enhanced version.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # 1. Vanilla XGBoost
    logger.info("Training vanilla XGBoost...")
    vanilla_config = XGBoostConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3
    )
    vanilla_model = XGBoostModel(vanilla_config)
    vanilla_model.fit(X_train, y_train)
    
    # Evaluate
    vanilla_pred = vanilla_model.predict(X_test)
    vanilla_csmf = vanilla_model.calculate_csmf_accuracy(X_test, y_test)
    results['vanilla'] = {
        'model': vanilla_model,
        'predictions': vanilla_pred,
        'csmf_accuracy': vanilla_csmf
    }
    logger.info(f"Vanilla XGBoost CSMF accuracy: {vanilla_csmf:.3f}")
    
    # 2. Prior-enhanced XGBoost (Feature Engineering)
    logger.info("Training XGBoost with prior features...")
    feature_config = XGBoostPriorConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        use_medical_priors=True,
        prior_method="feature_engineering",
        feature_prior_weight=1.0
    )
    feature_model = XGBoostPriorEnhanced(feature_config)
    feature_model.fit(X_train, y_train)
    
    feature_pred = feature_model.predict(X_test)
    feature_csmf = feature_model.calculate_csmf_accuracy(X_test, y_test)
    results['feature_engineering'] = {
        'model': feature_model,
        'predictions': feature_pred,
        'csmf_accuracy': feature_csmf
    }
    logger.info(f"Feature-enhanced XGBoost CSMF accuracy: {feature_csmf:.3f}")
    
    # 3. Prior-enhanced XGBoost (Custom Objective)
    logger.info("Training XGBoost with custom objective...")
    objective_config = XGBoostPriorConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        use_medical_priors=True,
        prior_method="custom_objective",
        lambda_prior=0.1
    )
    objective_model = XGBoostPriorEnhanced(objective_config)
    objective_model.fit(X_train, y_train)
    
    objective_pred = objective_model.predict(X_test)
    objective_csmf = objective_model.calculate_csmf_accuracy(X_test, y_test)
    results['custom_objective'] = {
        'model': objective_model,
        'predictions': objective_pred,
        'csmf_accuracy': objective_csmf
    }
    logger.info(f"Objective-enhanced XGBoost CSMF accuracy: {objective_csmf:.3f}")
    
    # 4. Prior-enhanced XGBoost (Both methods)
    logger.info("Training XGBoost with both prior methods...")
    both_config = XGBoostPriorConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        use_medical_priors=True,
        prior_method="both",
        lambda_prior=0.1,
        feature_prior_weight=1.0
    )
    both_model = XGBoostPriorEnhanced(both_config)
    both_model.fit(X_train, y_train)
    
    both_pred = both_model.predict(X_test)
    both_csmf = both_model.calculate_csmf_accuracy(X_test, y_test)
    results['both_methods'] = {
        'model': both_model,
        'predictions': both_pred,
        'csmf_accuracy': both_csmf
    }
    logger.info(f"Both-methods XGBoost CSMF accuracy: {both_csmf:.3f}")
    
    # Get prior influence report
    influence = both_model.get_prior_influence_report()
    logger.info(f"Prior influence report: {influence}")
    
    return results


def analyze_feature_importance(model, top_n=20):
    """Analyze and display feature importance.
    
    Args:
        model: Trained XGBoostPriorEnhanced model
        top_n: Number of top features to display
    """
    if not isinstance(model, XGBoostPriorEnhanced):
        logger.warning("Feature analysis only available for prior-enhanced models")
        return
        
    importance = model.get_feature_importance()
    
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    logger.info(f"\nTop {top_n} features by importance:")
    for i, (feature, score) in enumerate(sorted_features):
        logger.info(f"{i+1:2d}. {feature:40s}: {score:.4f}")
        
    # Analyze prior vs original features
    prior_features = [(f, s) for f, s in importance.items() if f.startswith("prior_")]
    original_features = [(f, s) for f, s in importance.items() if f.startswith("original_")]
    
    if prior_features:
        total_prior_importance = sum(s for _, s in prior_features)
        total_original_importance = sum(s for _, s in original_features)
        total_importance = sum(importance.values())
        
        logger.info(f"\nFeature importance breakdown:")
        logger.info(f"Prior features: {total_prior_importance/total_importance:.1%}")
        logger.info(f"Original features: {total_original_importance/total_importance:.1%}")


def main():
    """Run example comparison of XGBoost models."""
    logger.info("=== XGBoost Prior Enhancement Example ===")
    
    # Load data
    X, y = load_va_data()
    logger.info(f"Loaded data with shape: X={X.shape}, y={y.shape}")
    logger.info(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compare models
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Summary
    logger.info("\n=== Performance Summary ===")
    for method, result in results.items():
        logger.info(f"{method:20s}: CSMF = {result['csmf_accuracy']:.3f}")
        
    # Calculate improvements
    vanilla_csmf = results['vanilla']['csmf_accuracy']
    for method, result in results.items():
        if method != 'vanilla':
            improvement = (result['csmf_accuracy'] - vanilla_csmf) / vanilla_csmf * 100
            logger.info(f"{method:20s}: {improvement:+.1f}% vs vanilla")
            
    # Analyze feature importance for the combined model
    logger.info("\n=== Feature Importance Analysis ===")
    analyze_feature_importance(results['both_methods']['model'])
    
    # Show classification report for best model
    best_method = max(results.items(), key=lambda x: x[1]['csmf_accuracy'])[0]
    logger.info(f"\n=== Classification Report for Best Model ({best_method}) ===")
    print(classification_report(
        y_test, 
        results[best_method]['predictions'],
        target_names=[f"Cause_{i}" for i in range(len(np.unique(y)))]
    ))


if __name__ == "__main__":
    main()