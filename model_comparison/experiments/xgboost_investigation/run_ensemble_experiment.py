#!/usr/bin/env python
"""Run ensemble XGBoost experiment with site-specific models.

This experiment trains separate XGBoost models for each source site
and combines their predictions using weighted voting based on
domain similarity.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline.data_loader import VADataLoader
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.models.xgboost_adaptive_config import XGBoostAdaptiveConfig
from model_comparison.metrics.comparison_metrics import calculate_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostEnsemble:
    """Ensemble of site-specific XGBoost models."""
    
    def __init__(self, use_adaptive: bool = True):
        """Initialize ensemble.
        
        Args:
            use_adaptive: Whether to use adaptive configurations
        """
        self.use_adaptive = use_adaptive
        self.models: Dict[str, XGBoostModel] = {}
        self.site_weights: Dict[str, Dict[str, float]] = {}
        
    def fit(self, data_by_site: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
        """Train site-specific models.
        
        Args:
            data_by_site: Dictionary mapping site names to (X, y) tuples
        """
        logger.info(f"Training ensemble with {len(data_by_site)} site-specific models")
        
        for site, (X, y) in data_by_site.items():
            logger.info(f"Training model for site {site} with {len(X)} samples")
            
            if self.use_adaptive:
                # Analyze data and create adaptive config
                characteristics = XGBoostAdaptiveConfig.analyze_data_characteristics(X, y)
                config = XGBoostAdaptiveConfig.from_data_characteristics(
                    n_samples=characteristics["n_samples"],
                    n_classes=characteristics["n_classes"],
                    class_imbalance_ratio=characteristics["class_imbalance_ratio"],
                    site_name=site
                )
            else:
                # Use conservative config for all sites
                config = XGBoostEnhancedConfig.conservative()
            
            model = XGBoostModel(config=config)
            model.fit(X, y)
            self.models[site] = model
            
        # Calculate site similarity weights based on performance
        self._calculate_site_weights(data_by_site)
        
    def _calculate_site_weights(self, data_by_site: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
        """Calculate weights for combining predictions from different sites.
        
        Uses cross-site validation performance to weight models.
        """
        logger.info("Calculating site weights based on cross-validation")
        
        # For each pair of sites, evaluate how well model from site A
        # performs on data from site B
        performance_matrix = {}
        
        for source_site, source_model in self.models.items():
            performance_matrix[source_site] = {}
            
            for target_site, (X_target, y_target) in data_by_site.items():
                if source_site == target_site:
                    # In-domain performance gets bonus weight
                    performance_matrix[source_site][target_site] = 1.0
                else:
                    # Evaluate out-domain performance
                    y_pred = source_model.predict(X_target)
                    csmf_acc = source_model.calculate_csmf_accuracy(y_target, y_pred)
                    performance_matrix[source_site][target_site] = csmf_acc
        
        # Convert performance to weights (normalize by target site)
        self.site_weights = {}
        for target_site in data_by_site.keys():
            weights = {}
            total = 0
            
            for source_site in self.models.keys():
                weight = performance_matrix[source_site][target_site]
                weights[source_site] = weight
                total += weight
                
            # Normalize weights
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            else:
                # Equal weights if all perform poorly
                weights = {k: 1.0/len(self.models) for k in self.models.keys()}
                
            self.site_weights[target_site] = weights
            
        logger.info("Site weights calculated")
        for target, weights in self.site_weights.items():
            logger.info(f"  {target}: {weights}")
            
    def predict(self, X: pd.DataFrame, target_site: str = None) -> np.ndarray:
        """Make predictions using weighted ensemble.
        
        Args:
            X: Features to predict
            target_site: Target site (for site-specific weighting)
            
        Returns:
            Predicted labels
        """
        if not self.models:
            raise ValueError("Ensemble not fitted")
            
        # Get predictions from all models
        all_predictions = []
        all_probas = []
        
        for site, model in self.models.items():
            y_pred = model.predict(X)
            all_predictions.append(y_pred)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                all_probas.append(y_proba)
                
        # If we have probability predictions, use weighted average
        if all_probas and target_site and target_site in self.site_weights:
            weights = self.site_weights[target_site]
            weight_array = np.array([weights.get(site, 1.0/len(self.models)) 
                                    for site in self.models.keys()])
            
            # Weighted average of probabilities
            weighted_proba = np.zeros_like(all_probas[0])
            for i, (site, proba) in enumerate(zip(self.models.keys(), all_probas)):
                weighted_proba += weight_array[i] * proba
                
            # Return class with highest weighted probability
            return self.models[list(self.models.keys())[0]].classes_[np.argmax(weighted_proba, axis=1)]
        else:
            # Fall back to majority voting
            predictions_array = np.array(all_predictions)
            # Mode across models
            from scipy import stats
            return stats.mode(predictions_array, axis=0)[0].flatten()


def main():
    """Run ensemble experiment."""
    parser = argparse.ArgumentParser(description="Run XGBoost ensemble experiment")
    parser.add_argument("--data-path", required=True, help="Path to VA data")
    parser.add_argument("--sites", nargs="+", required=True, help="Sites to include")
    parser.add_argument("--n-bootstrap", type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    loader = VADataLoader(args.data_path)
    data = loader.load_data()
    
    # Filter to specified sites
    data = data[data["site"].isin(args.sites)]
    logger.info(f"Filtered to {len(data)} samples from sites: {args.sites}")
    
    # Prepare data by site
    data_by_site = {}
    for site in args.sites:
        site_data = data[data["site"] == site]
        if len(site_data) < 50:
            logger.warning(f"Skipping site {site} with only {len(site_data)} samples")
            continue
            
        # Prepare features and labels
        X, y = loader.prepare_features_labels(site_data)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
        )
        
        data_by_site[site] = (X_train, y_train)
        
    # Train ensemble
    logger.info("Training ensemble models")
    ensemble = XGBoostEnsemble(use_adaptive=True)
    ensemble.fit(data_by_site)
    
    # Evaluate on each site
    results = []
    
    for test_site in args.sites:
        logger.info(f"Evaluating on site {test_site}")
        
        # Get test data for this site
        site_data = data[data["site"] == test_site]
        if len(site_data) < 50:
            continue
            
        X, y = loader.prepare_features_labels(site_data)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
        )
        
        # Make predictions
        y_pred_ensemble = ensemble.predict(X_test, target_site=test_site)
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred_ensemble,
            n_bootstrap=args.n_bootstrap
        )
        
        # Also evaluate individual site models for comparison
        for train_site in data_by_site.keys():
            y_pred_single = ensemble.models[train_site].predict(X_test)
            metrics_single = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred_single,
                n_bootstrap=args.n_bootstrap
            )
            
            results.append({
                "model": f"xgboost_single_{train_site}",
                "experiment_type": "in_domain" if train_site == test_site else "out_domain",
                "train_site": train_site,
                "test_site": test_site,
                "csmf_accuracy": metrics_single["csmf_accuracy"],
                "cod_accuracy": metrics_single["cod_accuracy"],
                "n_train": len(data_by_site[train_site][1]),
                "n_test": len(y_test),
            })
        
        # Ensemble results
        results.append({
            "model": "xgboost_ensemble",
            "experiment_type": "ensemble",
            "train_site": "all",
            "test_site": test_site,
            "csmf_accuracy": metrics["csmf_accuracy"],
            "cod_accuracy": metrics["cod_accuracy"],
            "n_train": sum(len(y) for _, y in data_by_site.values()),
            "n_test": len(y_test),
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "va34_comparison_results.csv", index=False)
    
    # Calculate summary statistics
    logger.info("\n=== Ensemble Performance Summary ===")
    
    # Single model out-domain performance
    single_out = results_df[
        (results_df["model"].str.startswith("xgboost_single")) & 
        (results_df["experiment_type"] == "out_domain")
    ]["csmf_accuracy"].mean()
    
    # Ensemble performance
    ensemble_perf = results_df[
        results_df["model"] == "xgboost_ensemble"
    ]["csmf_accuracy"].mean()
    
    logger.info(f"Average single model out-domain CSMF: {single_out:.4f}")
    logger.info(f"Ensemble average CSMF: {ensemble_perf:.4f}")
    logger.info(f"Improvement: {(ensemble_perf - single_out) / single_out * 100:.1f}%")
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()