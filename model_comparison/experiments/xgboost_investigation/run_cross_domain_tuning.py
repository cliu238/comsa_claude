#!/usr/bin/env python
"""Run true cross-domain hyperparameter tuning for XGBoost.

This script implements proper leave-one-site-out cross-validation during
hyperparameter optimization to find parameters that generalize well.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig
from baseline.models.xgboost_adaptive_config import XGBoostAdaptiveConfig
from model_comparison.hyperparameter_tuning.cross_domain_optimizer import CrossDomainOptimizer
from data.va_data_handler import VADataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_by_site(
    data_path: str,
    sites: List[str],
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Load VA data split by site.
    
    Args:
        data_path: Path to VA data CSV
        sites: List of site names to load
        
    Returns:
        Dictionary mapping site names to (X, y) tuples
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load full dataset
    va_handler = VADataHandler()
    df = va_handler.load_va_data(data_path)
    
    # Prepare data by site
    data_by_site = {}
    
    for site in sites:
        logger.info(f"Preparing data for site: {site}")
        
        # Filter data for this site
        site_df = df[df['site'] == site].copy()
        
        if len(site_df) == 0:
            logger.warning(f"No data found for site {site}")
            continue
            
        # Prepare features and labels
        X, y = va_handler.prepare_features(site_df)
        
        # Remove site column from features if present
        if 'site' in X.columns:
            X = X.drop('site', axis=1)
            
        data_by_site[site] = (X, y)
        
        logger.info(f"Site {site}: {len(X)} samples, {len(X.columns)} features")
    
    return data_by_site


def evaluate_configuration(
    config: XGBoostEnhancedConfig,
    data_by_site: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    config_name: str,
) -> Dict:
    """Evaluate a configuration using leave-one-site-out validation.
    
    Args:
        config: XGBoost configuration to evaluate
        data_by_site: Dictionary mapping site names to (X, y) tuples
        config_name: Name of the configuration for logging
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"\nEvaluating configuration: {config_name}")
    
    sites = list(data_by_site.keys())
    results = []
    
    for held_out_site in sites:
        # Split data
        train_sites = [s for s in sites if s != held_out_site]
        
        # Combine training data
        X_train_list = []
        y_train_list = []
        for site in train_sites:
            X_site, y_site = data_by_site[site]
            X_train_list.append(X_site)
            y_train_list.append(y_site)
        
        X_train = pd.concat(X_train_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)
        
        # Get test data
        X_test, y_test = data_by_site[held_out_site]
        
        # Train model
        model = XGBoostModel(config=config)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        csmf_acc = model.calculate_csmf_accuracy(y_test, y_pred)
        cod_acc = (y_test == y_pred).mean()
        
        # Also evaluate on training sites (in-domain)
        in_domain_results = []
        for train_site in train_sites:
            X_site, y_site = data_by_site[train_site]
            y_pred_site = model.predict(X_site)
            csmf_site = model.calculate_csmf_accuracy(y_site, y_pred_site)
            cod_site = (y_site == y_pred_site).mean()
            in_domain_results.append({
                'site': train_site,
                'csmf_accuracy': csmf_site,
                'cod_accuracy': cod_site,
            })
        
        avg_in_domain_csmf = np.mean([r['csmf_accuracy'] for r in in_domain_results])
        avg_in_domain_cod = np.mean([r['cod_accuracy'] for r in in_domain_results])
        
        results.append({
            'config': config_name,
            'held_out_site': held_out_site,
            'train_sites': ','.join(train_sites),
            'out_domain_csmf': csmf_acc,
            'out_domain_cod': cod_acc,
            'in_domain_csmf': avg_in_domain_csmf,
            'in_domain_cod': avg_in_domain_cod,
            'generalization_gap': avg_in_domain_csmf - csmf_acc,
        })
        
        logger.info(
            f"  {','.join(train_sites)} → {held_out_site}: "
            f"CSMF={csmf_acc:.4f} (gap={avg_in_domain_csmf - csmf_acc:.4f})"
        )
    
    # Calculate overall metrics
    overall_metrics = {
        'config': config_name,
        'avg_in_domain_csmf': np.mean([r['in_domain_csmf'] for r in results]),
        'avg_out_domain_csmf': np.mean([r['out_domain_csmf'] for r in results]),
        'avg_in_domain_cod': np.mean([r['in_domain_cod'] for r in results]),
        'avg_out_domain_cod': np.mean([r['out_domain_cod'] for r in results]),
        'avg_generalization_gap': np.mean([r['generalization_gap'] for r in results]),
        'detailed_results': results,
    }
    
    logger.info(
        f"\nOverall {config_name}:\n"
        f"  In-domain CSMF: {overall_metrics['avg_in_domain_csmf']:.4f}\n"
        f"  Out-domain CSMF: {overall_metrics['avg_out_domain_csmf']:.4f}\n"
        f"  Generalization gap: {overall_metrics['avg_generalization_gap']:.4f}"
    )
    
    return overall_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-domain hyperparameter tuning for XGBoost"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to VA data CSV file"
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        required=True,
        help="List of sites to use"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of hyperparameter tuning trials"
    )
    parser.add_argument(
        "--in-domain-weight",
        type=float,
        default=0.3,
        help="Weight for in-domain performance (0-1)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_by_site = load_data_by_site(args.data_path, args.sites)
    
    if len(data_by_site) < 2:
        logger.error("Need at least 2 sites for cross-domain tuning")
        sys.exit(1)
    
    # Initialize optimizer
    optimizer = CrossDomainOptimizer(
        sites=list(data_by_site.keys()),
        in_domain_weight=args.in_domain_weight,
        optimization_metric="csmf_accuracy",
    )
    
    # Run optimization
    logger.info(
        f"\nStarting cross-domain optimization with:\n"
        f"  Sites: {list(data_by_site.keys())}\n"
        f"  In-domain weight: {args.in_domain_weight}\n"
        f"  Out-domain weight: {1 - args.in_domain_weight}\n"
        f"  Trials: {args.n_trials}"
    )
    
    best_params, study = optimizer.optimize(
        data_by_site=data_by_site,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )
    
    # Save optimization results
    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": study.best_value,
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "user_attrs": study.best_trial.user_attrs,
            },
            "n_trials": len(study.trials),
        }, f, indent=2)
    
    # Save all trials data
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "all_trials.csv", index=False)
    
    # Create optimized configuration
    optimized_config = XGBoostEnhancedConfig(**best_params)
    
    # Compare configurations
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION COMPARISON")
    logger.info("="*60)
    
    configs_to_test = {
        "Default Enhanced": XGBoostEnhancedConfig(),
        "Conservative": XGBoostEnhancedConfig.conservative(),
        "Optimized Subsampling": XGBoostEnhancedConfig.optimized_subsampling(),
        "Cross-Domain Optimized": optimized_config,
    }
    
    all_results = []
    
    for config_name, config in configs_to_test.items():
        results = evaluate_configuration(config, data_by_site, config_name)
        all_results.append(results)
    
    # Save comparison results
    comparison_df = pd.DataFrame([
        {
            'Configuration': r['config'],
            'In-Domain CSMF': r['avg_in_domain_csmf'],
            'Out-Domain CSMF': r['avg_out_domain_csmf'],
            'Generalization Gap': r['avg_generalization_gap'],
            'In-Domain COD': r['avg_in_domain_cod'],
            'Out-Domain COD': r['avg_out_domain_cod'],
        }
        for r in all_results
    ])
    
    comparison_df.to_csv(output_dir / "configuration_comparison.csv", index=False)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    print(comparison_df.to_string(index=False))
    
    # Find best configuration
    best_idx = comparison_df['Generalization Gap'].idxmin()
    best_config = comparison_df.loc[best_idx, 'Configuration']
    
    logger.info(
        f"\nBest configuration for generalization: {best_config}\n"
        f"  Generalization gap: {comparison_df.loc[best_idx, 'Generalization Gap']:.4f}\n"
        f"  Out-domain CSMF: {comparison_df.loc[best_idx, 'Out-Domain CSMF']:.4f}"
    )
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()