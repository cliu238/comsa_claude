#!/usr/bin/env python
"""Run InSilicoVA with AP-only testing to match R Journal 2023 methodology.

This script replicates the exact experimental setup from the R Journal 2023 paper:
- Training: 5 sites excluding Andhra Pradesh (Mexico, Dar, UP, Bohol, Pemba)
- Testing: Andhra Pradesh (AP) only
- Expected CSMF accuracy: ~0.74 (geographic generalization evaluation)
Expected runtime: 30-60 minutes depending on system performance.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add va-data to path
va_data_path = Path(__file__).parent.parent / "va-data"
if va_data_path.exists():
    sys.path.insert(0, str(va_data_path))

from baseline.config.data_config import DataConfig
from baseline.data.data_loader_preprocessor import VADataProcessor
from baseline.data.data_splitter import VADataSplitter
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.model_config import InSilicoVAConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compare_with_mixed_site_results(ap_results_dir: Path) -> None:
    """Compare AP-only results with existing mixed-site results.
    
    Args:
        ap_results_dir: Directory containing AP-only results
    """
    import pandas as pd
    import json
    
    logger.info("Step 8: Comparing with mixed-site results...")
    
    # Paths to results
    mixed_site_dir = Path("results/insilico_full_run")
    ap_benchmark_file = ap_results_dir / "benchmark_results.csv"
    mixed_benchmark_file = mixed_site_dir / "benchmark_results.csv"
    
    # Check if mixed-site results exist
    if not mixed_benchmark_file.exists():
        logger.warning("Mixed-site results not found. Skipping comparison.")
        return
    
    try:
        # Load benchmark results
        ap_results = pd.read_csv(ap_benchmark_file)
        mixed_results = pd.read_csv(mixed_benchmark_file)
        
        # Extract CSMF accuracies
        ap_csmf = ap_results[ap_results['metric'] == 'CSMF_accuracy']['value'].iloc[0]
        mixed_csmf = mixed_results[mixed_results['metric'] == 'CSMF_accuracy']['value'].iloc[0]
        
        # Create comparison summary
        comparison = {
            'methodology_comparison': {
                'mixed_site_evaluation': {
                    'description': 'Within-distribution testing (easier)',
                    'train_test_split': 'All 6 sites mixed (stratified)',
                    'csmf_accuracy': float(mixed_csmf),
                    'evaluation_type': 'Internal validity'
                },
                'ap_only_evaluation': {
                    'description': 'Geographic generalization (harder)', 
                    'train_test_split': 'Train: 5 sites, Test: AP only',
                    'csmf_accuracy': float(ap_csmf),
                    'evaluation_type': 'External validity'
                },
                'r_journal_2023_benchmark': {
                    'reported_csmf_accuracy': 0.74,
                    'methodology': 'Same as AP-only evaluation',
                    'reference': 'https://journal.r-project.org/articles/RJ-2023-020/'
                },
                'analysis': {
                    'performance_difference': float(mixed_csmf - ap_csmf),
                    'expected_pattern': 'Mixed-site > AP-only (geographic generalization is harder)',
                    'literature_validation': 'AP-only accuracy within ±0.05 of R Journal 2023 (0.74)',
                    'implementation_status': 'Validated' if abs(ap_csmf - 0.74) <= 0.05 else 'Needs review'
                }
            }
        }
        
        # Save comparison
        comparison_file = ap_results_dir / "methodology_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved methodology comparison to: {comparison_file}")
        
        # Log comparison results
        logger.info("\\n=== METHODOLOGY COMPARISON ===")
        logger.info(f"Mixed-site CSMF accuracy: {mixed_csmf:.3f} (within-distribution)")
        logger.info(f"AP-only CSMF accuracy: {ap_csmf:.3f} (geographic generalization)")
        logger.info("R Journal 2023 benchmark: 0.740 (same methodology as AP-only)")
        logger.info(f"Performance difference: {mixed_csmf - ap_csmf:.3f}")
        
        # Validation checks
        if mixed_csmf > ap_csmf:
            logger.info("✅ Expected pattern: Mixed-site > AP-only (geographic generalization is harder)")
        else:
            logger.warning("⚠️  Unexpected: AP-only >= Mixed-site (investigate)")
            
        if abs(ap_csmf - 0.74) <= 0.05:
            logger.info("✅ AP-only result validates against R Journal 2023 benchmark")
        else:
            logger.warning(f"⚠️  AP-only result differs from R Journal 2023: {abs(ap_csmf - 0.74):.3f}")
            
    except Exception as e:
        logger.error(f"Error in methodology comparison: {str(e)}")

def main() -> int:
    """Run InSilicoVA with AP-only testing to match R Journal 2023 methodology."""
    
    # Use real PHMRC data path
    real_data_path = "va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
    
    if not Path(real_data_path).exists():
        logger.error(f"Real PHMRC data not found: {real_data_path}")
        return 1
    
    logger.info("=== InSilicoVA AP-Only Evaluation (R Journal 2023 Replication) ===")
    logger.info("📋 Methodology: Geographic generalization evaluation")
    logger.info("🎯 Target: CSMF accuracy ~0.74 (R Journal 2023 benchmark)")
    logger.info("⚠️  This will take 30-60 minutes to complete")
    
    # Configure data loading with cross-site strategy
    data_config = DataConfig(
        data_path=real_data_path,
        output_dir="results/ap_only_insilico/",
        openva_encoding=True,
        stratify_by_site=True,  # Enable site-based stratification
        split_strategy="cross_site",  # KEY CHANGE: Use cross-site strategy
        train_sites=["Mexico", "Dar", "UP", "Bohol", "Pemba"],  # 5 sites for training
        test_sites=["AP"],  # AP only for testing
        random_state=42,
        label_column="va34",
        site_column="site"
    )
    
    logger.info("Step 1: Loading and preprocessing data")
    processor = VADataProcessor(data_config)
    processed_data = processor.load_and_process()
    
    logger.info(f"Total samples: {len(processed_data)}")
    logger.info(f"Total columns: {len(processed_data.columns)}")
    
    # Split data using cross-site strategy
    logger.info("Step 2: Splitting data (cross-site: AP-only testing)")
    splitter = VADataSplitter(data_config)
    split_result = splitter.split_data(processed_data)
    
    # Validate split results against R Journal 2023 expected sizes
    expected_train_size = 6287  # R Journal 2023 reported size
    expected_test_size = 1554   # R Journal 2023 reported size
    
    train_size = len(split_result.train)
    test_size = len(split_result.test)
    
    logger.info(f"Training samples: {train_size} (expected: ~{expected_train_size})")
    logger.info(f"Test samples: {test_size} (expected: ~{expected_test_size})")
    
    # Validate sample sizes are in expected range
    if not (expected_train_size - 200 <= train_size <= expected_train_size + 200):
        logger.warning(f"Training size {train_size} differs significantly from R Journal 2023: {expected_train_size}")
    if not (expected_test_size - 100 <= test_size <= expected_test_size + 100):
        logger.warning(f"Test size {test_size} differs significantly from R Journal 2023: {expected_test_size}")
    
    # Get feature columns by excluding ALL label-equivalent columns except target
    feature_exclusion_cols = processor._get_label_equivalent_columns(exclude_current_target=True)
    feature_cols = [col for col in split_result.train.columns if col not in feature_exclusion_cols]
    
    X_train = split_result.train[feature_cols]
    y_train = split_result.train[data_config.label_column]
    X_test = split_result.test[feature_cols]
    y_test = split_result.test[data_config.label_column]
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Unique causes in training: {y_train.nunique()}")
    logger.info(f"Unique causes in test: {y_test.nunique()}")
    
    # Verify site assignments
    train_sites = set(split_result.train[data_config.site_column].unique())
    test_sites = set(split_result.test[data_config.site_column].unique())
    
    logger.info(f"Training sites: {sorted(train_sites)}")
    logger.info(f"Test sites: {sorted(test_sites)}")
    
    # Validate site assignments match R Journal 2023
    expected_train_sites = {"Mexico", "Dar", "UP", "Bohol", "Pemba"}
    expected_test_sites = {"AP"}
    
    if train_sites != expected_train_sites:
        logger.error(f"Training sites mismatch. Expected: {expected_train_sites}, Got: {train_sites}")
        return 1
    if test_sites != expected_test_sites:
        logger.error(f"Test sites mismatch. Expected: {expected_test_sites}, Got: {test_sites}")
        return 1
        
    logger.info("✅ Site assignments match R Journal 2023 methodology")
    
    # Configure InSilicoVA model for full dataset
    logger.info("Step 3: Configuring InSilicoVA model for AP-only evaluation")
    
    config = InSilicoVAConfig(
        nsim=5000,  # Full MCMC iterations
        docker_timeout=3600,  # 1 hour timeout
        cause_column=data_config.label_column,
        verbose=True
    )
    
    # Initialize and train model
    logger.info("Step 4: Training InSilicoVA model on 5 sites (excluding AP)...")
    logger.info("⚠️  This may take 10-20 minutes...")
    
    model = InSilicoVAModel(config)
    model.fit(X_train, y_train)
    
    # Make predictions
    logger.info("Step 5: Making predictions on AP-only test set...")
    logger.info("⚠️  This may take 30-60 minutes...")
    
    predictions = model.predict(X_test)
    
    # Calculate CSMF accuracy
    logger.info("Step 6: Calculating CSMF accuracy...")
    import pandas as pd
    import json
    from datetime import datetime
    
    csmf_accuracy = model.calculate_csmf_accuracy(y_test, pd.Series(predictions))
    
    logger.info("\\n=== AP-ONLY EVALUATION RESULTS ===")
    logger.info(f"Training samples: {len(X_train)} (5 sites: Mexico, Dar, UP, Bohol, Pemba)")
    logger.info(f"Test samples: {len(X_test)} (1 site: AP)")
    logger.info(f"CSMF Accuracy: {csmf_accuracy:.3f}")
    logger.info("R Journal 2023 benchmark: 0.740")
    logger.info(f"Difference from benchmark: {abs(csmf_accuracy - 0.74):.3f}")
    
    # Show sample predictions
    logger.info("\\nSample predictions (first 10):")
    for i in range(min(10, len(predictions))):
        logger.info(f"  True: {y_test.iloc[i]}, Predicted: {predictions[i]}")
    
    # Show cause distribution comparison
    # Convert both to string for consistent comparison
    true_dist = y_test.astype(str).value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(predictions).astype(str).value_counts(normalize=True).sort_index()
    
    logger.info("\\nTop 10 causes distribution comparison (AP-only):")
    logger.info("Cause | True % | Pred %")
    logger.info("-" * 25)
    
    all_causes = sorted(set(true_dist.index) | set(pred_dist.index))
    for cause in all_causes[:10]:
        true_pct = true_dist.get(cause, 0) * 100
        pred_pct = pred_dist.get(cause, 0) * 100
        logger.info(f"{str(cause)[:10]:10s} | {true_pct:5.1f} | {pred_pct:5.1f}")
    
    # Save results to files
    logger.info("\\nStep 7: Saving AP-only results to files...")
    
    # Create results directory if it doesn't exist
    results_dir = Path(data_config.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save individual predictions
    predictions_df = pd.DataFrame({
        'true_cause': y_test.astype(str).values,
        'predicted_cause': predictions,
        'test_index': y_test.index,
        'test_site': 'AP'  # All test samples are from AP
    })
    predictions_file = results_dir / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to: {predictions_file}")
    
    # 2. Save benchmark results summary
    benchmark_results = pd.DataFrame({
        'metric': ['CSMF_accuracy', 'train_samples', 'test_samples', 'n_features', 'n_causes', 'train_sites', 'test_sites'],
        'value': [csmf_accuracy, len(X_train), len(X_test), len(feature_cols), y_train.nunique(), 5, 1]
    })
    benchmark_file = results_dir / "benchmark_results.csv"
    benchmark_results.to_csv(benchmark_file, index=False)
    logger.info(f"Saved benchmark results to: {benchmark_file}")
    
    # 3. Save cause distribution comparison
    cause_dist_df = pd.DataFrame({
        'cause': all_causes,
        'true_percentage': [true_dist.get(cause, 0) * 100 for cause in all_causes],
        'predicted_percentage': [pred_dist.get(cause, 0) * 100 for cause in all_causes]
    })
    cause_dist_file = results_dir / "cause_distribution.csv"
    cause_dist_df.to_csv(cause_dist_file, index=False)
    logger.info(f"Saved cause distribution to: {cause_dist_file}")
    
    # 4. Save complete results as JSON
    full_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': data_config.data_path,
            'methodology': 'AP-only testing (R Journal 2023 replication)',
            'evaluation_type': 'Geographic generalization',
            'total_samples': len(X_train) + len(X_test),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_sites': sorted(train_sites),
            'test_sites': sorted(test_sites),
            'n_features': len(feature_cols),
            'n_causes': y_train.nunique(),
            'model_type': 'InSilicoVA'
        },
        'model_config': {
            'nsim': config.nsim,
            'jump_scale': config.jump_scale,
            'auto_length': config.auto_length,
            'convert_type': config.convert_type,
            'cause_column': config.cause_column,
            'phmrc_type': config.phmrc_type,
            'docker_timeout': config.docker_timeout
        },
        'results': {
            'csmf_accuracy': float(csmf_accuracy),
            'r_journal_2023_benchmark': 0.74,
            'benchmark_difference': float(abs(csmf_accuracy - 0.74)),
            'predictions': predictions_df.to_dict('records'),
            'cause_distribution': cause_dist_df.to_dict('records'),
            'sample_predictions': [
                {'true': str(y_test.iloc[i]), 'predicted': str(predictions[i])} 
                for i in range(min(10, len(predictions)))
            ]
        }
    }
    
    json_file = results_dir / "full_results.json"
    with open(json_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Saved complete results to: {json_file}")
    
    # Compare with mixed-site results
    compare_with_mixed_site_results(results_dir)
    
    logger.info("\\n=== AP-ONLY EVALUATION COMPLETE ===")
    logger.info(f"✅ Final CSMF Accuracy: {csmf_accuracy:.3f}")
    logger.info("📊 R Journal 2023 benchmark: 0.740")
    logger.info(f"📈 Benchmark validation: {'✅ PASSED' if abs(csmf_accuracy - 0.74) <= 0.05 else '⚠️ REVIEW NEEDED'}")
    logger.info(f"✅ Results saved to: {results_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())