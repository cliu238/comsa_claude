#!/usr/bin/env python
"""Run InSilicoVA on full dataset with proper timeout settings.

This script is designed for production runs on the complete PHMRC dataset.
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

def main():
    """Run InSilicoVA on full dataset."""
    
    # Use real PHMRC data path
    real_data_path = "va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
    
    if not Path(real_data_path).exists():
        logger.error(f"Real PHMRC data not found: {real_data_path}")
        return 1
    
    logger.info("=== InSilicoVA Full Dataset Run ===")
    logger.info("⚠️  This will take 30-60 minutes to complete")
    
    # Configure data loading
    data_config = DataConfig(
        data_path=real_data_path,
        output_dir="results/insilico_full_run/",
        openva_encoding=True,
        split_strategy="train_test",
        test_size=0.2,  # Use 20% for testing (more reasonable for full run)
        random_state=42,
        label_column="va34",
        site_column="site"
    )
    
    logger.info("Step 1: Loading and preprocessing data")
    processor = VADataProcessor(data_config)
    processed_data = processor.load_and_process()
    
    logger.info(f"Total samples: {len(processed_data)}")
    logger.info(f"Total columns: {len(processed_data.columns)}")
    
    # Split data
    logger.info("Step 2: Splitting data")
    splitter = VADataSplitter(data_config)
    split_result = splitter.split_data(processed_data)
    
    # Get feature columns by excluding ALL label-equivalent columns except target
    feature_exclusion_cols = processor._get_label_equivalent_columns(exclude_current_target=True)
    feature_cols = [col for col in split_result.train.columns if col not in feature_exclusion_cols]
    
    X_train = split_result.train[feature_cols]
    y_train = split_result.train[data_config.label_column]
    X_test = split_result.test[feature_cols]
    y_test = split_result.test[data_config.label_column]
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Unique causes in training: {y_train.nunique()}")
    
    # Configure InSilicoVA model for full dataset
    logger.info("Step 3: Configuring InSilicoVA model for full dataset")
    
    config = InSilicoVAConfig(
        nsim=5000,  # Full MCMC iterations
        docker_timeout=3600,  # 1 hour timeout
        cause_column=data_config.label_column,
        verbose=True
    )
    
    # Initialize and train model
    logger.info("Step 4: Training InSilicoVA model...")
    logger.info("⚠️  This may take 10-20 minutes...")
    
    model = InSilicoVAModel(config)
    model.fit(X_train, y_train)
    
    # Make predictions
    logger.info("Step 5: Making predictions on full test set...")
    logger.info("⚠️  This may take 30-60 minutes...")
    
    predictions = model.predict(X_test)
    
    # Calculate CSMF accuracy
    logger.info("Step 6: Calculating CSMF accuracy...")
    import pandas as pd
    import json
    from datetime import datetime
    
    csmf_accuracy = model.calculate_csmf_accuracy(y_test, pd.Series(predictions))
    
    logger.info(f"\\n=== FULL DATASET RESULTS ===")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"CSMF Accuracy: {csmf_accuracy:.3f}")
    
    # Show sample predictions
    logger.info("\\nSample predictions (first 10):")
    for i in range(min(10, len(predictions))):
        logger.info(f"  True: {y_test.iloc[i]}, Predicted: {predictions[i]}")
    
    # Show cause distribution comparison
    # Convert both to string for consistent comparison
    true_dist = y_test.astype(str).value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(predictions).astype(str).value_counts(normalize=True).sort_index()
    
    logger.info("\\nTop 10 causes distribution comparison:")
    logger.info("Cause | True % | Pred %")
    logger.info("-" * 25)
    
    all_causes = sorted(set(true_dist.index) | set(pred_dist.index))
    for cause in all_causes[:10]:
        true_pct = true_dist.get(cause, 0) * 100
        pred_pct = pred_dist.get(cause, 0) * 100
        logger.info(f"{str(cause)[:10]:10s} | {true_pct:5.1f} | {pred_pct:5.1f}")
    
    # Save results to files
    logger.info("\\nStep 7: Saving results to files...")
    
    # Create results directory if it doesn't exist
    results_dir = Path(data_config.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save individual predictions
    predictions_df = pd.DataFrame({
        'true_cause': y_test.astype(str).values,
        'predicted_cause': predictions,
        'test_index': y_test.index
    })
    predictions_file = results_dir / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to: {predictions_file}")
    
    # 2. Save benchmark results summary
    benchmark_results = pd.DataFrame({
        'metric': ['CSMF_accuracy', 'train_samples', 'test_samples', 'n_features', 'n_causes'],
        'value': [csmf_accuracy, len(X_train), len(X_test), len(feature_cols), y_train.nunique()]
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
            'total_samples': len(X_train) + len(X_test),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
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
    
    logger.info("\\n=== FULL DATASET RUN COMPLETE ===")
    logger.info(f"✅ Final CSMF Accuracy: {csmf_accuracy:.3f}")
    logger.info(f"✅ Results saved to: {results_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())