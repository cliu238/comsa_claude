"""Example usage of InSilicoVA model with the baseline data pipeline.

This script demonstrates the full pipeline:
1. Loading data using VADataProcessor
2. Preprocessing for InSilicoVA compatibility  
3. Splitting data using VADataSplitter
4. Training InSilicoVA model
5. Making predictions and evaluating CSMF accuracy
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Run InSilicoVA model example."""
    
    # Step 1: Configure data loading and processing
    logger.info("=== InSilicoVA Model Example ===")
    logger.info("Step 1: Configuring data pipeline")
    
    # Use real PHMRC data (not synthetic/dummy data)
    real_data_path = "va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv"
    
    if not Path(real_data_path).exists():
        logger.error(f"Real PHMRC data not found: {real_data_path}")
        logger.error("Please ensure PHMRC data files are available")
        return 1
    
    data_config = DataConfig(
        data_path=real_data_path,
        output_dir="results/insilico_example/",
        openva_encoding=True,  # Important: InSilicoVA needs OpenVA encoding
        stratify_by_site=True,
        split_strategy="train_test",
        test_size=0.3,
        random_state=42,
        label_column="va34",  # Use numeric codes for InSilicoVA (gs_text34 dropped by OpenVA transform)
        site_column="site"
    )
    
    # Step 2: Load and preprocess data
    logger.info("Step 2: Loading and preprocessing data")
    
    processor = VADataProcessor(data_config)
    processed_data = processor.load_and_process()
    logger.info(f"Loaded {len(processed_data)} samples with {len(processed_data.columns)} columns")
    
    # Log feature exclusion results
    all_label_columns = processor._get_label_equivalent_columns(exclude_current_target=False)
    feature_exclusion_cols = processor._get_label_equivalent_columns(exclude_current_target=True)
    feature_cols = [col for col in processed_data.columns if col not in feature_exclusion_cols]
    logger.info(f"Identified {len(all_label_columns)} label-equivalent columns to prevent data leakage")
    logger.info(f"Excluded {len(feature_exclusion_cols)} columns from features")
    logger.info(f"Using {len(feature_cols)} legitimate feature columns (including target)")
    
    # Step 3: Split data
    logger.info("Step 3: Splitting data into train/test sets")
    
    splitter = VADataSplitter(data_config)
    split_result = splitter.split_data(processed_data)
    
    # Get feature columns by excluding ALL label-equivalent columns except target
    feature_exclusion_cols = processor._get_label_equivalent_columns(exclude_current_target=True)
    feature_cols = [col for col in split_result.train.columns if col not in feature_exclusion_cols]
    
    X_train = split_result.train[feature_cols]
    y_train = split_result.train[data_config.label_column]
    X_test = split_result.test[feature_cols]
    y_test = split_result.test[data_config.label_column]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Unique causes in training: {y_train.nunique()}")
    
    # Step 4: Configure and train InSilicoVA model
    logger.info("Step 4: Configuring InSilicoVA model")
    
    # Try different configurations to demonstrate flexibility
    model_configs = [
        {
            "name": "Quantile Prior (Default)",
            "config": InSilicoVAConfig(
                prior_type="quantile",
                nsim=5000,  # Reduced for faster demo
                jump_scale=0.05,
                docker_timeout=300,
                cause_column=data_config.label_column,
                verbose=True
            )
        },
        {
            "name": "Default Prior",
            "config": InSilicoVAConfig(
                prior_type="default",
                nsim=5000,
                jump_scale=0.05,
                docker_timeout=300,
                cause_column=data_config.label_column,
                verbose=True
            )
        }
    ]
    
    results = []
    
    for model_info in model_configs:
        logger.info(f"\nTesting configuration: {model_info['name']}")
        
        try:
            # Initialize model
            model = InSilicoVAModel(model_info['config'])
            
            # Fit model
            logger.info("Training InSilicoVA model...")
            model.fit(X_train, y_train)
            
            # Step 5: Make predictions
            logger.info("Making predictions on test set...")
            predictions = model.predict(X_test)
            
            # Calculate probabilities for analysis
            probabilities = model.predict_proba(X_test)
            
            # Step 6: Evaluate CSMF accuracy
            logger.info("Evaluating CSMF accuracy...")
            csmf_accuracy = model.calculate_csmf_accuracy(y_test, pd.Series(predictions))
            
            # Store results
            results.append({
                "config": model_info['name'],
                "csmf_accuracy": csmf_accuracy,
                "predictions": predictions,
                "probabilities": probabilities
            })
            
            # Display results
            logger.info(f"CSMF Accuracy: {csmf_accuracy:.3f}")
            
            # Show sample predictions
            logger.info("\nSample predictions (first 5):")
            for i in range(min(5, len(predictions))):
                logger.info(f"  True: {y_test.iloc[i]}, Predicted: {predictions[i]}")
            
            # Analyze cause distribution
            true_dist = y_test.value_counts(normalize=True).sort_index()
            pred_dist = pd.Series(predictions).value_counts(normalize=True).sort_index()
            
            logger.info("\nCause distribution comparison:")
            logger.info("Cause | True % | Pred %")
            logger.info("-" * 30)
            
            all_causes = sorted(set(true_dist.index) | set(pred_dist.index))
            for cause in all_causes[:10]:  # Show top 10 causes
                true_pct = true_dist.get(cause, 0) * 100
                pred_pct = pred_dist.get(cause, 0) * 100
                logger.info(f"{cause[:15]:15s} | {true_pct:6.1f} | {pred_pct:6.1f}")
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            logger.info("This might be due to Docker not being available or configured")
            logger.info("Please ensure Docker is installed and running")
    
    # Step 7: Compare results and benchmarks
    logger.info("\n=== Final Results Summary ===")
    
    if results:
        for result in results:
            logger.info(f"{result['config']}: CSMF Accuracy = {result['csmf_accuracy']:.3f}")
        
        # Compare against published literature benchmarks
        logger.info("\nPublished Literature Comparison:")
        logger.info("NOTE: These are target benchmarks from published papers, not our actual results")
        logger.info("OpenVA Toolkit paper reports: 0.74 ± 0.10")
        logger.info("Table 3 paper reports: 0.52-0.85 (varies by scenario)")
        
        best_accuracy = max(r['csmf_accuracy'] for r in results)
        logger.info(f"\nOur actual CSMF accuracy: {best_accuracy:.3f}")
        logger.info("This is the real performance on PHMRC data with proper feature exclusion")
        logger.info("(Previous claims of matching published benchmarks were not empirically validated)")
    
    logger.info("\n=== Example Complete ===")
    
    # Return 0 for success
    return 0


def create_dummy_va_data():
    """Create a dummy CSV file for demonstration."""
    import tempfile
    
    # Create minimal dummy data
    data = {
        'site': ['Site_A'] * 10,
        'gs_text34': ['Cause_1'] * 5 + ['Cause_2'] * 5,
        'symptom_1': [1, 0, 1, 0, 1] * 2,
        'symptom_2': [0, 1, 0, 1, 0] * 2,
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name


def create_synthetic_va_data(n_samples=200, n_features=50, n_causes=10):
    """Create synthetic VA data for demonstration."""
    np.random.seed(42)
    
    # Create symptom features (binary)
    features = {}
    for i in range(n_features):
        features[f'symptom_{i:03d}'] = np.random.binomial(1, 0.3, n_samples)
    
    # Create sites
    sites = [f'Site_{chr(65+i)}' for i in range(4)]
    features['site'] = np.random.choice(sites, n_samples)
    
    # Create causes with some realistic distribution
    causes = [f'Cause_{i:02d}' for i in range(1, n_causes + 1)]
    cause_probs = np.random.dirichlet(np.ones(n_causes) * 2)
    features['gs_text34'] = np.random.choice(causes, n_samples, p=cause_probs)
    
    df = pd.DataFrame(features)
    
    # Add some NA values
    mask = np.random.random(df.iloc[:, :-2].shape) < 0.05
    df.iloc[:, :-2][mask] = np.nan
    
    return df


if __name__ == "__main__":
    sys.exit(main())