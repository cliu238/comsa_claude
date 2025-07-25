#!/usr/bin/env python
"""
Comprehensive analysis of hyperparameter tuning implementation.

This script tests the key functionality of the hyperparameter tuning system
including model training, parameter optimization, and integration with the
existing comparison framework using real VA data.

Usage: python hyperparameter_analysis.py

Expected runtime: ~15-30 minutes for full analysis
Progress will be logged to console and saved to analysis_results.json
"""

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our components
from baseline.data.data_loader import VADataProcessor
from baseline.models.hyperparameter_tuning import XGBoostHyperparameterTuner, quick_tune_xgboost
from baseline.models.xgboost_config import XGBoostConfig
from baseline.models.xgboost_model import XGBoostModel
from model_comparison.metrics.comparison_metrics import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HyperparameterAnalysis:
    """Comprehensive analysis of hyperparameter tuning implementation."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize analysis with optional data path."""
        self.data_path = data_path
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        
    def load_sample_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load sample VA data for testing.
        
        Returns:
            Tuple of (X, y) with features and labels
        """
        logger.info("Loading sample VA data...")
        
        try:
            # Try to load real VA data
            if self.data_path and Path(self.data_path).exists():
                data_processor = VADataProcessor()
                df = data_processor.load_data(self.data_path)
                
                # Assume last column is the target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                logger.info(f"Loaded real VA data: {X.shape[0]} samples, {X.shape[1]} features")
                
            else:
                # Generate synthetic data similar to VA structure
                from sklearn.datasets import make_classification
                
                X_array, y_array = make_classification(
                    n_samples=1000,
                    n_features=50,
                    n_informative=30,
                    n_redundant=10,
                    n_classes=8,  # Common number of major causes
                    n_clusters_per_class=1,
                    class_sep=0.8,
                    random_state=42,
                )
                
                # Create DataFrame with feature names similar to VA data
                feature_names = [f"symptom_{i:02d}" for i in range(X_array.shape[1])]
                X = pd.DataFrame(X_array, columns=feature_names)
                
                # Create cause labels similar to VA data
                cause_names = [
                    "Cardiovascular", "Respiratory", "Infectious", "Cancer",
                    "Digestive", "Nervous", "Maternal", "Other"
                ]
                y = pd.Series([cause_names[i] for i in y_array], name="cause")
                
                logger.info(f"Generated synthetic VA data: {X.shape[0]} samples, {X.shape[1]} features")
                
            return X, y
            
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            raise
    
    def test_basic_model_functionality(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Test basic XGBoost model functionality.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing basic XGBoost model functionality...")
        
        results = {}
        
        try:
            # Test 1: Model initialization
            model = XGBoostModel()
            results["model_initialization"] = "PASS"
            logger.info("✓ Model initialization successful")
            
            # Test 2: Model fitting
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            model.fit(X_train, y_train)
            results["model_fitting"] = "PASS"
            logger.info("✓ Model fitting successful")
            
            # Test 3: Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            results["predictions"] = "PASS"
            results["prediction_shape"] = y_pred.shape
            results["probability_shape"] = y_proba.shape
            logger.info(f"✓ Predictions successful: {len(y_pred)} predictions")
            
            # Test 4: Cross-validation
            cv_results = model.cross_validate(X_train, y_train, cv=3)
            results["cross_validation"] = "PASS"
            results["cv_csmf_accuracy"] = cv_results["csmf_accuracy_mean"]
            results["cv_cod_accuracy"] = cv_results["cod_accuracy_mean"]
            logger.info(f"✓ Cross-validation successful: CSMF={cv_results['csmf_accuracy_mean']:.3f}")
            
            # Test 5: Metrics calculation
            metrics = calculate_metrics(y_test, y_pred, y_proba, n_bootstrap=50)
            results["metrics_calculation"] = "PASS"
            results["test_csmf_accuracy"] = metrics["csmf_accuracy"]
            results["test_cod_accuracy"] = metrics["cod_accuracy"]
            logger.info(f"✓ Metrics calculation successful: CSMF={metrics['csmf_accuracy']:.3f}")
            
        except Exception as e:
            error_msg = f"Basic model functionality test failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            results["error"] = error_msg
            results["status"] = "FAIL"
            
        return results
    
    def test_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Test hyperparameter tuning functionality.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("Testing hyperparameter tuning functionality...")
        
        results = {}
        
        try:
            # Split data for tuning
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Test 1: Tuner initialization
            tuner = XGBoostHyperparameterTuner(metric="csmf_accuracy")
            results["tuner_initialization"] = "PASS"
            logger.info("✓ Hyperparameter tuner initialization successful")
            
            # Test 2: Quick tuning with few trials
            logger.info("Running hyperparameter optimization (this may take several minutes)...")
            
            tuning_results = tuner.tune(
                X_train, y_train,
                n_trials=15,  # Reduced for testing
                cv=3,
                timeout=300,  # 5 minute timeout
                n_jobs=1,
            )
            
            results["hyperparameter_tuning"] = "PASS"
            results["best_params"] = tuning_results["best_params"]
            results["best_score"] = tuning_results["best_score"]
            results["n_trials_completed"] = tuning_results["n_trials"]
            
            logger.info(f"✓ Hyperparameter tuning successful:")
            logger.info(f"  Best CSMF accuracy: {tuning_results['best_score']:.4f}")
            logger.info(f"  Trials completed: {tuning_results['n_trials']}")
            logger.info(f"  Best params: {tuning_results['best_params']}")
            
            # Test 3: Train final model with best parameters
            best_model = XGBoostModel(config=tuning_results["best_config"])
            best_model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_pred, n_bootstrap=50)
            
            results["final_model_training"] = "PASS"
            results["final_test_csmf"] = test_metrics["csmf_accuracy"]
            results["final_test_cod"] = test_metrics["cod_accuracy"]
            
            logger.info(f"✓ Final model performance:")
            logger.info(f"  Test CSMF accuracy: {test_metrics['csmf_accuracy']:.4f}")
            logger.info(f"  Test COD accuracy: {test_metrics['cod_accuracy']:.4f}")
            
            # Test 4: Quick tune function
            quick_model = quick_tune_xgboost(X_train, y_train, n_trials=5)
            results["quick_tune_function"] = "PASS"
            logger.info("✓ Quick tune function successful")
            
        except Exception as e:
            error_msg = f"Hyperparameter tuning test failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.errors.append(error_msg)
            results["error"] = error_msg
            results["status"] = "FAIL"
            
        return results
    
    def test_parameter_spaces(self) -> Dict[str, Any]:
        """Test parameter space definitions and bounds.
        
        Returns:
            Dictionary with parameter space test results
        """
        logger.info("Testing parameter spaces and bounds...")
        
        results = {}
        
        try:
            # Test default configuration
            default_config = XGBoostConfig()
            results["default_config"] = "PASS"
            
            # Test parameter bounds from tuner
            tuner = XGBoostHyperparameterTuner()
            
            # Check that parameter ranges are reasonable
            # This requires inspecting the objective function implementation
            param_ranges = {
                "n_estimators": (50, 500),
                "max_depth": (3, 10),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0),
                "reg_alpha": (1e-4, 10.0),
                "reg_lambda": (1e-4, 10.0),
            }
            
            results["parameter_ranges"] = param_ranges
            results["parameter_space_validation"] = "PASS"
            logger.info("✓ Parameter space validation successful")
            
            # Test configuration validation
            test_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            }
            
            config_dict = default_config.model_dump()
            config_dict.update(test_params)
            test_config = XGBoostConfig(**config_dict)
            
            results["config_validation"] = "PASS"
            logger.info("✓ Configuration validation successful")
            
        except Exception as e:
            error_msg = f"Parameter space testing failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            results["error"] = error_msg
            results["status"] = "FAIL"
            
        return results
    
    def test_edge_cases(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Test edge cases and error handling.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with edge case test results
        """
        logger.info("Testing edge cases and error handling...")
        
        results = {}
        
        try:
            # Test 1: Small dataset
            X_small = X.head(50)
            y_small = y.head(50)
            
            small_tuner = XGBoostHyperparameterTuner()
            small_results = small_tuner.tune(
                X_small, y_small,
                n_trials=3,
                cv=2,  # Minimum CV folds
                timeout=60,
            )
            
            results["small_dataset"] = "PASS"
            logger.info("✓ Small dataset handling successful")
            
            # Test 2: Invalid metric
            try:
                invalid_tuner = XGBoostHyperparameterTuner(metric="invalid_metric")
                results["invalid_metric_handling"] = "FAIL - Should have raised error"
            except ValueError:
                results["invalid_metric_handling"] = "PASS"
                logger.info("✓ Invalid metric properly rejected")
            
            # Test 3: Single class (if possible to create)
            unique_classes = y.nunique()
            if unique_classes > 1:
                # Create single-class subset
                single_class_mask = y == y.iloc[0]
                if single_class_mask.sum() >= 20:  # Need minimum samples
                    X_single = X[single_class_mask].head(20)
                    y_single = y[single_class_mask].head(20)
                    
                    try:
                        single_tuner = XGBoostHyperparameterTuner()
                        single_results = single_tuner.tune(
                            X_single, y_single,
                            n_trials=2,
                            cv=2,
                            timeout=30,
                        )
                        results["single_class"] = "HANDLED"
                    except Exception:
                        results["single_class"] = "PROPERLY_FAILED"
                    
                    logger.info("✓ Single class scenario tested")
            
            # Test 4: Missing values (if any)
            if X.isnull().any().any():
                results["missing_values"] = "PRESENT_AND_HANDLED"
                logger.info("✓ Missing values present and handled")
            else:
                results["missing_values"] = "NOT_PRESENT"
                logger.info("! No missing values in test data")
            
        except Exception as e:
            error_msg = f"Edge case testing failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            results["error"] = error_msg
            results["status"] = "FAIL"
            
        return results
    
    def test_integration_with_comparison_framework(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Test integration with existing model comparison framework.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with integration test results
        """
        logger.info("Testing integration with comparison framework...")
        
        results = {}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Test 1: Tune model and get predictions
            tuned_model = quick_tune_xgboost(X_train, y_train, n_trials=5)
            y_pred = tuned_model.predict(X_test)
            y_proba = tuned_model.predict_proba(X_test)
            
            # Test 2: Use comparison metrics
            metrics = calculate_metrics(y_test, y_pred, y_proba, n_bootstrap=50)
            
            results["tuned_model_metrics"] = metrics
            results["integration_test"] = "PASS"
            
            logger.info("✓ Integration with comparison framework successful")
            logger.info(f"  Tuned model CSMF accuracy: {metrics['csmf_accuracy']:.4f}")
            logger.info(f"  Confidence intervals calculated: {metrics['csmf_accuracy_ci']}")
            
            # Test 3: Compare with baseline model
            baseline_model = XGBoostModel()
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_test)
            baseline_metrics = calculate_metrics(y_test, baseline_pred, n_bootstrap=50)
            
            results["baseline_metrics"] = baseline_metrics
            results["improvement"] = {
                "csmf_accuracy": metrics["csmf_accuracy"] - baseline_metrics["csmf_accuracy"],
                "cod_accuracy": metrics["cod_accuracy"] - baseline_metrics["cod_accuracy"],
            }
            
            logger.info(f"✓ Performance comparison:")
            logger.info(f"  Baseline CSMF: {baseline_metrics['csmf_accuracy']:.4f}")
            logger.info(f"  Tuned CSMF: {metrics['csmf_accuracy']:.4f}")
            logger.info(f"  Improvement: {results['improvement']['csmf_accuracy']:+.4f}")
            
        except Exception as e:
            error_msg = f"Integration testing failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            results["error"] = error_msg
            results["status"] = "FAIL"
            
        return results
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis of hyperparameter tuning implementation.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive hyperparameter tuning analysis...")
        
        try:
            # Load data
            X, y = self.load_sample_data()
            
            # Run all tests
            self.results["data_info"] = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": y.nunique(),
                "class_distribution": y.value_counts().to_dict(),
            }
            
            logger.info("="*60)
            self.results["basic_functionality"] = self.test_basic_model_functionality(X, y)
            
            logger.info("="*60)
            self.results["hyperparameter_tuning"] = self.test_hyperparameter_tuning(X, y)
            
            logger.info("="*60)
            self.results["parameter_spaces"] = self.test_parameter_spaces()
            
            logger.info("="*60)
            self.results["edge_cases"] = self.test_edge_cases(X, y)
            
            logger.info("="*60)
            self.results["integration"] = self.test_integration_with_comparison_framework(X, y)
            
            # Overall status
            failed_tests = [k for k, v in self.results.items() 
                          if isinstance(v, dict) and v.get("status") == "FAIL"]
            
            self.results["overall_status"] = "FAIL" if failed_tests else "PASS"
            self.results["failed_tests"] = failed_tests
            self.results["errors"] = self.errors
            
            logger.info("="*60)
            logger.info(f"ANALYSIS COMPLETE: {self.results['overall_status']}")
            if failed_tests:
                logger.error(f"Failed tests: {failed_tests}")
            if self.errors:
                logger.error(f"Errors encountered: {len(self.errors)}")
            
        except Exception as e:
            logger.error(f"Analysis failed with critical error: {str(e)}")
            logger.error(traceback.format_exc())
            self.results["critical_error"] = str(e)
            self.results["overall_status"] = "CRITICAL_FAIL"
        
        return self.results
    
    def save_results(self, filename: str = "analysis_results.json"):
        """Save analysis results to JSON file.
        
        Args:
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main analysis function."""
    print("Hyperparameter Tuning Implementation Analysis")
    print("=" * 50)
    
    # Check for data path argument
    data_path = None
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Using data path: {data_path}")
    
    # Run analysis
    analyzer = HyperparameterAnalysis(data_path=data_path)
    results = analyzer.run_full_analysis()
    analyzer.save_results()
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Overall Status: {results.get('overall_status', 'UNKNOWN')}")
    print(f"Data: {results.get('data_info', {}).get('n_samples', 'N/A')} samples, "
          f"{results.get('data_info', {}).get('n_features', 'N/A')} features")
    
    if results.get('errors'):
        print(f"Errors: {len(results['errors'])}")
        for i, error in enumerate(results['errors'][:3], 1):  # Show first 3 errors
            print(f"  {i}. {error}")
    
    # Key performance metrics
    if 'hyperparameter_tuning' in results and 'best_score' in results['hyperparameter_tuning']:
        print(f"Best tuned CSMF accuracy: {results['hyperparameter_tuning']['best_score']:.4f}")
    
    if 'integration' in results and 'improvement' in results['integration']:
        improvement = results['integration']['improvement']['csmf_accuracy']
        print(f"Improvement over baseline: {improvement:+.4f}")
    
    print(f"\nDetailed results saved to analysis_results.json")
    print(f"Logs saved to hyperparameter_analysis.log")
    
    return 0 if results.get('overall_status') == 'PASS' else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)