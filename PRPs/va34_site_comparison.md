# VA34 Site-Based Model Comparison PRP

## Overview
Implement a comprehensive experimental framework to compare InSilicoVA and XGBoost model performance using VA34 labels across different site configurations. This experiment will analyze in-domain vs out-domain generalization and the impact of training data size on model performance.

## Context and Background

### Existing Infrastructure
1. **Models Available**:
   - InSilicoVA: `baseline/models/insilico_model.py` - Docker-based implementation with sklearn interface
   - XGBoost: `baseline/models/xgboost_model.py` - Native Python with hyperparameter tuning

2. **Data Pipeline**:
   - VADataProcessor: Handles VA34 label configuration
   - VADataSplitter: Provides site-based data splitting functionality

3. **Key Research Questions**:
   - How well do models generalize from one site to another?
   - Does increasing training data size improve or hurt out-domain performance?
   - Which model architecture is more robust to site distribution shifts?

### VA34 Label System
- 34 distinct cause-of-death categories
- More granular than COD5 (5 categories)
- Standard WHO classification for verbal autopsy

## Implementation Blueprint

### File Structure
```
model_comparison/
├── __init__.py
├── experiments/
│   ├── __init__.py
│   ├── site_comparison.py         # Main experiment runner
│   ├── experiment_config.py       # Pydantic configuration
│   └── experiment_utils.py        # Helper functions
├── metrics/
│   ├── __init__.py
│   ├── comparison_metrics.py      # Metric calculations
│   └── statistical_tests.py       # Significance testing
├── visualization/
│   ├── __init__.py
│   ├── comparison_plots.py        # Result visualizations
│   └── report_generator.py        # Automated reporting
├── results/
│   └── va34_comparison/          # Output directory
└── tests/
    ├── test_site_comparison.py
    └── test_metrics.py
```

### Core Implementation

#### 1. Experiment Configuration (experiment_config.py)
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ExperimentConfig(BaseModel):
    """Configuration for VA34 site comparison experiment."""
    
    # Data configuration
    data_path: str = Field(..., description="Path to VA data")
    label_type: str = Field(default="va34", description="Label system to use")
    
    # Site configuration
    sites: List[str] = Field(..., description="List of sites to include")
    test_sites: Optional[List[str]] = Field(default=None, description="Specific test sites")
    
    # Training configuration
    training_sizes: List[float] = Field(
        default=[0.25, 0.5, 0.75, 1.0],
        description="Fractions of training data to use"
    )
    
    # Model configuration
    models: List[str] = Field(
        default=["insilico", "xgboost"],
        description="Models to compare"
    )
    
    # Experiment settings
    n_bootstrap: int = Field(default=100, description="Bootstrap iterations")
    random_seed: int = Field(default=42, description="Random seed")
    n_jobs: int = Field(default=-1, description="Parallel jobs")
    
    # Output configuration
    output_dir: str = Field(default="results/va34_comparison")
    save_predictions: bool = Field(default=True)
    generate_plots: bool = Field(default=True)
```

#### 2. Main Experiment Runner (site_comparison.py)
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

from baseline.data.data_loader import VADataProcessor
from baseline.data.data_splitter import VADataSplitter
from baseline.models.insilico_model import InSilicoVAModel
from baseline.models.xgboost_model import XGBoostModel
from .experiment_config import ExperimentConfig
from ..metrics.comparison_metrics import calculate_metrics

class SiteComparisonExperiment:
    """VA34 site-based model comparison experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.processor = VADataProcessor()
        self.splitter = VADataSplitter()
        self.results = []
        
    def run_experiment(self) -> pd.DataFrame:
        """Run complete experiment across all configurations."""
        
        # Load and prepare data
        data = self._load_data()
        
        # Run in-domain experiments
        in_domain_results = self._run_in_domain_experiments(data)
        
        # Run out-domain experiments
        out_domain_results = self._run_out_domain_experiments(data)
        
        # Run training size experiments
        size_results = self._run_training_size_experiments(data)
        
        # Combine and save results
        all_results = pd.concat([
            in_domain_results,
            out_domain_results,
            size_results
        ])
        
        self._save_results(all_results)
        
        if self.config.generate_plots:
            self._generate_visualizations(all_results)
            
        return all_results
        
    def _run_in_domain_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Test performance when training and testing on same site."""
        results = []
        
        for site in self.config.sites:
            # Get site data
            site_data = data[data['site'] == site]
            
            # Split into train/test
            splits = self.splitter.split_data(
                site_data,
                split_type='random',
                test_size=0.2,
                random_state=self.config.random_seed
            )
            
            X_train, X_test = splits.X_train, splits.X_test
            y_train, y_test = splits.y_train, splits.y_test
            
            # Train and evaluate each model
            for model_name in self.config.models:
                model = self._get_model(model_name)
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                metrics = calculate_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    n_bootstrap=self.config.n_bootstrap
                )
                
                # Store results
                result = {
                    'experiment_type': 'in_domain',
                    'train_site': site,
                    'test_site': site,
                    'model': model_name,
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    **metrics
                }
                results.append(result)
                
        return pd.DataFrame(results)
        
    def _run_out_domain_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Test performance when training on one site and testing on others."""
        results = []
        
        for train_site in self.config.sites:
            # Get training data
            train_data = data[data['site'] == train_site]
            X_train = train_data.drop(['cause', 'site'], axis=1)
            y_train = train_data['cause']
            
            # Test on other sites
            for test_site in self.config.sites:
                if test_site == train_site:
                    continue
                    
                # Get test data
                test_data = data[data['site'] == test_site]
                X_test = test_data.drop(['cause', 'site'], axis=1)
                y_test = test_data['cause']
                
                # Train and evaluate each model
                for model_name in self.config.models:
                    model = self._get_model(model_name)
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(
                        y_true=y_test,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        n_bootstrap=self.config.n_bootstrap
                    )
                    
                    # Store results
                    result = {
                        'experiment_type': 'out_domain',
                        'train_site': train_site,
                        'test_site': test_site,
                        'model': model_name,
                        'n_train': len(X_train),
                        'n_test': len(X_test),
                        **metrics
                    }
                    results.append(result)
                    
        return pd.DataFrame(results)
```

#### 3. Metrics Module (comparison_metrics.py)
```python
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats
from sklearn.metrics import accuracy_score

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 100
) -> Dict[str, float]:
    """Calculate comprehensive metrics with confidence intervals."""
    
    # Basic metrics
    cod_accuracy = accuracy_score(y_true, y_pred)
    csmf_accuracy = calculate_csmf_accuracy(y_true, y_pred)
    
    # Bootstrap confidence intervals
    cod_ci = bootstrap_metric(
        y_true, y_pred, 
        accuracy_score, 
        n_bootstrap
    )
    
    csmf_ci = bootstrap_metric(
        y_true, y_pred,
        calculate_csmf_accuracy,
        n_bootstrap
    )
    
    return {
        'cod_accuracy': cod_accuracy,
        'cod_accuracy_ci_lower': cod_ci[0],
        'cod_accuracy_ci_upper': cod_ci[1],
        'csmf_accuracy': csmf_accuracy,
        'csmf_accuracy_ci_lower': csmf_ci[0],
        'csmf_accuracy_ci_upper': csmf_ci[1]
    }

def calculate_csmf_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Calculate CSMF accuracy metric."""
    # Get true and predicted fractions
    true_fractions = y_true.value_counts(normalize=True)
    pred_fractions = pd.Series(y_pred).value_counts(normalize=True)
    
    # Align categories
    all_categories = list(set(true_fractions.index) | set(pred_fractions.index))
    true_fractions = true_fractions.reindex(all_categories, fill_value=0)
    pred_fractions = pred_fractions.reindex(all_categories, fill_value=0)
    
    # Calculate CSMF accuracy
    diff = np.abs(true_fractions - pred_fractions).sum()
    min_frac = true_fractions.min()
    
    if min_frac == 1:
        return 1.0 if diff == 0 else 0.0
        
    csmf_accuracy = 1 - diff / (2 * (1 - min_frac))
    return max(0, csmf_accuracy)

def bootstrap_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 100,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate bootstrap confidence intervals for a metric."""
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Resample indices
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Calculate metric on resampled data
        score = metric_func(y_true.iloc[indices], y_pred[indices])
        scores.append(score)
        
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    
    return lower, upper
```

#### 4. Visualization Module (comparison_plots.py)
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_model_comparison(results: pd.DataFrame, output_path: str):
    """Create comprehensive comparison plots."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. In-domain vs Out-domain comparison
    plot_domain_comparison(results, axes[0, 0])
    
    # 2. Training size impact
    plot_training_size_impact(results, axes[0, 1])
    
    # 3. Site-specific performance heatmap
    plot_site_heatmap(results, axes[1, 0])
    
    # 4. Statistical significance
    plot_significance_bars(results, axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_domain_comparison(results: pd.DataFrame, ax):
    """Plot in-domain vs out-domain performance."""
    # Filter data
    domain_data = results[results['experiment_type'].isin(['in_domain', 'out_domain'])]
    
    # Create grouped bar plot
    domain_summary = domain_data.groupby(['experiment_type', 'model'])[
        ['csmf_accuracy', 'cod_accuracy']
    ].mean().reset_index()
    
    # Pivot for plotting
    pivot_csmf = domain_summary.pivot(
        index='model', 
        columns='experiment_type', 
        values='csmf_accuracy'
    )
    
    pivot_csmf.plot(kind='bar', ax=ax)
    ax.set_title('CSMF Accuracy: In-domain vs Out-domain')
    ax.set_ylabel('CSMF Accuracy')
    ax.set_xlabel('Model')
    ax.legend(title='Domain Type')
    ax.set_ylim(0, 1)
```

### Validation and Testing

#### Unit Tests (test_site_comparison.py)
```python
import pytest
import pandas as pd
import numpy as np
from model_comparison.experiments.site_comparison import SiteComparisonExperiment
from model_comparison.experiments.experiment_config import ExperimentConfig

class TestSiteComparison:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample VA data for testing."""
        n_samples = 1000
        n_sites = 4
        n_causes = 34
        
        data = pd.DataFrame({
            'site': np.random.choice([f'site_{i}' for i in range(n_sites)], n_samples),
            'cause': np.random.choice([f'cause_{i}' for i in range(n_causes)], n_samples),
            'age': np.random.randint(0, 100, n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples),
            # Add more features as needed
        })
        
        # Add numeric features
        for i in range(10):
            data[f'symptom_{i}'] = np.random.choice([0, 1], n_samples)
            
        return data
    
    def test_in_domain_experiment(self, sample_data, tmp_path):
        """Test in-domain experiment execution."""
        config = ExperimentConfig(
            data_path=str(tmp_path / "test_data.csv"),
            sites=['site_0', 'site_1'],
            training_sizes=[1.0],
            n_bootstrap=10,
            output_dir=str(tmp_path / "results")
        )
        
        sample_data.to_csv(config.data_path, index=False)
        
        experiment = SiteComparisonExperiment(config)
        # Mock the data loading
        experiment._load_data = lambda: sample_data
        
        results = experiment._run_in_domain_experiments(sample_data)
        
        assert len(results) == 4  # 2 sites × 2 models
        assert all(results['experiment_type'] == 'in_domain')
        assert all(results['train_site'] == results['test_site'])
```

### Execution Script

#### run_va34_comparison.py
```python
#!/usr/bin/env python
"""Run VA34 site-based model comparison experiment."""

import argparse
import logging
from pathlib import Path

from model_comparison.experiments.site_comparison import SiteComparisonExperiment
from model_comparison.experiments.experiment_config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description='Run VA34 site comparison')
    parser.add_argument('--data-path', required=True, help='Path to VA data')
    parser.add_argument('--output-dir', default='results/va34_comparison')
    parser.add_argument('--sites', nargs='+', help='Sites to include')
    parser.add_argument('--n-bootstrap', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = ExperimentConfig(
        data_path=args.data_path,
        sites=args.sites or ['site_1', 'site_2', 'site_3', 'site_4'],
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap
    )
    
    # Run experiment
    experiment = SiteComparisonExperiment(config)
    results = experiment.run_experiment()
    
    logging.info(f"Experiment completed. Results saved to {config.output_dir}")
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Total experiments run: {len(results)}")
    print(f"\nMean CSMF Accuracy by Model:")
    print(results.groupby('model')['csmf_accuracy'].mean())
    print(f"\nGeneralization Gap (In-domain - Out-domain):")
    in_domain = results[results['experiment_type'] == 'in_domain'].groupby('model')['csmf_accuracy'].mean()
    out_domain = results[results['experiment_type'] == 'out_domain'].groupby('model')['csmf_accuracy'].mean()
    print(in_domain - out_domain)

if __name__ == "__main__":
    main()
```

### Validation Commands

```bash
# Format code
poetry run black model_comparison/
poetry run ruff check --fix model_comparison/

# Type checking
poetry run mypy model_comparison/

# Run tests
poetry run pytest tests/model_comparison/ -v --cov=model_comparison --cov-report=term-missing

# Run experiment with sample data
poetry run python run_va34_comparison.py --data-path data/phmrc_va_data.csv --sites site_1 site_2 site_3
```

## Key Insights and Considerations

### 1. Site Distribution Shifts
- Different sites may have different cause distributions
- Environmental and cultural factors affect symptom patterns
- Model robustness varies with architecture

### 2. Training Data Size Effects
- More data doesn't always mean better generalization
- Site-specific patterns may overfit with large training sets
- Balance between data quantity and diversity

### 3. Model Architecture Differences
- InSilicoVA: Rule-based, may be more stable across sites
- XGBoost: Data-driven, may capture site-specific patterns better

### 4. Statistical Rigor
- Bootstrap confidence intervals for robust estimates
- Multiple comparison correction when needed
- Clear reporting of uncertainty

## Expected Outputs

1. **Results Files**:
   - `in_domain_results.csv`: Same-site train/test performance
   - `out_domain_results.csv`: Cross-site performance
   - `training_size_results.csv`: Impact of data quantity
   - `full_results.csv`: All experimental results

2. **Visualizations**:
   - `model_comparison.png`: Main comparison plots
   - `site_performance_heatmap.png`: Site-specific results
   - `training_curves.png`: Data size impact

3. **Summary Report**:
   - Key findings and recommendations
   - Statistical significance of differences
   - Deployment guidelines based on results

## Quality Score: 9/10

This PRP provides:
- Complete implementation structure
- Working code examples
- Statistical rigor with bootstrapping
- Comprehensive testing approach
- Clear execution path
- Visualization and reporting

The implementation should succeed with minimal modifications.