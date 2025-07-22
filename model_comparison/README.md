# Model Comparison Framework

This module implements a comprehensive framework for comparing Verbal Autopsy (VA) models across different data collection sites.

## Structure

```
model_comparison/
├── experiments/           # Core experiment logic
│   ├── experiment_config.py    # Pydantic configuration models
│   └── site_comparison.py      # Main experiment runner
├── metrics/              # Metric calculations
│   └── comparison_metrics.py   # CSMF accuracy, COD accuracy, bootstrap CIs
├── visualization/        # Plotting and visualization
│   └── comparison_plots.py     # Generate comparison plots
├── scripts/              # Executable scripts
│   └── run_va34_comparison.py  # Main entry point
├── tests/                # Unit tests
│   ├── test_metrics.py         # Test metric calculations
│   └── test_site_comparison.py # Test experiment logic
└── results/              # Output directory (git-ignored)
```

## Running Experiments

### Basic Usage

```bash
cd model_comparison/scripts
python run_va34_comparison.py --data-path path/to/va_data.csv
```

### Full Example

```bash
python run_va34_comparison.py \
  --data-path ../../results/baseline/processed_data/adult_openva_20250717_163656.csv \
  --sites AP Bohol Dar Mexico UP \
  --models insilico xgboost \
  --training-sizes 0.25 0.5 0.75 1.0 \
  --n-bootstrap 50 \
  --output-dir ../results/va34_comparison
```

### Command Line Options

- `--data-path`: Path to VA data CSV (required)
- `--sites`: Specific sites to include (default: all sites)
- `--models`: Models to compare [`insilico`, `xgboost`] (default: both)
- `--training-sizes`: Training data fractions (default: [0.25, 0.5, 0.75, 1.0])
- `--n-bootstrap`: Bootstrap iterations for CIs (default: 100)
- `--output-dir`: Where to save results (default: model_comparison/results/va34_comparison)
- `--no-plots`: Skip visualization generation
- `--debug`: Enable debug logging

## Data Requirements

Your VA data must have:
- `site` column: Site/location identifiers
- `va34` column: VA34 cause of death labels (will be renamed to 'cause')
- Feature columns: Symptom and demographic features

## Output Files

Results are saved to the output directory:
- `full_results.csv`: All experimental results
- `in_domain_results.csv`: Same-site train/test performance
- `out_domain_results.csv`: Cross-site train/test performance
- `training_size_results.csv`: Impact of training data size
- `summary_statistics.csv`: Aggregated statistics
- `*.png`: Visualization plots (unless --no-plots)

## Metrics

- **COD Accuracy**: Individual cause-of-death prediction accuracy
- **CSMF Accuracy**: Cause-specific mortality fraction accuracy
- **Confidence Intervals**: Bootstrap-based 95% CIs

## Models Supported

1. **InSilicoVA**: Bayesian probabilistic model (requires Docker)
2. **XGBoost**: Gradient boosting classifier

## Notes

- InSilicoVA requires Docker to be installed and running
- Experiments may take 10-30 minutes depending on data size and bootstrap iterations
- Results are automatically saved and can be resumed if interrupted