# Ensemble Model Usage Guide

This guide explains how to use ensemble models with the VA model comparison framework.

## Overview

The ensemble implementation supports:
- Mixed data formats (numeric for ML models, OpenVA format for InSilicoVA)
- Multiple voting strategies (soft, hard)
- Weight optimization strategies (none, performance)
- Flexible ensemble composition
- Both in-domain and out-domain evaluation

## Key Features

1. **Proper Data Preprocessing**: Uses VADataProcessor to prevent label leakage by excluding all cause-of-death related columns

2. **Mixed Model Support**: Each model receives data in its required format (numeric for ML models, OpenVA encoding for InSilicoVA)

3. **Comprehensive Evaluation**: Tests both in-domain and out-domain performance to identify ensembles that improve generalization

## Usage Examples

### Basic Ensemble Experiment

```bash
# Run ensemble experiments with proper preprocessing
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Dar Mexico AP UP \
    --models ensemble \
    --ensemble-voting-strategies soft hard \
    --ensemble-weight-strategies none performance \
    --ensemble-sizes 3 5 \
    --ensemble-base-models all \
    --training-sizes 0.7 \
    --n-bootstrap 30 \
    --output-dir results/ensemble_exploration
```

### Combined Individual + Ensemble Experiments

To explore ensemble models alongside individual models for both in-domain and out-domain performance:

#### Option 1: Run Everything in One Command
```bash
# Run individual models AND ensemble exploration together
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Dar Mexico AP UP \
    --models xgboost random_forest categorical_nb logistic_regression insilico ensemble \
    --enable-ensemble-exploration \
    --ensemble-voting-strategies soft hard \
    --ensemble-weight-strategies none performance \
    --ensemble-sizes 3 5 \
    --ensemble-base-models all \
    --ensemble-combination-strategy smart \
    --training-sizes 0.7 \
    --n-bootstrap 30 \
    --output-dir results/combined_analysis
```

#### Option 2: Separate Runs for Better Control
```bash
# First run individual models to establish baselines
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Dar Mexico AP UP \
    --models xgboost random_forest categorical_nb logistic_regression insilico \
    --training-sizes 0.7 \
    --n-bootstrap 30 \
    --output-dir results/phase1_individual

# Then run ensemble exploration (only needs --models ensemble)
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Dar Mexico AP UP \
    --models ensemble \
    --ensemble-voting-strategies soft hard \
    --ensemble-weight-strategies none performance \
    --ensemble-sizes 3 5 \
    --ensemble-base-models all \
    --training-sizes 0.7 \
    --n-bootstrap 30 \
    --output-dir results/phase2_ensemble
```

#### Option 3: Focused Ensemble Exploration
```bash
# Run specific ensemble combinations with individual baselines
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Dar Mexico AP UP \
    --models xgboost random_forest insilico ensemble \
    --ensemble-voting-strategies soft \
    --ensemble-weight-strategies performance \
    --ensemble-sizes 3 \
    --ensemble-base-models xgboost random_forest insilico \
    --training-sizes 0.7 \
    --n-bootstrap 30 \
    --output-dir results/focused_analysis
```

## Ensemble Configuration Options

### Voting Strategies
- `soft`: Average predicted probabilities (recommended)
- `hard`: Majority vote on predicted classes

### Weight Strategies
- `none`: Equal weights for all models
- `performance`: Weight by individual model performance

### Combination Strategies
- `smart`: Intelligent selection ensuring diversity (default)
- `exhaustive`: Test all possible combinations (limited by max-combinations)

### Base Models
- `all`: Include all available models
- Specific list: `--ensemble-base-models xgboost random_forest insilico`

## Expected Outcomes

### Improved Generalization
Ensembles should show:
- Smaller gap between in-domain and out-domain performance
- Better out-domain accuracy than individual models
- Maintained or improved in-domain performance

### How Performance is Evaluated

The framework automatically evaluates both in-domain and out-domain performance:

1. **In-Domain Performance**: Model trained and tested on data from the same sites
2. **Out-Domain Performance**: Model trained on some sites, tested on held-out sites

Example site splits:
- Train sites: AP, BD
- In-domain test: Held-out data from AP, BD
- Out-domain test: Data from OD, UP, RJ

### Example Results Analysis
```
Individual Model Performance:
- XGBoost: 91% in-domain, 60% out-domain (31% gap)
- Random Forest: 89% in-domain, 58% out-domain (31% gap)
- InSilicoVA: 80% in-domain, 45% out-domain (35% gap)
- CategoricalNB: 85% in-domain, 70% out-domain (15% gap)

Ensemble Performance:
- Soft voting (XGB+RF+CNB): 92% in-domain, 75% out-domain (17% gap)
- Performance-weighted: 93% in-domain, 77% out-domain (16% gap)
- All models ensemble: 90% in-domain, 72% out-domain (18% gap)

Key Insights:
- CategoricalNB has best generalization (smallest gap)
- Ensembles improve out-domain performance significantly
- Performance weighting slightly improves results
```

### Reading the Output Reports

The results directory will contain:
1. `summary_report.json` - Overall performance metrics
2. `detailed_results.csv` - Detailed experiment results
3. `performance_plots/` - Visualizations showing:
   - In-domain vs out-domain performance comparison
   - Model performance across different sites
   - Ensemble composition analysis

## Implementation Details

### Data Flow
1. VADataProcessor loads and preprocesses data
2. Label-equivalent columns are excluded to prevent leakage
3. Both numeric and OpenVA formats are prepared
4. Each model in ensemble receives appropriate format
5. Predictions are combined using voting strategy

### Label Exclusion
The VADataProcessor automatically excludes label-equivalent columns to prevent data leakage, including:
- Cause classification columns (va34, va46, va55, gs_text*, gs_code*)
- COD groupings (cod5)
- Comorbidity and metadata columns

## Data Requirements and Known Issues

### Stratification Errors
**Issue**: "The least populated class in y has only 1 member, which is too few"

**Cause**: The PHMRC dataset has 34 cause-of-death classes with very imbalanced distributions. Some sites have classes with only 1-2 samples, making stratified splitting impossible.

**Data distribution per site**:
- **Dar**: 1,726 samples (largest, most balanced)
- **Mexico**: 1,586 samples 
- **AP**: 1,554 samples
- **UP**: 1,419 samples
- **Bohol**: 1,259 samples
- **Pemba**: 297 samples (smallest, avoid for ensemble experiments)

**Solutions**:
1. **Use training_sizes ≤ 0.7** to ensure enough samples for validation
2. **Use larger sites** (Dar, Mexico, AP, UP) for experiments
3. **Avoid Pemba** in ensemble experiments due to insufficient data
4. **Use Bohol cautiously** as it has limited samples for some classes

### Recommended Site Combinations

**For comprehensive analysis**:
```bash
--sites Dar Mexico AP UP
```

**For quick testing**:
```bash
--sites Dar Mexico
```

**All sites (only with small training_sizes)**:
```bash
--sites Dar Mexico AP UP Bohol Pemba --training-sizes 0.5
```

## Troubleshooting

### Memory Issues
If running out of memory with many ensemble combinations:
- Use smaller `--batch-size` (default is 5)
- Limit `--n-workers` (default uses all CPUs)
- Start with fewer sites for initial testing

### InSilicoVA Import Errors
If you see "No module named 'baseline.metrics'" errors:
- This has been fixed in the latest version
- The import was corrected from `baseline.metrics.va_metrics` to `model_comparison.metrics.comparison_metrics`

### Performance Issues
For faster experimentation:
- Use `--ensemble-combination-strategy smart` instead of exhaustive
- Reduce `--n-bootstrap` for quicker metric calculation
- Start with smaller `--ensemble-sizes`
- Use smaller `--training-sizes` (e.g., 0.5) for initial exploration