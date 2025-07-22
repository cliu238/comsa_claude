# NEXT_STEPS for Task IM-035: VA34 Site-Based Model Comparison

## Task Overview
Implement a comprehensive experiment to compare InSilicoVA vs XGBoost performance using VA34 labels across different site configurations (in-domain vs out-domain) and analyze the impact of training data size on model generalization.

## Context
- We have both InSilicoVA and XGBoost models implemented with sklearn-like interfaces
- VADataSplitter supports site-based data splitting
- Need to test hypothesis: does more training data hurt out-domain performance?
- Focus on VA34 labels (34 cause categories) for this experiment

## Implementation Steps

### 1. Create Experiment Module Structure
```
model_comparison/
├── __init__.py
├── experiments/
│   ├── __init__.py
│   ├── site_comparison.py      # Main experiment runner
│   └── experiment_config.py    # Configuration for experiments
├── metrics/
│   ├── __init__.py
│   └── comparison_metrics.py   # CSMF, COD accuracy calculations
├── visualization/
│   ├── __init__.py
│   └── comparison_plots.py     # Result visualization
└── results/
    └── va34_comparison/        # Experiment outputs
```

### 2. Experiment Design

#### 2.1 Site Configurations
- **In-domain**: Train and test on same site (e.g., site A train → site A test)
- **Out-domain**: Train on one site, test on others (e.g., site A train → site B, C, D test)
- **Mixed-domain**: Train on multiple sites, test on held-out site

#### 2.2 Training Data Size Variations
- 25% of available training data
- 50% of available training data
- 75% of available training data
- 100% of available training data

#### 2.3 Evaluation Metrics
- CSMF Accuracy (primary metric)
- COD Accuracy (individual prediction accuracy)
- Per-site performance breakdown
- Confidence intervals via bootstrapping

### 3. Implementation Components

#### 3.1 Experiment Configuration (experiment_config.py)
- Define experiment parameters using Pydantic
- Specify sites to use
- Set training data percentages
- Configure random seeds for reproducibility

#### 3.2 Site Comparison Runner (site_comparison.py)
- Load data using VADataProcessor with VA34 labels
- Split data by site using VADataSplitter
- Run experiments for each configuration:
  - In-domain scenarios
  - Out-domain scenarios
  - Different training sizes
- Store results in structured format

#### 3.3 Comparison Metrics (comparison_metrics.py)
- Implement consistent metric calculation
- Add bootstrapping for confidence intervals
- Calculate per-cause performance breakdown

#### 3.4 Visualization (comparison_plots.py)
- Performance comparison plots (bar charts, box plots)
- Training size impact curves
- Site-specific performance heatmaps
- Statistical significance indicators

### 4. Expected Outputs

#### 4.1 Results CSV Files
- `in_domain_results.csv`: Performance for same-site train/test
- `out_domain_results.csv`: Performance for cross-site scenarios
- `training_size_impact.csv`: Effect of data size on performance

#### 4.2 Visualizations
- Performance comparison charts
- Generalization gap analysis (in-domain vs out-domain)
- Training data size impact curves

#### 4.3 Summary Report
- Key findings about model generalization
- Statistical significance of differences
- Recommendations for deployment scenarios

## Technical Considerations

1. **Data Loading**: Use VADataProcessor with `label_type='va34'` configuration
2. **Site Handling**: Leverage VADataSplitter's site-based splitting functionality
3. **Model Interfaces**: Both models follow sklearn interface (fit, predict, predict_proba)
4. **Reproducibility**: Set random seeds for data splitting and model training
5. **Memory Management**: Process sites sequentially to avoid memory issues
6. **Progress Tracking**: Use tqdm for long-running experiments

## Dependencies
- baseline.models.insilico_model
- baseline.models.xgboost_model
- baseline.data.data_loader (VADataProcessor)
- baseline.data.data_splitter (VADataSplitter)
- pandas, numpy, matplotlib, seaborn
- scipy for statistical tests

## Success Criteria
1. Complete experiment runs without errors
2. Generate reproducible results
3. Clear visualization of performance differences
4. Statistical validation of findings
5. Actionable insights about model deployment strategies

## Next Steps After Implementation
- Review results with domain experts
- Prepare for COD5 comparison (IM-041)
- Use findings to inform transfer learning approach (IM-015)