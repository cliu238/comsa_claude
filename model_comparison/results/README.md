# Results Directory

This directory contains the output from model comparison experiments. 

## Important Note

**This directory is git-ignored** to prevent large result files from being committed to the repository.

## Result File Structure

Each experiment run creates a subdirectory with:

```
experiment_name/
├── experiment.log            # Detailed execution log
├── full_results.csv          # All experimental results with metrics
├── in_domain_results.csv     # Results from same-site train/test
├── out_domain_results.csv    # Results from cross-site train/test
├── training_size_results.csv # Results from training size experiments
├── summary_statistics.csv    # Aggregated statistics by model and experiment type
├── model_comparison.png      # Main 4-panel comparison plot
├── model_performance.png     # Bar charts of model performance
└── generalization_gap.png    # Analysis of generalization gaps
```

## Column Descriptions

### full_results.csv
- `experiment_type`: Type of experiment (in_domain, out_domain, training_size)
- `train_site`: Site used for training
- `test_site`: Site used for testing
- `model`: Model name (insilico, xgboost)
- `n_train`: Number of training samples
- `n_test`: Number of test samples
- `training_fraction`: Fraction of training data used (for training_size experiments)
- `cod_accuracy`: Individual cause-of-death accuracy
- `cod_accuracy_ci_lower/upper`: 95% CI bounds for COD accuracy
- `csmf_accuracy`: Cause-specific mortality fraction accuracy
- `csmf_accuracy_ci_lower/upper`: 95% CI bounds for CSMF accuracy

### summary_statistics.csv
Aggregated statistics grouped by experiment_type and model, showing mean, std, min, and max for each metric.

## Sharing Results

To share results with others:
1. Compress the experiment directory
2. Share the compressed file
3. Document the experiment parameters used