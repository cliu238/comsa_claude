# XGBoost Generalization Investigation Report

**Date**: Thu Jul 31 15:32:10 EDT 2025
**Objective**: Investigate why XGBoost shows superior in-domain performance but poor out-of-domain generalization compared to InSilicoVA.

## Key Findings

### Baseline Performance (Original Experiment)
- **XGBoost**: In-domain CSMF=0.8663, Out-domain CSMF=0.3999 (53.8% gap)
- **InSilicoVA**: In-domain CSMF=0.7997, Out-domain CSMF=0.4605 (42.4% gap)

### Root Causes Identified
1. **Overfitting to site-specific patterns**: XGBoost memorizes local symptom reporting quirks
2. **Insufficient regularization**: Default hyperparameters allow too complex models
3. **In-domain optimization bias**: Tuning on standard CV doesn't optimize for transfer

## Experiments Conducted

### 1. Regularization Comparison
Tested three XGBoost configurations:
- Standard enhanced configuration
- Conservative configuration (shallow trees, strong regularization)
- Fixed conservative parameters without tuning

### 2. Cross-Domain Tuning
Compared different tuning objectives:
- In-domain only (standard)
- Cross-domain validation
- Transfer-focused optimization

### 3. Model Complexity Analysis
Analyzed:
- Tree depth distributions
- Feature usage patterns
- Overfitting indicators

### 4. Optimized Subsampling
Tested optimized subsampling parameters:
- Baseline enhanced configuration
- Fixed optimized subsampling configuration
- Tuned with optimized search space

## Results Summary
See generated plots and CSV files for detailed results.

## Recommendations
Based on the investigation, we recommend:
1. Adopting the configuration that achieved the lowest performance gap
2. Implementing cross-domain validation during model selection
3. Monitoring overfitting metrics during deployment

## Files in This Report
- `investigation_log.txt`: Complete execution log
- `experiment_summary.csv`: Summary of all experiments
- `performance_gap_comparison.png`: Visual comparison of strategies
- `in_vs_out_domain_scatter.png`: Generalization performance scatter plot
- Additional analysis outputs in subdirectories
