# Ensemble Analysis Scripts

This folder contains the corrected ensemble vs baseline analysis scripts for VA model comparison.

## Scripts

### analyze_ensemble_vs_baseline.py
The main streamlined analysis script that:
- Loads ensemble and baseline results
- Performs proper train/test site matching
- Compares ensemble models with individual baseline models
- Generates performance rankings and head-to-head comparisons
- Creates summary reports

**Usage:**
```bash
python scripts/analyze_ensemble_vs_baseline.py
```

### final_corrected_ensemble_analysis.py
Comprehensive analysis script with detailed stratifications including:
- 3-model vs 5-model ensemble comparisons
- Individual ensemble combination analysis
- Site-specific performance patterns
- Computational cost analysis

### streamlined_ensemble_analysis.py
Identical to `analyze_ensemble_vs_baseline.py` - kept for reference.

## Key Findings

1. **XGBoost is the best model** (0.7484 CSMF accuracy)
2. **Ensembles underperform** by ~10% compared to XGBoost
3. **No computational justification** for using ensembles (3-5x cost for worse performance)

## Data Sources

- Ensemble results: `results/ensemble_with_names-v2/va34_comparison_results.csv`
- Baseline results: `results/full_comparison_20250729_155434/va34_comparison_results.csv`

## Output

Results are saved to: `results/final_corrected_ensemble_analysis/`

## Methodology Notes

- Uses only training_fraction=1.0 for fair comparison
- Matches exact train/test site combinations
- Compares on 9 common site combinations (AP, Mexico, UP)
- Primary metric: CSMF accuracy