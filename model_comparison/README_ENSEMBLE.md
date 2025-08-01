# Ensemble Model Experiments (Phase 2-4)

This document describes the ensemble model functionality added to the VA model comparison framework.

## Overview

The ensemble implementation includes:
- **DuckVotingEnsemble**: Enhanced voting classifier supporting both hard and soft voting
- **Weight optimization**: Multiple strategies for optimizing estimator weights
- **Diversity constraints**: Ensure base estimators are sufficiently diverse
- **Comprehensive experiments**: Compare voting strategies, weights, and ensemble sizes

## Phase 2: Voting Strategy Comparison

Compare hard vs soft voting with equal weights:

```bash
python model_comparison/scripts/run_ensemble_comparison.py \
    --data-path data/va34_data.csv \
    --sites Pemba Bohol UP Mexico AP Dar \
    --base-results results/full_comparison_20250729_155434/va34_comparison_results.csv \
    --phase voting \
    --output-dir results/ensemble_phase2
```

## Phase 3: Weight Optimization & Size

### Weight Optimization
Compare different weight strategies:

```bash
python model_comparison/scripts/run_ensemble_comparison.py \
    --data-path data/va34_data.csv \
    --sites Pemba Bohol UP Mexico AP Dar \
    --base-results results/full_comparison_20250729_155434/va34_comparison_results.csv \
    --phase weights \
    --weight-strategies none performance cv \
    --output-dir results/ensemble_phase3_weights
```

### Ensemble Size Impact
Test different ensemble sizes:

```bash
python model_comparison/scripts/run_ensemble_comparison.py \
    --data-path data/va34_data.csv \
    --sites Pemba Bohol UP Mexico AP Dar \
    --base-results results/full_comparison_20250729_155434/va34_comparison_results.csv \
    --phase size \
    --ensemble-sizes 3 5 7 \
    --output-dir results/ensemble_phase3_size
```

## Phase 4: Full Ensemble Optimization

Run comprehensive ensemble experiments:

```bash
python model_comparison/scripts/run_ensemble_comparison.py \
    --data-path data/va34_data.csv \
    --sites Pemba Bohol UP Mexico AP Dar \
    --base-results results/full_comparison_20250729_155434/va34_comparison_results.csv \
    --phase all \
    --voting-strategies soft hard \
    --weight-strategies none performance \
    --ensemble-sizes 3 5 7 \
    --min-diversity 0.2 \
    --output-dir results/ensemble_full_comparison
```

## Configuration Options

### Voting Strategies
- **soft**: Average predicted probabilities (recommended)
- **hard**: Majority vote based on predicted classes

### Weight Strategies
- **none**: Equal weights for all estimators
- **manual**: User-specified weights
- **performance**: Weights based on individual model performance
- **cv**: Cross-validation based weight optimization

### Ensemble Sizes
- Recommended: 3, 5, or 7 estimators
- Larger ensembles may have diminishing returns

### Diversity Settings
- **min_diversity**: Minimum required diversity (0.0-1.0)
- **diversity_metric**: "disagreement" or "correlation"

## Output Structure

Results are saved in the specified output directory:
```
results/ensemble_comparison/
├── ensemble_results/
│   └── ensemble_comparison_results.csv
├── ensemble_plots/
│   ├── voting_strategy_comparison.png
│   ├── weight_optimization_comparison.png
│   ├── ensemble_size_impact.png
│   └── diversity_analysis.png
└── checkpoints/
    └── checkpoint_*.json
```

## Key Implementation Details

### DuckVotingEnsemble
- Located in `baseline/models/ensemble_model.py`
- Follows sklearn estimator interface
- Supports class alignment for different base estimators
- Implements diversity calculation and constraints

### Ensemble Configuration
- Located in `baseline/models/ensemble_config.py`
- Validates ensemble parameters
- Supports weight normalization
- Configurable diversity requirements

### Experiment Framework
- Located in `model_comparison/experiments/ensemble_comparison.py`
- Extends parallel experiment infrastructure
- Supports loading pretrained base models from Phase 1
- Implements phased experimental approach

## Testing

Run ensemble tests:
```bash
pytest tests/test_ensemble_models.py -v
pytest model_comparison/tests/test_ensemble_comparison.py -v
```

## Performance Expectations

Based on the Phase 1 results:
- Best individual model (XGBoost): ~0.82-0.88 CSMF accuracy
- Expected ensemble improvement: 2-5% over best individual model
- Soft voting typically outperforms hard voting
- Performance-based weights often improve results
- Optimal ensemble size is typically 5 estimators

## Next Steps

1. Run Phase 2-4 experiments using the scripts above
2. Analyze results to identify optimal ensemble configuration
3. Consider implementing more advanced ensemble methods:
   - Stacking
   - Dynamic weight adjustment
   - Online ensemble learning
4. Test ensemble robustness across different data distributions