# Next Steps: Fix Bootstrap Confidence Intervals in Model Comparison Framework (IM-052)

## Task Overview

**Task ID**: IM-052  
**Priority**: High  
**Dependencies**: IM-035 (VA34 comparison), IM-051 (Ray optimization)  
**Target Date**: Q1 2025  

## Problem Statement

The model comparison framework is not calculating bootstrap confidence intervals despite `n_bootstrap=100` being specified in the configuration. This is critical for statistical validation of model performance differences.

### Root Cause Analysis

Based on the task notes, the issue appears to be a data format mismatch:
- `ray_tasks.py` expects confidence intervals in list format `[lower, upper]`
- Current metrics calculation returns separate bounds
- `ExperimentResult` class may not be properly handling CI data

## Implementation Steps

### 1. Investigate Current Implementation
- Examine `model_comparison/ray_tasks.py` to understand expected CI format
- Review metrics calculation functions to see current output format
- Check `ExperimentResult` class structure for CI handling
- Identify exact location where bootstrap CI calculation fails

### 2. Fix Metrics Calculation
- Modify metrics functions to return confidence intervals in `[lower, upper]` format
- Ensure bootstrap sampling is properly implemented with specified iterations
- Add proper error handling for edge cases (e.g., insufficient samples)

### 3. Update Data Structures
- Update `ExperimentResult` class to properly store and serialize CI data
- Ensure CI fields are included in result exports (CSV, JSON)
- Add validation to ensure CI data is in correct format

### 4. Validation and Testing
- Implement unit tests for bootstrap CI calculation
- Test with varying bootstrap iterations (100, 500, 1000)
- Verify CI values are reasonable (e.g., lower < point estimate < upper)
- Test edge cases: small sample sizes, extreme class imbalance

### 5. Integration Testing
- Run full VA34 comparison with bootstrap enabled
- Verify CI appears in all output files
- Check visualization scripts can handle CI data
- Ensure distributed Ray execution preserves CI calculations

## Technical Approach

### Bootstrap Implementation
```python
# Expected implementation pattern
def calculate_metric_with_ci(y_true, y_pred, metric_func, n_bootstrap=100):
    """Calculate metric with bootstrap confidence intervals."""
    point_estimate = metric_func(y_true, y_pred)
    
    if n_bootstrap > 0:
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        
        # Calculate 95% CI
        lower = np.percentile(bootstrap_scores, 2.5)
        upper = np.percentile(bootstrap_scores, 97.5)
        
        return {
            'value': point_estimate,
            'ci': [lower, upper],  # List format expected by ray_tasks.py
            'n_bootstrap': n_bootstrap
        }
    
    return {'value': point_estimate, 'ci': None, 'n_bootstrap': 0}
```

### Data Structure Updates
```python
# ExperimentResult should handle CI data
@dataclass
class ExperimentResult:
    # ... existing fields ...
    csmf_accuracy: float
    csmf_accuracy_ci: Optional[List[float]] = None
    cod_accuracy: float
    cod_accuracy_ci: Optional[List[float]] = None
    # ... other metrics with CI ...
```

## Deliverables

1. **Fixed ray_tasks.py** with proper CI handling
2. **Updated metrics calculation** returning correct format
3. **Modified ExperimentResult** class with CI fields
4. **Unit tests** for bootstrap functionality
5. **Integration test results** showing working CI
6. **Documentation updates** explaining CI usage

## Success Criteria

- Bootstrap CI calculated for all metrics when `n_bootstrap > 0`
- CI values appear in all output files (CSV, JSON)
- CI format is consistent: `[lower_bound, upper_bound]`
- Tests pass with 100% coverage of new CI code
- Full VA34 comparison runs successfully with CI enabled
- Performance impact is acceptable (<20% slowdown for n_bootstrap=100)

## Risk Mitigation

- **Performance**: Use numpy vectorization for bootstrap sampling
- **Memory**: Process bootstrap samples in batches if needed
- **Compatibility**: Ensure backward compatibility with existing results
- **Validation**: Add checks to ensure CI bounds are sensible

## Notes for Implementation

- Start by creating a minimal test case that reproduces the issue
- Use existing XGBoost/InSilicoVA comparison results to validate fix
- Consider adding progress bar for bootstrap iterations in long runs
- Document any changes to the experiment configuration format
- Consider making bootstrap iterations configurable per metric