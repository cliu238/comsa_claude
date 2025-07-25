# Next Steps for Task IM-052: Fix Bootstrap Confidence Intervals

## Task Overview

**Task ID**: IM-052  
**Priority**: High  
**Dependencies**: IM-035 (VA34 comparison), IM-051 (Ray optimization)  
**Objective**: Fix bootstrap confidence intervals in the model comparison framework

## Current Issue

The bootstrap confidence intervals are not being properly calculated or stored despite `n_bootstrap=100` being specified. The root cause has been identified:

1. **Metrics Calculation Issue**: The `calculate_metrics()` function in `comparison_metrics.py` returns confidence intervals as separate keys (`csmf_accuracy_ci_lower`, `csmf_accuracy_ci_upper`, etc.) rather than as lists
2. **Data Structure Mismatch**: The `ExperimentResult` class expects confidence intervals as lists (`List[float]`), but the current implementation in `ray_tasks.py` tries to handle them incorrectly

## Proposed Solution

### Step 1: Update Metrics Calculation
Modify `model_comparison/metrics/comparison_metrics.py` to return confidence intervals in the expected list format:

```python
# Current (incorrect) format:
metrics = {
    "csmf_accuracy_ci_lower": csmf_ci[0],
    "csmf_accuracy_ci_upper": csmf_ci[1],
    # ...
}

# Proposed (correct) format:
metrics = {
    "csmf_accuracy_ci": [csmf_ci[0], csmf_ci[1]],
    "cod_accuracy_ci": [cod_ci[0], cod_ci[1]],
    # ...
}
```

### Step 2: Simplify Ray Tasks
Update `model_comparison/orchestration/ray_tasks.py` to directly use the list format from metrics:

```python
# Remove the complex conditional logic:
csmf_accuracy_ci=metrics.get("csmf_accuracy_ci") if isinstance(metrics.get("csmf_accuracy_ci"), list) else None,

# Replace with direct assignment:
csmf_accuracy_ci=metrics.get("csmf_accuracy_ci"),
cod_accuracy_ci=metrics.get("cod_accuracy_ci"),
```

### Step 3: Validate Bootstrap Iterations
Ensure the bootstrap calculation is actually running with the specified iterations:
- Add logging to track bootstrap progress
- Verify that `n_bootstrap=100` is being passed correctly through the call chain
- Consider increasing to 1000 iterations for production runs

## Implementation Plan

1. **Update `comparison_metrics.py`**:
   - Modify `calculate_metrics()` to return CI as lists
   - Add debug logging for bootstrap calculations
   - Ensure backward compatibility if needed

2. **Update `ray_tasks.py`**:
   - Simplify the CI assignment logic
   - Remove unnecessary type checking
   - Add validation that CIs are properly formatted

3. **Add Tests**:
   - Create unit tests to verify CI calculation
   - Test with various bootstrap iteration counts (100, 1000)
   - Verify the format matches `ExperimentResult` expectations

4. **Validation**:
   - Run a small experiment to verify CIs are calculated
   - Check that results CSV files contain CI columns with proper values
   - Ensure CIs are reasonable (e.g., contain the point estimate)

## Expected Outcomes

- Bootstrap confidence intervals will be properly calculated and stored
- Results files will contain `csmf_accuracy_ci` and `cod_accuracy_ci` columns with `[lower, upper]` values
- The framework will support configurable bootstrap iterations (100-1000)
- Statistical validation of model comparisons will be more robust

## Success Criteria

1. Running `run_distributed_comparison.py` produces results with non-null CI values
2. CI values are stored as lists in the format `[lower_bound, upper_bound]`
3. The CI contains the point estimate (lower ≤ point estimate ≤ upper)
4. Unit tests pass for various bootstrap configurations
5. No performance regression from the additional bootstrap calculations

## Potential Challenges

1. **Performance Impact**: Bootstrap with 1000 iterations may slow down experiments
   - Mitigation: Make bootstrap iterations configurable per experiment
   - Consider parallel bootstrap calculation within each experiment

2. **Memory Usage**: Large bootstrap arrays may increase memory consumption
   - Mitigation: Use efficient numpy operations
   - Clear intermediate arrays after calculation

3. **Edge Cases**: Small sample sizes or homogeneous predictions may cause bootstrap failures
   - Mitigation: Add error handling for degenerate cases
   - Return None for CIs when bootstrap fails

## Next Actions

1. Create a feature branch: `fix/im-052-bootstrap-confidence-intervals`
2. Implement the changes to `comparison_metrics.py`
3. Update `ray_tasks.py` to use the new format
4. Add comprehensive unit tests
5. Run validation experiments
6. Create PR linking to GitHub issue for IM-052