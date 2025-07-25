name: "Fix Bootstrap Confidence Intervals in Model Comparison Framework"
description: |

## Purpose
Fix the bootstrap confidence interval calculation in the VA model comparison framework to ensure proper statistical validation of model performance differences. Currently, bootstrap CI is not being calculated despite configuration settings.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Fix the bootstrap confidence interval calculation to properly compute and return CI values in the expected format when `n_bootstrap > 0` is specified in the model comparison configuration. This will enable statistical validation of model performance differences.

## Why
- **Business value**: Provides statistical rigor to model comparisons, allowing data scientists to determine if performance differences are statistically significant
- **Integration**: Essential for the VA34 comparison framework (IM-035) and Ray optimization (IM-051) 
- **Problems solved**: Without CI, we cannot determine if one model truly outperforms another or if differences are due to random variation

## What
The system should calculate bootstrap confidence intervals for all metrics (CSMF accuracy, COD accuracy) when `n_bootstrap > 0` is specified, returning them in the format expected by `ray_tasks.py` and properly storing them in `ExperimentResult`.

### Success Criteria
- [ ] Bootstrap CI calculated for all metrics when `n_bootstrap > 0`
- [ ] CI values appear in all output files (CSV, JSON)
- [ ] CI format is consistent: `[lower_bound, upper_bound]`
- [ ] Tests pass with 100% coverage of new CI code
- [ ] Full VA34 comparison runs successfully with CI enabled
- [ ] Performance impact is acceptable (<20% slowdown for n_bootstrap=100)

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
  why: sklearn's resample function for bootstrap sampling with stratify parameter support
  
- url: https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
  why: Best practices for bootstrap CI implementation in ML, percentile method
  
- file: model_comparison/metrics/comparison_metrics.py
  why: Current implementation that needs fixing - returns wrong CI format
  
- file: model_comparison/orchestration/ray_tasks.py
  why: Consumer of metrics that expects CI in list format [lower, upper]
  
- file: model_comparison/orchestration/config.py
  why: ExperimentResult dataclass that stores CI as Optional[List[float]]

- file: model_comparison/tests/test_metrics.py
  why: Existing test patterns to follow for new bootstrap CI tests
```

### Current Codebase tree
```bash
model_comparison/
├── metrics/
│   └── comparison_metrics.py  # Returns CI as separate fields
├── orchestration/
│   ├── ray_tasks.py          # Expects CI as [lower, upper] list
│   └── config.py             # ExperimentResult with CI fields
└── tests/
    └── test_metrics.py       # Existing metric tests
```

### Desired Codebase tree with files to be added
```bash
model_comparison/
├── metrics/
│   └── comparison_metrics.py  # MODIFIED: Returns CI in correct format
├── orchestration/
│   ├── ray_tasks.py          # NO CHANGE: Already handles list format
│   └── config.py             # NO CHANGE: Already has correct fields
└── tests/
    └── test_metrics.py       # MODIFIED: Add bootstrap CI tests
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: ray_tasks.py expects CI in list format [lower, upper]
# Lines 141-142 check: isinstance(metrics.get("csmf_accuracy_ci"), list)

# CRITICAL: Must maintain backward compatibility for existing results
# Some code may expect the old format with separate _ci_lower/_ci_upper fields

# CRITICAL: Bootstrap with stratification may fail with very small classes
# Need fallback to regular bootstrap if stratified sampling fails

# CRITICAL: Use numpy's RandomState(42) for reproducible bootstrap
# Don't use global random state which could affect other parts of the system

# CRITICAL: Performance consideration - bootstrap is O(n_bootstrap * n_samples)
# For large datasets, this can be slow. Consider progress indication for long runs.
```

## Implementation Blueprint

### Data models and structure

The metrics should return a dictionary with CI in list format:
```python
# Expected return format from calculate_metrics
{
    "cod_accuracy": 0.85,
    "cod_accuracy_ci": [0.82, 0.88],  # List format, not separate fields
    "csmf_accuracy": 0.91,
    "csmf_accuracy_ci": [0.89, 0.93],
    # Optional: Keep old format for backward compatibility
    "cod_accuracy_ci_lower": 0.82,  # Deprecated
    "cod_accuracy_ci_upper": 0.88,  # Deprecated
}
```

### List of tasks to be completed in order

```yaml
Task 1:
MODIFY model_comparison/metrics/comparison_metrics.py:
  - FIND pattern: "metrics = {"
  - MODIFY to return CI in list format while keeping old format for compatibility
  - ENSURE bootstrap_metric returns tuple that gets converted to list
  - ADD progress indication for long bootstrap runs

Task 2:
MODIFY model_comparison/tests/test_metrics.py:
  - ADD test for new CI list format
  - ADD test for bootstrap with small sample sizes
  - ADD test for bootstrap with stratification failure
  - ADD performance test for n_bootstrap scaling

Task 3:
CREATE integration test to verify end-to-end CI calculation:
  - Test that CI appears in ExperimentResult
  - Test that CI is saved to output files
  - Test Ray distributed execution preserves CI
```

### Per task pseudocode

```python
# Task 1 - Fix calculate_metrics in comparison_metrics.py
def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
) -> Dict[str, float]:
    # PATTERN: Validate inputs first (existing pattern)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
    
    # Calculate point estimates
    cod_accuracy = accuracy_score(y_true, y_pred)
    csmf_accuracy = calculate_csmf_accuracy(y_true, y_pred)
    
    # CRITICAL: Only calculate CI if n_bootstrap > 0
    if n_bootstrap > 0:
        # Bootstrap with progress for long runs
        cod_ci = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap)
        csmf_ci = bootstrap_metric(y_true, y_pred, calculate_csmf_accuracy, n_bootstrap)
        
        # PATTERN: Return CI in list format expected by ray_tasks.py
        metrics = {
            "cod_accuracy": cod_accuracy,
            "cod_accuracy_ci": list(cod_ci),  # Convert tuple to list
            "csmf_accuracy": csmf_accuracy,
            "csmf_accuracy_ci": list(csmf_ci),
            # BACKWARD COMPATIBILITY: Keep old format
            "cod_accuracy_ci_lower": cod_ci[0],
            "cod_accuracy_ci_upper": cod_ci[1],
            "csmf_accuracy_ci_lower": csmf_ci[0],
            "csmf_accuracy_ci_upper": csmf_ci[1],
        }
    else:
        # No bootstrap requested
        metrics = {
            "cod_accuracy": cod_accuracy,
            "cod_accuracy_ci": None,
            "csmf_accuracy": csmf_accuracy,
            "csmf_accuracy_ci": None,
        }
    
    # Add per-cause metrics if space allows (existing logic)
    if len(np.unique(y_true)) <= 10:
        cause_accuracies = calculate_per_cause_accuracy(y_true, y_pred)
        for cause, acc in cause_accuracies.items():
            metrics[f"accuracy_{cause}"] = acc
    
    return metrics

# Task 1 - Enhance bootstrap_metric with progress
def bootstrap_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    # PATTERN: Use consistent random state for reproducibility
    rng = np.random.RandomState(42)
    
    scores = []
    n_samples = len(y_true)
    
    # CRITICAL: Handle pandas Series/numpy array mismatch
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Progress indication for long runs (optional, if tqdm available)
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_bootstrap), desc="Bootstrap CI", disable=n_bootstrap < 50)
    except ImportError:
        iterator = range(n_bootstrap)
    
    for _ in iterator:
        # Resample with replacement
        indices = rng.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            scores.append(score)
        except Exception:
            # GOTCHA: Skip if metric fails (e.g., missing classes in resample)
            continue
    
    if not scores:
        # CRITICAL: Return sensible defaults if all bootstrap samples failed
        return (0.0, 0.0)
    
    # Calculate percentile confidence intervals
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    
    return float(lower), float(upper)
```

### Integration Points
```yaml
METRICS:
  - file: model_comparison/metrics/comparison_metrics.py
  - changes: Return CI in list format, maintain backward compatibility
  
RAY_TASKS:
  - file: model_comparison/orchestration/ray_tasks.py
  - changes: NONE - already expects list format
  
TESTS:
  - file: model_comparison/tests/test_metrics.py
  - changes: Add new tests for CI format and edge cases
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
cd /Users/ericliu/projects5/context-engineering-intro
poetry run ruff check model_comparison/metrics/comparison_metrics.py --fix
poetry run mypy model_comparison/metrics/comparison_metrics.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# ADD to test_metrics.py:
def test_calculate_metrics_with_bootstrap_ci_list_format(sample_predictions):
    """Test that calculate_metrics returns CI in list format."""
    y_true, y_pred = sample_predictions
    
    # Calculate with bootstrap
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=50)
    
    # Check CI is in list format
    assert isinstance(metrics["cod_accuracy_ci"], list)
    assert len(metrics["cod_accuracy_ci"]) == 2
    assert isinstance(metrics["csmf_accuracy_ci"], list)
    assert len(metrics["csmf_accuracy_ci"]) == 2
    
    # Check values are sensible
    assert metrics["cod_accuracy_ci"][0] <= metrics["cod_accuracy"]
    assert metrics["cod_accuracy"] <= metrics["cod_accuracy_ci"][1]

def test_calculate_metrics_without_bootstrap():
    """Test that CI is None when n_bootstrap=0."""
    y_true = pd.Series(["A", "B", "A", "B"])
    y_pred = np.array(["A", "B", "A", "B"])
    
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=0)
    
    assert metrics["cod_accuracy_ci"] is None
    assert metrics["csmf_accuracy_ci"] is None

def test_bootstrap_with_small_sample():
    """Test bootstrap handles small samples gracefully."""
    # Only 3 samples - bootstrap should still work
    y_true = pd.Series(["A", "B", "A"])
    y_pred = np.array(["A", "B", "B"])
    
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=20)
    
    assert metrics["cod_accuracy_ci"] is not None
    assert 0 <= metrics["cod_accuracy_ci"][0] <= metrics["cod_accuracy_ci"][1] <= 1

def test_bootstrap_reproducibility():
    """Test that bootstrap results are reproducible."""
    y_true = pd.Series(["A"] * 50 + ["B"] * 50)
    y_pred = np.array(["A"] * 45 + ["B"] * 55)
    
    # Run twice - should get same results
    metrics1 = calculate_metrics(y_true, y_pred, n_bootstrap=100)
    metrics2 = calculate_metrics(y_true, y_pred, n_bootstrap=100)
    
    assert metrics1["cod_accuracy_ci"] == metrics2["cod_accuracy_ci"]
    assert metrics1["csmf_accuracy_ci"] == metrics2["csmf_accuracy_ci"]
```

```bash
# Run and iterate until passing:
cd /Users/ericliu/projects5/context-engineering-intro
poetry run pytest model_comparison/tests/test_metrics.py::test_calculate_metrics_with_bootstrap_ci_list_format -v
poetry run pytest model_comparison/tests/test_metrics.py::test_calculate_metrics_without_bootstrap -v
poetry run pytest model_comparison/tests/test_metrics.py::test_bootstrap_with_small_sample -v
poetry run pytest model_comparison/tests/test_metrics.py::test_bootstrap_reproducibility -v

# Run all metric tests
poetry run pytest model_comparison/tests/test_metrics.py -v
```

### Level 3: Integration Test
```python
# Create a simple integration test
# test_bootstrap_integration.py
import pandas as pd
import numpy as np
from model_comparison.metrics.comparison_metrics import calculate_metrics
from model_comparison.orchestration.config import ExperimentResult

def test_metrics_integration_with_experiment_result():
    """Test that metrics integrate properly with ExperimentResult."""
    # Create test data
    y_true = pd.Series(["cause_1"] * 30 + ["cause_2"] * 30 + ["cause_3"] * 40)
    y_pred = np.array(["cause_1"] * 25 + ["cause_2"] * 35 + ["cause_3"] * 40)
    
    # Calculate metrics with bootstrap
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=100)
    
    # Create ExperimentResult - this should work without errors
    result = ExperimentResult(
        experiment_id="test_001",
        model_name="test_model",
        experiment_type="bootstrap_test",
        train_site="site_A",
        test_site="site_B",
        csmf_accuracy=metrics["csmf_accuracy"],
        cod_accuracy=metrics["cod_accuracy"],
        csmf_accuracy_ci=metrics.get("csmf_accuracy_ci"),
        cod_accuracy_ci=metrics.get("cod_accuracy_ci"),
        n_train=100,
        n_test=100,
        execution_time_seconds=1.5,
    )
    
    # Verify CI is stored correctly
    assert isinstance(result.csmf_accuracy_ci, list)
    assert len(result.csmf_accuracy_ci) == 2
    assert isinstance(result.cod_accuracy_ci, list)
    assert len(result.cod_accuracy_ci) == 2
```

```bash
# Run integration test
poetry run pytest test_bootstrap_integration.py -v
```

## Final Validation Checklist
- [ ] All tests pass: `poetry run pytest model_comparison/tests/ -v`
- [ ] No linting errors: `poetry run ruff check model_comparison/`
- [ ] No type errors: `poetry run mypy model_comparison/`
- [ ] Bootstrap CI appears in output when n_bootstrap > 0
- [ ] CI format is [lower, upper] list
- [ ] Backward compatibility maintained (old format still available)
- [ ] Performance acceptable for n_bootstrap=100
- [ ] Results are reproducible (same seed = same CI)

---

## Anti-Patterns to Avoid
- ❌ Don't use global random state - use np.random.RandomState(seed)
- ❌ Don't skip bootstrap when it fails - handle edge cases gracefully
- ❌ Don't return separate CI fields only - maintain list format
- ❌ Don't ignore small sample sizes - bootstrap should still work
- ❌ Don't make bootstrap mandatory - respect n_bootstrap=0
- ❌ Don't break backward compatibility without migration plan

---

## PRP Quality Score: 9/10

**Confidence level for one-pass implementation: Very High**

This PRP provides comprehensive context including:
- Exact format mismatch issue identified
- Clear solution with code examples
- Existing patterns from the codebase
- Test cases covering edge scenarios
- Validation steps at multiple levels
- Performance and compatibility considerations

The only reason it's not 10/10 is that we might discover additional edge cases during implementation, but the core solution is well-defined and testable.