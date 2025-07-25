name: "Fix Bootstrap Confidence Intervals - Task IM-052"
description: |

## Purpose
Fix bootstrap confidence intervals in the VA model comparison framework to ensure proper calculation and storage of confidence intervals for CSMF and COD accuracy metrics.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Fix the bootstrap confidence intervals calculation and storage in the model comparison framework so that:
- Bootstrap CIs are properly calculated with the specified `n_bootstrap` iterations
- CIs are stored in the correct format expected by `ExperimentResult`
- Results CSV files contain properly formatted CI columns
- The framework supports configurable bootstrap iterations (100-1000)

## Why
- **Statistical Rigor**: Confidence intervals provide crucial uncertainty estimates for model comparisons
- **Decision Support**: CIs help determine if performance differences between models are statistically significant
- **Research Validity**: Proper bootstrap implementation is essential for VA research publications
- **Framework Completeness**: This fix completes the statistical validation capabilities of the comparison framework

## What
The system should:
- Calculate bootstrap confidence intervals for CSMF and COD accuracy metrics
- Store CIs as `[lower, upper]` lists in `ExperimentResult` 
- Support configurable bootstrap iterations (default 100, up to 1000 for production)
- Log bootstrap calculation progress for debugging
- Handle edge cases gracefully (small samples, degenerate distributions)

### Success Criteria
- [ ] Running `run_distributed_comparison.py` produces results with non-null CI values
- [ ] CI values are stored as lists in the format `[lower_bound, upper_bound]`
- [ ] The CI contains the point estimate (lower ≤ point estimate ≤ upper)
- [ ] Unit tests pass for various bootstrap configurations
- [ ] No performance regression from the additional bootstrap calculations

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
  why: For understanding the base accuracy metric used in COD accuracy
  
- url: https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-9-28
  why: Robust metrics for VA - defines CSMF accuracy metric and validation approaches
  
- url: https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
  why: Best practices for bootstrap CI implementation in Python
  
- file: /Users/ericliu/projects5/context-engineering-intro/model_comparison/metrics/comparison_metrics.py
  why: Current implementation that needs to be modified
  
- file: /Users/ericliu/projects5/context-engineering-intro/model_comparison/orchestration/ray_tasks.py
  why: Consumer of metrics that needs simplification
  
- file: /Users/ericliu/projects5/context-engineering-intro/model_comparison/orchestration/config.py
  why: ExperimentResult class defining expected CI format
```

### Current Codebase tree (relevant portions)
```bash
context-engineering-intro/
├── model_comparison/
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── comparison_metrics.py  # Returns CI as separate keys
│   ├── orchestration/
│   │   ├── config.py  # ExperimentResult expects List[float] for CIs
│   │   └── ray_tasks.py  # Complex conditional logic for CI handling
│   └── tests/
│       └── test_metrics.py  # Existing test patterns to follow
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Bootstrap implementation considerations
# 1. scikit-learn doesn't provide built-in bootstrap CI functions - we implement our own
# 2. Random seed must be set for reproducibility (already done with RandomState(42))
# 3. Bootstrap samples can fail for degenerate cases (single class, empty data)
# 4. Number of bootstrap iterations affects both accuracy and performance
# 5. VA-specific: CSMF accuracy can be sensitive to class distribution in bootstrap samples

# Current bug: Mismatch between metric output format and expected format
# metrics returns: {"csmf_accuracy_ci_lower": 0.8, "csmf_accuracy_ci_upper": 0.9}
# ExperimentResult expects: {"csmf_accuracy_ci": [0.8, 0.9]}
```

## Implementation Blueprint

### Data models and structure

The key data structures that need alignment:

```python
# Current (incorrect) metrics output
metrics = {
    "cod_accuracy": 0.85,
    "cod_accuracy_ci_lower": 0.82,
    "cod_accuracy_ci_upper": 0.88,
    "csmf_accuracy": 0.75,
    "csmf_accuracy_ci_lower": 0.70,
    "csmf_accuracy_ci_upper": 0.80,
}

# Required (correct) metrics output  
metrics = {
    "cod_accuracy": 0.85,
    "cod_accuracy_ci": [0.82, 0.88],  # List format
    "csmf_accuracy": 0.75,
    "csmf_accuracy_ci": [0.70, 0.80],  # List format
}

# ExperimentResult model (from config.py)
class ExperimentResult(BaseModel):
    csmf_accuracy_ci: Optional[List[float]] = Field(
        default=None, description="CSMF accuracy confidence interval"
    )
    cod_accuracy_ci: Optional[List[float]] = Field(
        default=None, description="COD accuracy confidence interval"
    )
```

### List of tasks to be completed to fulfill the PRP

```yaml
Task 1: Update calculate_metrics to return CI as lists
MODIFY model_comparison/metrics/comparison_metrics.py:
  - FIND pattern: 'metrics = {' block around line 40
  - REPLACE separate CI keys with list format
  - ADD debug logging for bootstrap progress
  - PRESERVE existing metrics calculations

Task 2: Simplify ray_tasks.py CI handling
MODIFY model_comparison/orchestration/ray_tasks.py:
  - FIND pattern: Complex conditional CI assignment around line 140
  - REPLACE with direct assignment from metrics dict
  - REMOVE unnecessary type checking
  - ADD validation that CIs are properly formatted

Task 3: Add comprehensive unit tests
MODIFY model_comparison/tests/test_metrics.py:
  - ADD test for CI list format
  - ADD test for various bootstrap iterations
  - ADD test for edge cases (single class, small samples)
  - VERIFY CI contains point estimate

Task 4: Add integration test
CREATE model_comparison/tests/test_bootstrap_integration.py:
  - TEST end-to-end CI calculation through ray_tasks
  - VERIFY ExperimentResult contains proper CI format
  - TEST with different n_bootstrap values

Task 5: Update existing tests if needed
CHECK model_comparison/tests/:
  - UPDATE any tests expecting old CI format
  - ENSURE backward compatibility if needed
```

### Per task pseudocode

```python
# Task 1: Update calculate_metrics
def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
) -> Dict[str, float]:
    """Calculate comprehensive metrics with confidence intervals."""
    # PATTERN: Log bootstrap start (see baseline/utils logger pattern)
    logger = get_logger(__name__)
    logger.debug(f"Starting bootstrap with {n_bootstrap} iterations")
    
    # Basic metrics (unchanged)
    cod_accuracy = accuracy_score(y_true, y_pred)
    csmf_accuracy = calculate_csmf_accuracy(y_true, y_pred)

    # Bootstrap confidence intervals (unchanged calculation)
    cod_ci = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap)
    csmf_ci = bootstrap_metric(y_true, y_pred, calculate_csmf_accuracy, n_bootstrap)
    
    # CRITICAL: Return CI as lists, not separate keys
    metrics = {
        "cod_accuracy": cod_accuracy,
        "cod_accuracy_ci": [cod_ci[0], cod_ci[1]],  # List format
        "csmf_accuracy": csmf_accuracy,
        "csmf_accuracy_ci": [csmf_ci[0], csmf_ci[1]],  # List format
    }
    
    # Add per-cause metrics if space allows (unchanged)
    if len(np.unique(y_true)) <= 10:
        cause_accuracies = calculate_per_cause_accuracy(y_true, y_pred)
        for cause, acc in cause_accuracies.items():
            metrics[f"accuracy_{cause}"] = acc
    
    logger.debug(f"Bootstrap complete. COD CI: {metrics['cod_accuracy_ci']}, "
                f"CSMF CI: {metrics['csmf_accuracy_ci']}")
    
    return metrics

# Task 2: Simplify ray_tasks.py
# In train_and_evaluate_model function around line 130:
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=y_test, y_pred=y_pred, y_proba=y_proba, n_bootstrap=n_bootstrap
    )
    
    # Create result - SIMPLIFIED CI assignment
    result = ExperimentResult(
        experiment_id=experiment_metadata["experiment_id"],
        model_name=model_name,
        # ... other fields ...
        csmf_accuracy=metrics["csmf_accuracy"],
        cod_accuracy=metrics["cod_accuracy"],
        csmf_accuracy_ci=metrics.get("csmf_accuracy_ci"),  # Direct assignment
        cod_accuracy_ci=metrics.get("cod_accuracy_ci"),    # Direct assignment
        # ... rest of fields ...
    )
    
    # PATTERN: Add validation (defensive programming)
    if result.csmf_accuracy_ci:
        assert len(result.csmf_accuracy_ci) == 2, "CI must be [lower, upper]"
        assert result.csmf_accuracy_ci[0] <= result.csmf_accuracy <= result.csmf_accuracy_ci[1]

# Task 3: Unit test example
def test_calculate_metrics_ci_format(sample_predictions):
    """Test that CIs are returned as lists."""
    y_true, y_pred = sample_predictions
    
    # Calculate with bootstrap
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=50)
    
    # Check CI format
    assert "cod_accuracy_ci" in metrics
    assert isinstance(metrics["cod_accuracy_ci"], list)
    assert len(metrics["cod_accuracy_ci"]) == 2
    
    # Check CI bounds make sense
    lower, upper = metrics["cod_accuracy_ci"]
    assert 0 <= lower <= upper <= 1
    assert lower <= metrics["cod_accuracy"] <= upper
```

### Integration Points
```yaml
METRICS:
  - location: model_comparison/metrics/comparison_metrics.py
  - change: Return CI as lists instead of separate keys
  - impact: All consumers of calculate_metrics need to handle list format
  
RAY_TASKS:
  - location: model_comparison/orchestration/ray_tasks.py
  - change: Remove conditional CI handling, use direct assignment
  - impact: Simplifies code, removes potential bugs
  
LOGGING:
  - add to: calculate_metrics and bootstrap_metric functions
  - pattern: "logger.debug(f'Bootstrap iteration {i}/{n_bootstrap}')"
  - purpose: Track progress for long-running bootstraps
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
cd /Users/ericliu/projects5/context-engineering-intro
poetry run ruff check model_comparison/metrics/comparison_metrics.py --fix
poetry run mypy model_comparison/metrics/comparison_metrics.py
poetry run mypy model_comparison/orchestration/ray_tasks.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# Test cases to add to test_metrics.py:

def test_bootstrap_ci_list_format():
    """Verify CIs are returned as lists."""
    # Setup
    y_true = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    y_pred = y_true.copy()
    y_pred.iloc[0:10] = "B"  # Introduce some errors
    
    # Test
    metrics = calculate_metrics(y_true, y_pred, n_bootstrap=100)
    
    # Assertions
    assert isinstance(metrics["cod_accuracy_ci"], list)
    assert isinstance(metrics["csmf_accuracy_ci"], list)
    assert len(metrics["cod_accuracy_ci"]) == 2
    assert len(metrics["csmf_accuracy_ci"]) == 2

def test_bootstrap_iterations_configurable():
    """Test different bootstrap iteration counts."""
    y_true = pd.Series(np.random.choice(["A", "B", "C"], 100))
    y_pred = y_true.copy()
    
    # Test with different iterations
    for n_boot in [10, 100, 500]:
        metrics = calculate_metrics(y_true, y_pred, n_bootstrap=n_boot)
        assert "cod_accuracy_ci" in metrics
        assert "csmf_accuracy_ci" in metrics

def test_bootstrap_edge_cases():
    """Test edge cases that might break bootstrap."""
    # Single class
    y_single = pd.Series(["A"] * 50)
    metrics = calculate_metrics(y_single, y_single.values, n_bootstrap=20)
    assert metrics["cod_accuracy"] == 1.0
    assert metrics["cod_accuracy_ci"] == [1.0, 1.0]
    
    # Very small sample
    y_small = pd.Series(["A", "B", "A"])
    y_pred = np.array(["A", "A", "B"])
    metrics = calculate_metrics(y_small, y_pred, n_bootstrap=10)
    assert "cod_accuracy_ci" in metrics
```

```bash
# Run unit tests
poetry run pytest model_comparison/tests/test_metrics.py::test_bootstrap_ci_list_format -v
poetry run pytest model_comparison/tests/test_metrics.py::test_bootstrap_iterations_configurable -v
poetry run pytest model_comparison/tests/test_metrics.py::test_bootstrap_edge_cases -v

# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Create a small test experiment to verify end-to-end
cd /Users/ericliu/projects5/context-engineering-intro

# Run a minimal distributed comparison
poetry run python model_comparison/scripts/run_distributed_comparison.py \
  --data-path data/va34_data.csv \
  --sites site_1 site_2 \
  --models xgboost \
  --training-sizes 0.5 \
  --n-bootstrap 50 \
  --n-workers 2 \
  --output-dir results/test_bootstrap

# Verify results contain CIs
poetry run python -c "
import pandas as pd
df = pd.read_csv('results/test_bootstrap/model_comparison_results.csv')
print('Columns:', df.columns.tolist())
print('Sample CI values:')
print(df[['csmf_accuracy_ci', 'cod_accuracy_ci']].head())
# CIs should be lists like '[0.7, 0.8]'
"

# Expected: Non-null CI values in list format
```

## Final Validation Checklist
- [ ] All tests pass: `poetry run pytest model_comparison/tests/test_metrics.py -v`
- [ ] No linting errors: `poetry run ruff check model_comparison/ --fix`
- [ ] No type errors: `poetry run mypy model_comparison/metrics/ model_comparison/orchestration/`
- [ ] Integration test produces CIs: Check results CSV has proper CI columns
- [ ] CIs contain point estimates: Lower ≤ point ≤ upper for all metrics
- [ ] Performance acceptable: Bootstrap doesn't significantly slow experiments
- [ ] Edge cases handled: Single class, small samples don't crash

---

## Anti-Patterns to Avoid
- ❌ Don't change the bootstrap algorithm itself - just the output format
- ❌ Don't remove the RandomState(42) - needed for reproducibility
- ❌ Don't ignore edge cases - handle them gracefully
- ❌ Don't make breaking changes - maintain backward compatibility where possible
- ❌ Don't hardcode bootstrap iterations - keep it configurable
- ❌ Don't skip validation that CIs make mathematical sense

## Confidence Score: 9/10

This PRP provides comprehensive context for fixing the bootstrap confidence intervals issue. The solution is straightforward (changing the output format), the existing code patterns are clear, and the validation steps are thorough. The only minor uncertainty is around potential backward compatibility needs, but the solution handles this gracefully.