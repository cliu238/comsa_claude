name: "Hyperparameter Tuning for All ML Models - IM-053"
description: |

## Purpose
Implement comprehensive hyperparameter tuning for XGBoost, Random Forest, and Logistic Regression models in the VA pipeline, integrating with existing Ray distributed infrastructure and achieving 10-30% performance improvement.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a flexible, distributed hyperparameter tuning system that seamlessly integrates with the existing model_comparison framework, supporting multiple optimization methods (Grid Search, Random Search, Bayesian Optimization with Optuna, and Ray Tune) while maintaining backward compatibility.

## Why
- **Business value**: 10-30% improvement in CSMF accuracy can significantly improve public health decision-making
- **Integration**: Enhances existing model_comparison pipeline without disrupting current workflows
- **Problems solved**: 
  - Default hyperparameters are suboptimal for VA data characteristics
  - Manual tuning is time-consuming and doesn't explore the full parameter space
  - Different sites may benefit from different hyperparameters

## What
### User-visible behavior:
- New CLI flags in `run_distributed_comparison.py`:
  - `--tune-hyperparameters`: Enable hyperparameter tuning
  - `--tuning-method`: Choose optimization method (grid, random, optuna, ray_tune)
  - `--tuning-trials`: Number of trials for optimization
  - `--tuning-timeout`: Maximum time for tuning per model
  - `--tuning-metric`: Metric to optimize (csmf_accuracy or cod_accuracy)

### Technical requirements:
- Distributed tuning using existing Ray infrastructure
- Support for all three ML models (XGBoost, Random Forest, Logistic Regression)
- Cache tuned parameters for reproducibility
- Progress monitoring and early stopping
- Integration with ExperimentResult for tracking

### Success Criteria
- [ ] All three models support hyperparameter tuning
- [ ] 10-30% improvement in CSMF accuracy demonstrated
- [ ] Tuning completes within reasonable time (< 30 min for 100 trials)
- [ ] Backward compatibility maintained
- [ ] Results are reproducible with cached parameters
- [ ] All tests pass including new hyperparameter tuning tests

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_pruning.html
  why: Optuna pruning callbacks for early stopping unpromising trials
  
- url: https://docs.ray.io/en/latest/tune/index.html
  why: Ray Tune documentation for distributed hyperparameter optimization
  
- url: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html
  why: Parameter sampling strategies for random search
  
- file: /Users/ericliu/projects5/context-engineering-intro/baseline/models/hyperparameter_tuning.py
  why: Existing XGBoost hyperparameter tuning implementation to extend
  
- file: /Users/ericliu/projects5/context-engineering-intro/model_comparison/orchestration/ray_tasks.py
  why: Ray task patterns for distributed execution
  
- file: /Users/ericliu/projects5/context-engineering-intro/model_comparison/experiments/experiment_config.py
  why: ExperimentConfig structure to extend with tuning parameters

- doc: https://xgboost.readthedocs.io/en/stable/parameter.html
  section: Parameters for Tree Booster
  critical: Understanding parameter interactions and valid ranges

- doc: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  section: Parameters
  critical: Random Forest specific parameters and their impact
```

### Current Codebase tree
```bash
model_comparison/
├── experiments/
│   ├── experiment_config.py
│   ├── parallel_experiment.py
│   └── site_comparison.py
├── orchestration/
│   ├── config.py
│   ├── prefect_flows.py
│   └── ray_tasks.py
├── scripts/
│   └── run_distributed_comparison.py
└── tests/
    └── test_ray_tasks.py

baseline/
├── models/
│   ├── hyperparameter_tuning.py  # Existing XGBoost tuning
│   ├── xgboost_config.py
│   ├── xgboost_model.py
│   ├── random_forest_config.py
│   ├── random_forest_model.py
│   ├── logistic_regression_config.py
│   └── logistic_regression_model.py
└── metrics/
    └── va_metrics.py
```

### Desired Codebase tree with files to be added
```bash
model_comparison/
├── hyperparameter_tuning/  # NEW PACKAGE
│   ├── __init__.py
│   ├── search_spaces.py      # Model-specific search space definitions
│   ├── tuner.py             # Base tuner interface
│   ├── grid_tuner.py        # Grid search implementation
│   ├── random_tuner.py      # Random search implementation  
│   ├── optuna_tuner.py      # Optuna Bayesian optimization
│   ├── ray_tuner.py         # Ray Tune integration
│   └── utils.py             # Helper functions for caching, progress
├── experiments/
│   └── experiment_config.py  # MODIFIED: Add HyperparameterSearchConfig
├── orchestration/
│   └── ray_tasks.py         # MODIFIED: Add tuning phase
└── tests/
    └── test_hyperparameter_tuning.py  # NEW: Comprehensive tests

baseline/
├── models/
│   ├── hyperparameter_tuning.py  # MODIFIED: Generalize for all models
│   └── model_factory.py     # NEW: Factory for creating models with configs
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Optuna requires objective function to return a single float
# The function should return negative values for metrics to maximize

# CRITICAL: Ray Tune requires specific imports inside remote functions
# All model imports must be inside the remote function, not at module level

# CRITICAL: XGBoost n_jobs=-1 conflicts with Ray's parallelism
# Set n_jobs=1 when using Ray for distributed tuning

# CRITICAL: Logistic Regression solver compatibility
# 'saga' solver supports all penalties, 'liblinear' only l1/l2
# When penalty='elasticnet', must use solver='saga' and provide l1_ratio

# CRITICAL: Random Forest max_features validation
# If max_features > n_features, sklearn will raise ValueError
# Always validate max_features against actual feature count

# CRITICAL: Class imbalance in VA data
# Some causes are rare, stratified CV may fail
# Implement fallback to regular KFold when StratifiedKFold fails
```

## Implementation Blueprint

### Data models and structure

```python
# search_spaces.py - Define search spaces for each model
from typing import Dict, List, Union, Any
from pydantic import BaseModel, Field

class SearchSpace(BaseModel):
    """Base class for hyperparameter search spaces."""
    name: str
    type: str  # 'int', 'float', 'categorical'
    values: Union[List[Any], Dict[str, Any]]  # List for categorical, dict for ranges
    
class ModelSearchSpace(BaseModel):
    """Complete search space for a model."""
    model_name: str
    parameters: Dict[str, SearchSpace]
    
# experiment_config.py - Extend with tuning configuration  
class HyperparameterSearchConfig(BaseModel):
    """Configuration for hyperparameter search."""
    enabled: bool = Field(default=False)
    method: str = Field(default="optuna", pattern="^(grid|random|optuna|ray_tune)$")
    n_trials: int = Field(default=50, ge=1)
    timeout_seconds: Optional[float] = Field(default=1800)  # 30 minutes
    metric: str = Field(default="csmf_accuracy")
    cv_folds: int = Field(default=5, ge=2)
    cache_dir: str = Field(default="cache/tuned_params")
    
# Update ExperimentConfig
class ExperimentConfig(BaseModel):
    # ... existing fields ...
    hyperparameter_search: Optional[HyperparameterSearchConfig] = Field(
        default=None, description="Hyperparameter tuning configuration"
    )
```

### List of tasks to be completed in order

```yaml
Task 1:
CREATE model_comparison/hyperparameter_tuning/__init__.py:
  - Export main interfaces: BaseTuner, get_tuner, SearchSpace
  - Follow pattern from model_comparison/__init__.py

Task 2:  
CREATE model_comparison/hyperparameter_tuning/search_spaces.py:
  - Define SearchSpace and ModelSearchSpace Pydantic models
  - Create get_search_space(model_name: str) -> ModelSearchSpace function
  - Define specific search spaces for xgboost, random_forest, logistic_regression
  - Include parameter validation and compatibility checks

Task 3:
CREATE model_comparison/hyperparameter_tuning/tuner.py:
  - Define abstract BaseTuner class with tune() method
  - Create TuningResult dataclass for storing results
  - Implement parameter caching/loading utilities
  - Add progress reporting interface

Task 4:
CREATE model_comparison/hyperparameter_tuning/grid_tuner.py:
  - Implement GridSearchTuner(BaseTuner)
  - Use sklearn ParameterGrid for combinations
  - Integrate with Ray for parallel evaluation
  - Add early stopping for poor performers

Task 5:
CREATE model_comparison/hyperparameter_tuning/optuna_tuner.py:
  - Implement OptunaTuner(BaseTuner) 
  - Extend existing baseline/models/hyperparameter_tuning.py patterns
  - Support all three model types
  - Add multi-objective optimization option
  - Implement pruning callbacks

Task 6:
MODIFY model_comparison/experiments/experiment_config.py:
  - Add HyperparameterSearchConfig class
  - Update ExperimentConfig with hyperparameter_search field
  - Add validators for config compatibility

Task 7:
MODIFY model_comparison/orchestration/ray_tasks.py:
  - Add tune_hyperparameters remote function
  - Integrate tuning phase before model training
  - Cache tuned parameters by experiment_id
  - Update train_and_evaluate_model to use tuned params

Task 8:
CREATE baseline/models/model_factory.py:
  - Implement create_model(name, config) factory function
  - Support creating models with custom hyperparameters
  - Handle config validation for each model type

Task 9:
MODIFY model_comparison/scripts/run_distributed_comparison.py:
  - Add CLI arguments for hyperparameter tuning
  - Update argument parsing and validation
  - Pass tuning config to experiment

Task 10:
CREATE model_comparison/tests/test_hyperparameter_tuning.py:
  - Test search space generation for each model
  - Test tuner implementations with small datasets
  - Test caching and parameter loading
  - Test integration with ray_tasks

Task 11:
CREATE model_comparison/hyperparameter_tuning/utils.py:
  - Implement parameter caching functions
  - Add progress monitoring utilities
  - Create validation helpers
  - Add result visualization functions
```

### Per task pseudocode

```python
# Task 2: search_spaces.py core logic
def get_search_space(model_name: str) -> ModelSearchSpace:
    """Get search space for a specific model."""
    if model_name == "xgboost":
        return ModelSearchSpace(
            model_name="xgboost",
            parameters={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    values={"low": 100, "high": 500}
                ),
                "max_depth": SearchSpace(
                    name="max_depth", 
                    type="int",
                    values={"low": 3, "high": 12}
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="float", 
                    values={"low": 0.01, "high": 0.3, "log": True}
                ),
                # ... more parameters
            }
        )
    # Similar for other models

# Task 5: optuna_tuner.py core logic
class OptunaTuner(BaseTuner):
    def objective(self, trial: Trial) -> float:
        # PATTERN: Get hyperparameters from trial
        params = self._suggest_params(trial)
        
        # CRITICAL: Create model with suggested params
        model = self.model_factory.create(self.model_name, params)
        
        # PATTERN: Use cross-validation for robust evaluation
        cv_scores = cross_validate(
            model, self.X, self.y, 
            cv=self.cv_folds,
            scoring=self.scoring_func
        )
        
        # GOTCHA: Return negative for maximization
        return -cv_scores["test_score"].mean()
        
    def tune(self) -> TuningResult:
        # PATTERN: Create study with proper direction
        study = optuna.create_study(
            direction="minimize",  # Because we negate scores
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner()  # Early stopping
        )
        
        # CRITICAL: Handle exceptions in optimization
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
                callbacks=[self._progress_callback]
            )
        except Exception as e:
            logger.warning(f"Optimization stopped: {e}")
            
        # PATTERN: Extract and cache results
        best_params = study.best_params
        best_score = -study.best_value
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=len(study.trials),
            study=study
        )

# Task 7: ray_tasks.py modification
@ray.remote
def tune_hyperparameters(
    model_name: str,
    train_data: Tuple[pd.DataFrame, pd.Series],
    tuning_config: Dict,
    experiment_id: str
) -> Dict[str, Any]:
    """Tune hyperparameters for a model using Ray."""
    # CRITICAL: Import inside remote function
    from model_comparison.hyperparameter_tuning import get_tuner
    from baseline.utils import get_logger
    
    logger = get_logger(__name__)
    X_train, y_train = train_data
    
    # PATTERN: Check cache first
    cache_key = f"{experiment_id}_{model_name}"
    cached_params = load_cached_params(cache_key)
    if cached_params:
        logger.info(f"Using cached parameters for {model_name}")
        return cached_params
        
    # GOTCHA: Set n_jobs=1 to avoid nested parallelism
    tuning_config["n_jobs"] = 1
    
    # Create and run tuner
    tuner = get_tuner(
        method=tuning_config["method"],
        model_name=model_name,
        X=X_train,
        y=y_train,
        **tuning_config
    )
    
    result = tuner.tune()
    
    # Cache results
    save_cached_params(cache_key, result.best_params)
    
    return {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "tuning_time_seconds": result.duration_seconds
    }
```

### Integration Points
```yaml
CLI:
  - add to: model_comparison/scripts/run_distributed_comparison.py
  - pattern: |
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning before training"
    )
    parser.add_argument(
        "--tuning-method",
        choices=["grid", "random", "optuna", "ray_tune"],
        default="optuna",
        help="Hyperparameter tuning method"
    )
  
CONFIG:
  - modify: model_comparison/experiments/experiment_config.py
  - add: HyperparameterSearchConfig class
  - update: ExperimentConfig to include hyperparameter_search field
  
RAY_TASKS:
  - modify: model_comparison/orchestration/ray_tasks.py
  - inject: Tuning phase before train_and_evaluate_model
  - pattern: Check if tuning enabled, run tune_hyperparameters, pass params
  
CACHING:
  - create dir: cache/tuned_params/
  - pattern: Use experiment_id + model_name as cache key
  - format: JSON files with parameters and metadata
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
cd /Users/ericliu/projects5/context-engineering-intro

# Check new files
poetry run ruff check model_comparison/hyperparameter_tuning/ --fix
poetry run mypy model_comparison/hyperparameter_tuning/

# Check modified files  
poetry run ruff check model_comparison/experiments/experiment_config.py
poetry run mypy model_comparison/experiments/experiment_config.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# test_hyperparameter_tuning.py key tests
import pytest
from model_comparison.hyperparameter_tuning import get_search_space, get_tuner
from model_comparison.hyperparameter_tuning.search_spaces import ModelSearchSpace

def test_search_space_generation():
    """Test that search spaces are generated correctly for each model."""
    for model_name in ["xgboost", "random_forest", "logistic_regression"]:
        space = get_search_space(model_name)
        assert isinstance(space, ModelSearchSpace)
        assert space.model_name == model_name
        assert len(space.parameters) > 0

def test_tuner_factory():
    """Test tuner creation for different methods."""
    X_dummy = pd.DataFrame(np.random.rand(100, 10))
    y_dummy = pd.Series(np.random.randint(0, 5, 100))
    
    for method in ["grid", "random", "optuna"]:
        tuner = get_tuner(
            method=method,
            model_name="xgboost",
            X=X_dummy,
            y=y_dummy,
            n_trials=5
        )
        assert tuner is not None
        
def test_optuna_tuner_with_small_data():
    """Test Optuna tuner runs successfully."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=20, n_classes=5)
    
    tuner = get_tuner(
        method="optuna",
        model_name="xgboost", 
        X=pd.DataFrame(X),
        y=pd.Series(y),
        n_trials=10,
        cv_folds=3
    )
    
    result = tuner.tune()
    assert result.best_params is not None
    assert result.best_score > 0
    assert result.n_trials_completed <= 10

def test_parameter_caching():
    """Test that parameters are cached and loaded correctly."""
    from model_comparison.hyperparameter_tuning.utils import (
        save_cached_params, load_cached_params
    )
    
    test_params = {"n_estimators": 200, "max_depth": 8}
    cache_key = "test_experiment_xgboost"
    
    # Save and load
    save_cached_params(cache_key, test_params)
    loaded = load_cached_params(cache_key)
    
    assert loaded == test_params

def test_integration_with_ray_task():
    """Test hyperparameter tuning integrates with Ray."""
    import ray
    from model_comparison.orchestration.ray_tasks import tune_hyperparameters
    
    if not ray.is_initialized():
        ray.init(local_mode=True)  # Local mode for testing
        
    X = pd.DataFrame(np.random.rand(100, 10)) 
    y = pd.Series(np.random.randint(0, 5, 100))
    
    tuning_config = {
        "method": "optuna",
        "n_trials": 5,
        "cv_folds": 3,
        "metric": "accuracy"
    }
    
    future = tune_hyperparameters.remote(
        "xgboost", 
        (X, y),
        tuning_config,
        "test_exp_001" 
    )
    
    result = ray.get(future)
    assert "best_params" in result
    assert "best_score" in result
```

```bash
# Run tests
poetry run pytest model_comparison/tests/test_hyperparameter_tuning.py -v

# Run with coverage
poetry run pytest model_comparison/tests/test_hyperparameter_tuning.py --cov=model_comparison.hyperparameter_tuning -v
```

### Level 3: Integration Test
```bash
# Test with small dataset to verify integration
cd /Users/ericliu/projects5/context-engineering-intro

# Create test data if needed
poetry run python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
n_samples = 500
n_features = 50
X = pd.DataFrame(np.random.rand(n_samples, n_features))
X['site'] = np.random.choice(['site1', 'site2'], n_samples)
X['cause'] = np.random.choice(['Cause_' + str(i) for i in range(10)], n_samples)
X.to_csv('test_va_data.csv', index=False)
print('Test data created')
"

# Run distributed comparison with hyperparameter tuning
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path test_va_data.csv \
    --sites site1 site2 \
    --models xgboost random_forest \
    --training-sizes 0.5 1.0 \
    --tune-hyperparameters \
    --tuning-method optuna \
    --tuning-trials 20 \
    --n-workers 2 \
    --output-dir results/tuning_test

# Expected: Should complete successfully with tuning results logged
# Check logs for "Best parameters found" messages
# Verify results saved to results/tuning_test/
```

### Level 4: Performance Validation
```bash
# Compare performance with and without tuning
poetry run python -c "
import pandas as pd
import json

# Load results
with_tuning = pd.read_csv('results/tuning_test/experiment_results.csv')
without_tuning = pd.read_csv('results/baseline/experiment_results.csv')  # Previous run

# Compare CSMF accuracy
tuned_avg = with_tuning['csmf_accuracy'].mean()
default_avg = without_tuning['csmf_accuracy'].mean()

improvement = (tuned_avg - default_avg) / default_avg * 100
print(f'Average CSMF Accuracy:')
print(f'  Default parameters: {default_avg:.3f}')
print(f'  Tuned parameters: {tuned_avg:.3f}')
print(f'  Improvement: {improvement:.1f}%')

# Check if meets success criteria
if improvement >= 10:
    print('✓ Success: Achieved target improvement!')
else:
    print('⚠ Warning: Improvement below 10% target')
"
```

## Final Validation Checklist
- [ ] All tests pass: `poetry run pytest model_comparison/tests/`
- [ ] No linting errors: `poetry run ruff check model_comparison/`
- [ ] No type errors: `poetry run mypy model_comparison/`
- [ ] CLI integration works: Test all new flags
- [ ] Caching works correctly: Check cache/tuned_params/
- [ ] Performance improvement demonstrated: >= 10% on test data
- [ ] Backward compatibility: Old commands still work
- [ ] Progress monitoring: Tuning progress displayed clearly
- [ ] Documentation updated: README includes tuning examples

---

## Anti-Patterns to Avoid
- ❌ Don't use n_jobs=-1 in models when using Ray (nested parallelism)
- ❌ Don't skip parameter validation (incompatible solver/penalty combos)
- ❌ Don't ignore stratification failures (fall back to regular CV)
- ❌ Don't cache failed tuning results
- ❌ Don't run excessive trials without timeout
- ❌ Don't modify default parameters in model configs
- ❌ Don't break existing CLI interface

## Additional Notes

### Memory Considerations
- Ray Tune can be memory intensive with large datasets
- Consider implementing data subsampling for tuning phase
- Monitor memory usage and implement cleanup between trials

### Scalability
- Start with Grid/Random search for initial exploration  
- Use Optuna for refined optimization
- Ray Tune for large-scale distributed tuning
- Implement checkpointing for long-running tuning jobs

### Monitoring and Debugging
- Log all trial results, not just the best
- Save convergence plots for analysis
- Include parameter importance analysis
- Track which parameters hit bounds (may need expansion)

## Confidence Score: 8.5/10

This PRP provides comprehensive context for implementing hyperparameter tuning across all ML models. The score reflects:
- Strong foundation with existing XGBoost tuning code
- Clear integration points with Ray infrastructure
- Well-defined search spaces based on research
- Comprehensive validation strategy
- Some complexity in multi-model support and distributed coordination

The implementation should succeed in one pass with minor iterations for optimization.