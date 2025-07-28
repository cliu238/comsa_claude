# PRP: Comprehensive Hyperparameter Tuning for ML Models (IM-053)

## Problem Statement
Implement comprehensive hyperparameter tuning for all ML baseline models (XGBoost, Random Forest, Logistic Regression) to improve model performance across different VA datasets and sites. The tuning should leverage the existing Ray infrastructure for distributed optimization and integrate seamlessly with the current model comparison framework.

Current baseline performance shows XGBoost at 81.5% in-domain CSMF accuracy using default parameters, with potential for 10-30% improvement through hyperparameter optimization.

## Context and Requirements

### Existing Infrastructure
- Ray-based distributed comparison framework already implemented (IM-051)
- Three ML models with sklearn-compatible interfaces: XGBoostModel, RandomForestModel, LogisticRegressionModel
- Models implement `get_params()` and `set_params()` methods for parameter management
- ExperimentConfig class for experiment configuration
- Ray tasks for parallel model training (`train_and_evaluate_model`)

### Key Files to Reference
- `/Users/ericliu/projects5/context-engineering-intro/model_comparison/orchestration/ray_tasks.py` - Ray remote tasks
- `/Users/ericliu/projects5/context-engineering-intro/model_comparison/experiments/experiment_config.py` - Experiment configuration
- `/Users/ericliu/projects5/context-engineering-intro/baseline/models/xgboost_model.py` - XGBoost implementation pattern
- `/Users/ericliu/projects5/context-engineering-intro/baseline/models/xgboost_config.py` - Configuration pattern
- `/Users/ericliu/projects5/context-engineering-intro/model_comparison/scripts/run_distributed_comparison.py` - Main script

### External Documentation
- Ray Tune Documentation: https://docs.ray.io/en/latest/tune/index.html
- Ray Tune XGBoost Example: https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html
- Ray Tune Key Concepts: https://docs.ray.io/en/latest/tune/key-concepts.html

## Implementation Blueprint

### Step 1: Create Hyperparameter Tuning Module Structure

```
model_comparison/
├── hyperparameter_tuning/
│   ├── __init__.py
│   ├── search_spaces.py      # Parameter grids for each model
│   ├── tuning_strategies.py  # GridSearch, Bayesian, etc.
│   ├── ray_tuner.py         # Ray Tune integration
│   └── tuning_utils.py      # Helper functions
```

### Step 2: Define Search Spaces (search_spaces.py)

```python
from ray import tune
from typing import Dict, Any

def get_xgboost_search_space() -> Dict[str, Any]:
    """Get XGBoost hyperparameter search space."""
    return {
        'config__max_depth': tune.choice([3, 5, 7, 10]),
        'config__learning_rate': tune.loguniform(0.01, 0.3),
        'config__n_estimators': tune.choice([100, 200, 500]),
        'config__subsample': tune.uniform(0.7, 1.0),
        'config__colsample_bytree': tune.uniform(0.7, 1.0),
        'config__reg_alpha': tune.loguniform(1e-4, 1.0),
        'config__reg_lambda': tune.loguniform(1.0, 10.0)
    }

def get_random_forest_search_space() -> Dict[str, Any]:
    """Get Random Forest hyperparameter search space."""
    return {
        'config__n_estimators': tune.choice([100, 200, 500]),
        'config__max_depth': tune.choice([None, 10, 20, 30]),
        'config__min_samples_split': tune.choice([2, 5, 10]),
        'config__min_samples_leaf': tune.choice([1, 2, 4]),
        'config__max_features': tune.choice(['sqrt', 'log2', 0.5]),
        'config__bootstrap': tune.choice([True, False])
    }

def get_logistic_regression_search_space() -> Dict[str, Any]:
    """Get Logistic Regression hyperparameter search space."""
    return {
        'config__C': tune.loguniform(0.001, 100.0),
        'config__penalty': tune.choice(['l1', 'l2', 'elasticnet']),
        'config__solver': 'saga',  # Fixed for all penalty types
        'config__l1_ratio': tune.uniform(0.15, 0.85),  # Only for elasticnet
        'config__max_iter': tune.choice([1000, 2000])
    }
```

### Step 3: Ray Tune Integration (ray_tuner.py)

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig, CheckpointConfig
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score
import numpy as np

class RayTuner:
    """Ray Tune integration for hyperparameter optimization."""
    
    def __init__(self, 
                 n_trials: int = 100,
                 n_cpus_per_trial: float = 1.0,
                 max_concurrent_trials: Optional[int] = None,
                 search_algorithm: str = "bayesian",
                 metric: str = "csmf_accuracy",
                 mode: str = "max"):
        """Initialize Ray Tuner.
        
        Args:
            n_trials: Number of hyperparameter combinations to try
            n_cpus_per_trial: CPUs allocated per trial
            max_concurrent_trials: Max trials running in parallel
            search_algorithm: "grid", "random", or "bayesian"
            metric: Metric to optimize
            mode: "min" or "max"
        """
        self.n_trials = n_trials
        self.n_cpus_per_trial = n_cpus_per_trial
        self.max_concurrent_trials = max_concurrent_trials
        self.search_algorithm = search_algorithm
        self.metric = metric
        self.mode = mode
        
        # Initialize scheduler for early stopping
        self.scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=100,
            grace_period=10,
            reduction_factor=3
        )
        
    def tune_model(self,
                   model_class: Any,
                   search_space: Dict[str, Any],
                   train_data: Tuple[pd.DataFrame, pd.Series],
                   val_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   cv_folds: int = 5) -> Dict[str, Any]:
        """Run hyperparameter tuning for a model.
        
        Args:
            model_class: Model class to tune
            search_space: Hyperparameter search space
            train_data: Training data (X, y)
            val_data: Optional validation data
            cv_folds: Number of CV folds if val_data not provided
            
        Returns:
            Best hyperparameters and performance metrics
        """
        X_train, y_train = train_data
        
        def objective(config):
            """Objective function for Ray Tune."""
            # Create model with config
            model = model_class()
            model.set_params(**config)
            
            if val_data is not None:
                # Use validation set
                X_val, y_val = val_data
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate CSMF accuracy
                csmf_acc = model.calculate_csmf_accuracy(y_val, y_pred)
                cod_acc = (y_val == y_pred).mean()
            else:
                # Use cross-validation
                scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv_folds, 
                    scoring='accuracy'  # Will use custom scorer
                )
                csmf_acc = scores.mean()
                cod_acc = scores.mean()
            
            return {
                "csmf_accuracy": csmf_acc,
                "cod_accuracy": cod_acc
            }
        
        # Configure search algorithm
        if self.search_algorithm == "bayesian":
            search_alg = OptunaSearch()
        else:
            search_alg = None
        
        # Run tuning
        tuner = tune.Tuner(
            objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=self.n_trials,
                scheduler=self.scheduler,
                search_alg=search_alg,
                max_concurrent_trials=self.max_concurrent_trials
            ),
            run_config=RunConfig(
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=10,
                    checkpoint_at_end=True
                )
            )
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric=self.metric, mode=self.mode)
        
        return {
            "best_params": best_result.config,
            "best_score": best_result.metrics[self.metric],
            "all_results": results.get_dataframe()
        }
```

### Step 4: Update ExperimentConfig

Add tuning configuration to `experiment_config.py`:

```python
class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""
    enabled: bool = Field(default=False, description="Enable hyperparameter tuning")
    n_trials: int = Field(default=100, description="Number of tuning trials")
    search_algorithm: str = Field(default="bayesian", description="Search algorithm")
    max_concurrent_trials: Optional[int] = Field(default=None, description="Max parallel trials")
    cv_folds: int = Field(default=5, description="Cross-validation folds for tuning")
    tuning_metric: str = Field(default="csmf_accuracy", description="Metric to optimize")
    
class ExperimentConfig(BaseModel):
    # ... existing fields ...
    
    # Tuning configuration
    tuning: TuningConfig = Field(default_factory=TuningConfig)
```

### Step 5: Integrate Tuning into Ray Tasks

Update `train_and_evaluate_model` in `ray_tasks.py` to support tuning:

```python
@ray.remote
def tune_and_train_model(
    model_name: str,
    train_data: Tuple[pd.DataFrame, pd.Series],
    test_data: Tuple[pd.DataFrame, pd.Series],
    experiment_metadata: Dict,
    tuning_config: Optional[Dict] = None,
    n_bootstrap: int = 100,
) -> ExperimentResult:
    """Tune hyperparameters and train model."""
    # ... existing imports ...
    
    if tuning_config and tuning_config.get("enabled", False):
        # Import tuning components
        from model_comparison.hyperparameter_tuning.ray_tuner import RayTuner
        from model_comparison.hyperparameter_tuning.search_spaces import (
            get_xgboost_search_space,
            get_random_forest_search_space,
            get_logistic_regression_search_space
        )
        
        # Get model class and search space
        model_class, search_space = {
            "xgboost": (XGBoostModel, get_xgboost_search_space()),
            "random_forest": (RandomForestModel, get_random_forest_search_space()),
            "logistic_regression": (LogisticRegressionModel, get_logistic_regression_search_space())
        }[model_name]
        
        # Run tuning
        tuner = RayTuner(
            n_trials=tuning_config.get("n_trials", 100),
            search_algorithm=tuning_config.get("search_algorithm", "bayesian"),
            metric=tuning_config.get("tuning_metric", "csmf_accuracy")
        )
        
        tuning_results = tuner.tune_model(
            model_class=model_class,
            search_space=search_space,
            train_data=train_data,
            cv_folds=tuning_config.get("cv_folds", 5)
        )
        
        # Create model with best params
        model = model_class()
        model.set_params(**tuning_results["best_params"])
        
        # Store tuning results in metadata
        experiment_metadata["tuning_results"] = tuning_results
    else:
        # Use default parameters
        # ... existing model initialization code ...
```

### Step 6: Update CLI Script

Add tuning arguments to `run_distributed_comparison.py`:

```python
# Add to argument parser
parser.add_argument(
    "--enable-tuning",
    action="store_true",
    help="Enable hyperparameter tuning"
)
parser.add_argument(
    "--tuning-trials",
    type=int,
    default=100,
    help="Number of tuning trials per model"
)
parser.add_argument(
    "--tuning-algorithm",
    choices=["grid", "random", "bayesian"],
    default="bayesian",
    help="Hyperparameter search algorithm"
)

# Update config creation
tuning_config = TuningConfig(
    enabled=args.enable_tuning,
    n_trials=args.tuning_trials,
    search_algorithm=args.tuning_algorithm
)

experiment_config = ExperimentConfig(
    # ... existing config ...
    tuning=tuning_config
)
```

## Critical Implementation Details

### 1. Parameter Naming Convention
- Use `config__` prefix for nested config parameters (e.g., `config__max_depth`)
- This allows `set_params()` to properly update the config object

### 2. Conditional Parameters
- Handle conditional parameters like `l1_ratio` for elasticnet:
```python
def filter_params(params: Dict, model_name: str) -> Dict:
    """Filter parameters based on model requirements."""
    if model_name == "logistic_regression":
        if params.get("config__penalty") != "elasticnet":
            params.pop("config__l1_ratio", None)
    return params
```

### 3. Resource Management
- Set `n_cpus_per_trial` based on model complexity
- Use `max_concurrent_trials` to prevent memory overflow
- Implement checkpointing for long-running searches

### 4. Metric Calculation
- Ensure CSMF accuracy calculation is consistent across tuning and evaluation
- Consider using a custom sklearn scorer for cross-validation

## Validation Gates

```bash
# 1. Code quality checks
poetry run ruff check model_comparison/hyperparameter_tuning/ --fix
poetry run mypy model_comparison/hyperparameter_tuning/

# 2. Unit tests
poetry run pytest model_comparison/tests/test_hyperparameter_tuning.py -v

# 3. Integration test with small dataset
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path data/test_subset.csv \
    --sites site_1 \
    --models xgboost \
    --enable-tuning \
    --tuning-trials 10 \
    --n-workers 2

# 4. Performance validation
# Verify tuned models achieve better performance than defaults
poetry run python model_comparison/scripts/compare_tuned_vs_default.py
```

## Testing Strategy

### Unit Tests (`test_hyperparameter_tuning.py`)
```python
import pytest
from model_comparison.hyperparameter_tuning.search_spaces import (
    get_xgboost_search_space,
    get_random_forest_search_space
)
from model_comparison.hyperparameter_tuning.ray_tuner import RayTuner

class TestSearchSpaces:
    def test_xgboost_search_space(self):
        space = get_xgboost_search_space()
        assert "config__max_depth" in space
        assert "config__learning_rate" in space
        
    def test_parameter_prefix(self):
        space = get_xgboost_search_space()
        for key in space.keys():
            assert key.startswith("config__")

class TestRayTuner:
    @pytest.fixture
    def sample_data(self):
        # Create small test dataset
        pass
        
    def test_tuner_initialization(self):
        tuner = RayTuner(n_trials=10)
        assert tuner.n_trials == 10
        assert tuner.metric == "csmf_accuracy"
        
    def test_tune_model_with_validation(self, sample_data):
        # Test tuning with validation set
        pass
```

### Integration Tests
- Test end-to-end tuning workflow
- Verify distributed execution across multiple workers
- Test checkpoint recovery
- Validate performance improvements

## Potential Issues and Solutions

### Issue 1: Memory Usage with Large Search Spaces
**Solution**: Use ASHA scheduler for early stopping and limit concurrent trials

### Issue 2: Tuning Time Exceeds 5-minute Limit
**Solution**: Create standalone tuning script for manual execution with progress saving

### Issue 3: Different Parameter Constraints
**Solution**: Implement parameter validation in search space definitions

### Issue 4: Reproducibility
**Solution**: Set random seeds in Ray Tune and save best parameters to JSON

## Expected Outcomes

1. **Performance**: 10-30% improvement in CSMF accuracy
2. **Scalability**: Distributed tuning completes in < 2 hours
3. **Integration**: Seamless integration with existing pipeline
4. **Documentation**: Clear logs of best parameters and tuning history

## Implementation Order

1. Create hyperparameter tuning module structure
2. Implement search spaces for all models
3. Build Ray Tuner class with basic functionality
4. Add tuning configuration to ExperimentConfig
5. Integrate tuning into ray_tasks.py
6. Update CLI script with tuning options
7. Write comprehensive tests
8. Run performance validation
9. Document findings and best practices

## Success Metrics

- [ ] All three ML models have automated hyperparameter tuning
- [ ] Tuning integrates with `run_distributed_comparison.py`
- [ ] Demonstrated 10-30% performance improvement
- [ ] Tuning completes within 2 hours for full experiment
- [ ] Results are reproducible with saved parameters
- [ ] 95%+ test coverage for new code

## Confidence Score: 8/10

The implementation plan is comprehensive with clear integration points into existing infrastructure. The score is not 10/10 due to:
- Potential complexity in handling conditional parameters across different models
- Need to carefully manage Ray resources to avoid conflicts with existing Ray usage
- Some uncertainty around exact performance improvements achievable

The plan leverages existing patterns and infrastructure while adding new capabilities in a modular way, making it highly likely to succeed in one-pass implementation.