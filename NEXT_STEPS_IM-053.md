# Next Steps for Task IM-053: Implement Hyperparameter Tuning for All ML Models

## Task Overview

**Task ID**: IM-053  
**Priority**: High  
**Dependencies**: IM-045 (XGBoost ✅), IM-046 (Random Forest ✅), IM-047 (Logistic Regression ✅)  
**Target Date**: Q1 2025  
**Expected Outcome**: 10-30% performance improvement through systematic hyperparameter optimization

## Current State Analysis

### 1. Existing Model Implementations
All three ML models are implemented with sklearn-compatible interfaces and default configurations:
- **XGBoost**: Uses default parameters (n_estimators=100, max_depth=6, learning_rate=0.3)
- **Random Forest**: Uses default parameters (n_estimators=100, max_depth=None, min_samples_split=2)
- **Logistic Regression**: Uses default parameters (C=1.0, penalty='l2', solver='saga')

### 2. Infrastructure Available
- **Ray**: Already integrated for distributed computing in model_comparison framework
- **Prefect**: Workflow orchestration in place
- **ExperimentConfig**: Configuration system ready for extension

### 3. Integration Points
- Must integrate with `model_comparison/scripts/run_distributed_comparison.py`
- Leverage existing Ray infrastructure for distributed tuning
- Update ExperimentConfig to include hyperparameter search spaces

## Implementation Strategy

### Phase 1: Design Hyperparameter Search Configuration

1. **Create Hyperparameter Search Space Schema**
   - Define search spaces for each model using Pydantic
   - Support both grid search and Bayesian optimization approaches
   - Include model-specific parameter ranges based on VA data characteristics

2. **Extend ExperimentConfig**
   ```python
   class HyperparameterSearchConfig(BaseModel):
       search_method: Literal["grid", "random", "bayesian", "optuna"]
       n_trials: int = Field(default=50, description="Number of trials for optimization")
       cv_folds: int = Field(default=5, description="Cross-validation folds")
       scoring_metric: str = Field(default="csmf_accuracy", description="Optimization metric")
       search_spaces: Dict[str, ModelSearchSpace] = Field(default_factory=dict)
   ```

### Phase 2: Implement Model-Specific Search Spaces

1. **XGBoost Hyperparameter Space**
   - `max_depth`: [3, 6, 9, 12]
   - `learning_rate`: [0.01, 0.05, 0.1, 0.3]
   - `n_estimators`: [100, 200, 300, 500]
   - `subsample`: [0.6, 0.8, 1.0]
   - `colsample_bytree`: [0.6, 0.8, 1.0]
   - `reg_alpha`: [0, 0.01, 0.1, 1.0]
   - `reg_lambda`: [1, 2, 5, 10]

2. **Random Forest Hyperparameter Space**
   - `n_estimators`: [100, 200, 300, 500]
   - `max_depth`: [None, 10, 20, 30]
   - `min_samples_split`: [2, 5, 10]
   - `min_samples_leaf`: [1, 2, 4]
   - `max_features`: ["sqrt", "log2", 0.5, 0.8]

3. **Logistic Regression Hyperparameter Space**
   - `C`: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
   - `penalty`: ["l1", "l2", "elasticnet"]
   - `l1_ratio`: [0.1, 0.5, 0.9] (when penalty="elasticnet")
   - `solver`: ["saga", "liblinear"] (based on penalty compatibility)

### Phase 3: Create Hyperparameter Tuning Module

1. **Create `model_comparison/hyperparameter_tuning/` Package**
   ```
   model_comparison/hyperparameter_tuning/
   ├── __init__.py
   ├── search_spaces.py      # Model-specific search space definitions
   ├── tuner.py             # Main tuning interface
   ├── ray_tuner.py         # Ray-based distributed tuning
   ├── optuna_tuner.py      # Optuna Bayesian optimization
   └── utils.py             # Helper functions
   ```

2. **Implement Base Tuner Interface**
   - Abstract base class for different tuning strategies
   - Support for cross-validation with CSMF accuracy metric
   - Integration with existing model interfaces

### Phase 4: Integrate with Distributed Comparison Pipeline

1. **Modify `ray_tasks.py`**
   - Add hyperparameter tuning phase before model training
   - Cache tuned parameters for reproducibility
   - Log best parameters and scores

2. **Update `run_distributed_comparison.py`**
   - Add CLI flags for hyperparameter tuning:
     - `--tune-hyperparameters`: Enable tuning
     - `--tuning-method`: Choose optimization method
     - `--tuning-trials`: Number of trials
     - `--tuning-timeout`: Maximum time for tuning

3. **Extend ExperimentResult**
   - Add fields for best hyperparameters
   - Include tuning history and convergence plots
   - Store cross-validation scores

### Phase 5: Implement Tuning Strategies

1. **Grid Search Implementation**
   - Exhaustive search over parameter grid
   - Parallel evaluation using Ray
   - Early stopping for poor configurations

2. **Bayesian Optimization with Optuna**
   - Tree-structured Parzen Estimator (TPE) algorithm
   - Pruning unpromising trials
   - Multi-objective optimization (CSMF + COD accuracy)

3. **Ray Tune Integration**
   - Leverage Ray Tune's advanced schedulers (ASHA, PBT)
   - Distributed hyperparameter optimization
   - Resource-aware scheduling

### Phase 6: Validation and Testing

1. **Unit Tests**
   - Test search space generation
   - Validate parameter compatibility
   - Mock tuning runs with small datasets

2. **Integration Tests**
   - End-to-end tuning pipeline
   - Verify improved model performance
   - Check reproducibility with saved parameters

3. **Performance Benchmarks**
   - Compare tuned vs. default parameters
   - Measure tuning overhead
   - Validate 10-30% improvement target

## Key Deliverables

1. **Hyperparameter Tuning Module**
   - Flexible, extensible architecture
   - Support for multiple optimization methods
   - Integration with existing Ray infrastructure

2. **Updated Model Comparison Pipeline**
   - Seamless hyperparameter tuning integration
   - Minimal changes to existing workflow
   - Backward compatibility with non-tuned runs

3. **Documentation and Examples**
   - Tuning best practices for VA data
   - Example configurations for each model
   - Performance improvement case studies

## Success Criteria

1. **Performance Metrics**
   - Achieve 10-30% improvement in CSMF accuracy
   - Maintain or improve COD accuracy
   - Reduce overfitting on small training sets

2. **Technical Requirements**
   - Distributed tuning completes within reasonable time
   - Tuned parameters are reproducible
   - Integration doesn't break existing functionality

3. **Usability**
   - Simple CLI interface for enabling tuning
   - Clear logging of tuning progress
   - Easy interpretation of results

## Risk Mitigation

1. **Computational Cost**
   - Implement intelligent search strategies (Bayesian optimization)
   - Use early stopping for unpromising trials
   - Provide time/resource budgets

2. **Overfitting Risk**
   - Use proper cross-validation
   - Monitor validation vs. training performance
   - Include regularization in search spaces

3. **Integration Complexity**
   - Maintain backward compatibility
   - Implement feature flags for gradual rollout
   - Comprehensive testing before deployment

## Next Immediate Actions

1. **Create hyperparameter search space definitions** for all three models
2. **Design the tuning module architecture** with clear interfaces
3. **Implement Grid Search** as the first tuning method
4. **Add CLI integration** to run_distributed_comparison.py
5. **Test with small subset** of VA34 data to validate approach

## Technical Considerations

### Ray Integration Strategy
```python
# Leverage existing Ray actor pool
@ray.remote
class HyperparameterTuner:
    def tune_model(self, model_type, X_train, y_train, search_space):
        # Distributed hyperparameter search
        pass
```

### Caching and Reproducibility
```python
# Store tuned parameters
class TunedModelCache:
    def save_best_params(self, experiment_id, model_type, params):
        # Save to disk for reproducibility
        pass
```

### Progress Monitoring
```python
# Real-time tuning progress
class TuningMonitor:
    def log_trial(self, trial_id, params, score):
        # Track convergence and best scores
        pass
```

This implementation will significantly enhance the VA model comparison framework by finding optimal configurations for each model, leading to more accurate cause-of-death predictions and better cross-site generalization.