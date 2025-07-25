# Next Steps: IM-053 - Implement Hyperparameter Tuning for All ML Models

## FEATURE:

Implement comprehensive hyperparameter tuning for all ML baseline models (XGBoost, Random Forest, Logistic Regression) to improve model performance across different VA datasets and sites. The tuning should leverage the existing Ray infrastructure for distributed optimization and integrate seamlessly with the current model comparison framework.

## CURRENT STATE:

### Completed Components:
- ✅ XGBoost baseline model with default hyperparameters (IM-045)
- ✅ Random Forest baseline model with default hyperparameters (IM-046)
- ✅ Logistic Regression baseline model with default hyperparameters (IM-047)
- ✅ Ray-based distributed comparison framework (IM-051)
- ✅ Bootstrap confidence intervals for metrics (IM-052)

### Current Performance Baseline:
Based on VA34 comparison results:
- **XGBoost**: 81.5% in-domain CSMF accuracy, 43.8% out-domain
- **InSilicoVA**: 80.0% in-domain, 46.1% out-domain
- Models currently use default scikit-learn configurations

## IMPLEMENTATION PLAN:

### 1. Design Hyperparameter Search Spaces

**XGBoost Parameters:**
```python
{
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 500],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
    'reg_lambda': [1, 5, 10]     # L2 regularization
}
```

**Random Forest Parameters:**
```python
{
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True, False]
}
```

**Logistic Regression Parameters:**
```python
{
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['saga'],  # supports all penalties
    'l1_ratio': [0.15, 0.5, 0.85],  # for elasticnet
    'max_iter': [1000, 2000]
}
```

### 2. Integration Architecture

```
model_comparison/
├── hyperparameter_tuning/
│   ├── __init__.py
│   ├── search_spaces.py      # Parameter grids for each model
│   ├── tuning_strategies.py  # GridSearch, Bayesian, etc.
│   ├── ray_tuner.py         # Ray Tune integration
│   └── tuning_utils.py      # Helper functions
├── scripts/
│   └── run_distributed_comparison.py  # Update to include tuning
└── experiments/
    └── experiment_config.py   # Add tuning configuration
```

### 3. Implementation Steps

#### Step 3.1: Create Base Tuning Infrastructure
- [ ] Create `hyperparameter_tuning` module structure
- [ ] Define search spaces for each model type
- [ ] Implement tuning strategy interface (GridSearch, BayesianOpt)

#### Step 3.2: Ray Tune Integration
- [ ] Integrate Ray Tune with existing Ray infrastructure
- [ ] Implement distributed hyperparameter search
- [ ] Add checkpointing for long-running searches
- [ ] Include early stopping for poor configurations

#### Step 3.3: Update Model Classes
- [ ] Modify XGBoostVAModel to accept hyperparameter configs
- [ ] Update RandomForestVAModel for dynamic parameters
- [ ] Enhance LogisticRegressionVAModel with tuning support
- [ ] Ensure backward compatibility with default parameters

#### Step 3.4: Modify Experiment Configuration
- [ ] Update ExperimentConfig to include tuning specifications
- [ ] Add tuning_enabled flag and search parameters
- [ ] Configure cross-validation strategy for tuning
- [ ] Set computational budget constraints

#### Step 3.5: Update Distributed Comparison Script
- [ ] Add hyperparameter tuning phase before model training
- [ ] Implement tuning results caching
- [ ] Log best parameters for reproducibility
- [ ] Update progress tracking for tuning phase

### 4. Validation & Testing

#### Unit Tests:
- [ ] Test search space definitions
- [ ] Test tuning strategy implementations
- [ ] Test parameter validation
- [ ] Test Ray Tune integration

#### Integration Tests:
- [ ] Test end-to-end tuning workflow
- [ ] Test distributed tuning across multiple workers
- [ ] Test tuning with different dataset sizes
- [ ] Test reproducibility of results

#### Performance Tests:
- [ ] Verify 10-30% improvement target
- [ ] Compare tuned vs default configurations
- [ ] Analyze computational cost vs benefit
- [ ] Test scalability with Ray

### 5. Expected Outcomes

- **Performance Improvement**: 10-30% increase in CSMF accuracy
- **Better Generalization**: Improved out-domain performance through regularization
- **Reproducibility**: All tuned parameters logged and trackable
- **Efficiency**: Distributed tuning completes within reasonable time
- **Documentation**: Clear guide for adding new models/parameters

## TECHNICAL CONSIDERATIONS:

### Ray Tune Configuration:
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    metric="csmf_accuracy",
    mode="max",
    max_t=100,  # max iterations
    grace_period=10,
    reduction_factor=3
)
```

### Cross-Validation Strategy:
- Use stratified k-fold (k=5) for tuning
- Preserve site-based splits for final evaluation
- Balance computational cost with statistical validity

### Computational Budget:
- Set max trials based on search space size
- Use early stopping to eliminate poor performers
- Cache intermediate results for resilience

## DEPENDENCIES:

- Ray Tune (add to poetry dependencies)
- Optuna (optional, for Bayesian optimization)
- Existing model implementations
- Current Ray infrastructure

## SUCCESS CRITERIA:

1. All three ML models have automated hyperparameter tuning
2. Tuning integrates seamlessly with `run_distributed_comparison.py`
3. Demonstrated performance improvement (target: 10-30%)
4. Tuning completes within reasonable time (< 2 hours for full experiment)
5. Results are reproducible and well-documented
6. Unit test coverage > 95% for new code

## NEXT ACTIONS:

1. Create the hyperparameter_tuning module structure
2. Define search spaces based on domain knowledge and literature
3. Implement Ray Tune integration with existing infrastructure
4. Update model classes to support dynamic parameters
5. Run initial tuning experiments on a subset of data
6. Analyze results and refine search spaces
7. Full implementation with all models and sites
8. Document findings and best practices

## NOTES:

- Consider using Bayesian optimization for more efficient search
- Monitor for overfitting, especially with increased model complexity
- Ensure tuned models still maintain good out-domain performance
- Document computational requirements for future users
- Consider creating pre-tuned model configs for common scenarios