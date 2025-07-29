# NEXT STEPS - Task IM-046: Random Forest Baseline Model

## Overview
Implement a Random Forest baseline model for VA cause-of-death prediction, following the established patterns from XGBoost and InSilicoVA implementations. This model will serve as another classical ML baseline for comparison in the model evaluation framework.

## Context & Requirements

### Task Details
- **Task ID**: IM-046
- **Priority**: High
- **Dependencies**: 
  - VADataProcessor (✓ completed)
  - Numeric encoding support (✓ completed)
  - XGBoost model pattern (✓ reference implementation)

### Business Value
- Provides another strong baseline model for VA analysis
- Random Forest offers inherent feature importance analysis
- Robust to overfitting and handles non-linear relationships well
- Can provide ensemble predictions with uncertainty estimates
- Complements XGBoost by offering a different tree-based approach

## Technical Approach

### 1. Architecture Pattern
Follow the established model architecture:
```
baseline/models/
├── random_forest_config.py   # Pydantic configuration
├── random_forest_model.py    # Main model implementation
└── __init__.py              # Export RandomForestModel
```

### 2. Configuration Design
Create `RandomForestConfig` with parameters:
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)
- `min_samples_split`: Minimum samples to split node (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)
- `max_features`: Features to consider for splits (default: "sqrt")
- `bootstrap`: Whether to use bootstrap samples (default: True)
- `class_weight`: Handle imbalanced classes (default: "balanced")
- `n_jobs`: Parallel processing (default: -1)
- `random_state`: For reproducibility (default: 42)

### 3. Model Implementation
Key features to implement:
- **sklearn Interface**: `fit()`, `predict()`, `predict_proba()`
- **Feature Importance**: Multiple importance metrics
  - Mean Decrease in Impurity (MDI)
  - Permutation importance
  - Feature importance visualization
- **CSMF Accuracy**: Reuse calculation from XGBoost
- **Cross-validation**: Stratified K-fold with metrics
- **Class Imbalance**: Built-in handling via class_weight

### 4. Integration Points
- Compatible with `VADataProcessor` output
- Works with numeric encoding from data pipeline
- Integrates with model comparison framework
- Supports the same evaluation metrics as other models

## Implementation Steps

### Step 1: Configuration Class
Create `random_forest_config.py` with:
- Pydantic BaseModel with validation
- Parameter bounds and type checking
- Sensible defaults for VA data

### Step 2: Model Implementation
Create `random_forest_model.py` with:
- RandomForestModel class extending BaseEstimator, ClassifierMixin
- Methods matching XGBoostModel interface
- Feature importance analysis methods
- CSMF accuracy calculation

### Step 3: Test Suite
Create `test_random_forest_model.py` with:
- Configuration validation tests
- Model fitting and prediction tests
- Feature importance tests
- Cross-validation tests
- Edge cases and error handling

### Step 4: Documentation
- Add usage examples to README
- Document feature importance interpretation
- Comparison with other baseline models

## Validation Criteria

### Code Quality
- [ ] Passes ruff linting
- [ ] Passes mypy type checking
- [ ] Follows existing code patterns

### Functionality
- [ ] Model trains successfully on VA data
- [ ] Predictions match expected format
- [ ] Feature importance works correctly
- [ ] CSMF accuracy calculation is accurate
- [ ] Cross-validation produces stable results

### Testing
- [ ] Unit test coverage >90%
- [ ] All tests pass
- [ ] Integration with data pipeline verified
- [ ] Performance benchmarks documented

### Integration
- [ ] Works with model comparison framework
- [ ] Compatible with existing evaluation scripts
- [ ] Can be used in parallel experiments

## Expected Outcomes

1. **Model Performance**
   - CSMF accuracy: ~0.70-0.85 (similar to XGBoost)
   - Training time: Moderate (slower than XGBoost, faster than InSilicoVA)
   - Prediction time: Fast
   - Memory usage: Moderate

2. **Feature Insights**
   - Top important features for COD prediction
   - Feature interaction patterns
   - Stability of importance across folds

3. **Comparison Value**
   - Different ensemble approach than boosting
   - More interpretable than XGBoost
   - Better uncertainty estimates
   - Robust baseline for comparison

## Risk Mitigation

1. **Memory Issues**: 
   - Use `max_samples` parameter if needed
   - Implement batch prediction for large datasets

2. **Class Imbalance**:
   - Use `class_weight="balanced"`
   - Monitor per-class performance

3. **Overfitting**:
   - Use appropriate max_depth
   - Implement early stopping in CV

## Success Metrics

- Implementation completed in single sprint
- All validation tests pass
- Successfully integrated into comparison pipeline
- Feature importance provides actionable insights
- Performance comparable to other baselines

## Next Actions

1. Generate PRP with detailed implementation blueprint
2. Create GitHub issue and feature branch
3. Implement following the PRP
4. Validate and test thoroughly
5. Document and integrate
6. Create PR and complete task