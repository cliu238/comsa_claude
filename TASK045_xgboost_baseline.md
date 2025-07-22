# TASK045: XGBoost Baseline Model Implementation

## Task Overview
**Task ID**: IM-045  
**Status**: 📋 Planned  
**Priority**: High  
**Dependencies**: VADataProcessor, numeric encoding  
**Target Date**: Q2 2025  

## Description
Implement XGBoost baseline model for VA cause-of-death prediction as part of the classical ML models suite. This model will serve as a high-performance baseline for comparison with both traditional VA algorithms (InSilicoVA, InterVA) and other ML approaches.

## Implementation Requirements

### Core Functionality
1. **Model Architecture**
   - Multi-class classification for cause-of-death prediction
   - Support for handling imbalanced classes inherent in VA data
   - Configurable hyperparameters with sensible defaults
   - GPU support optional but recommended for large datasets

2. **Integration Points**
   - Seamless integration with VADataProcessor for data loading
   - Use numeric encoding output from existing pipeline
   - Compatible with model comparison framework architecture
   - Follow sklearn-like interface pattern (fit, predict, predict_proba)

3. **Hyperparameter Tuning**
   - Implement grid search or Bayesian optimization
   - Key parameters to tune:
     - max_depth, learning_rate, n_estimators
     - subsample, colsample_bytree
     - scale_pos_weight for class imbalance
   - Cross-validation strategy respecting site stratification

### Performance Metrics
- CSMF (Cause-Specific Mortality Fraction) accuracy
- COD (Cause of Death) accuracy at individual level
- Classification metrics: precision, recall, F1-score per class
- Feature importance analysis for interpretability
- Training time and inference speed benchmarks

### Expected Outcomes
- Achieve CSMF accuracy comparable to or better than InSilicoVA (~0.79)
- Provide interpretable feature importance rankings
- Handle missing data appropriately (XGBoost native support)
- Scale efficiently to full PHMRC dataset

## Technical Specifications

### File Structure
```
baseline/models/
├── xgboost_model.py      # Main XGBoost implementation
├── hyperparameter_tuning.py  # Tuning utilities
└── tests/
    └── test_xgboost_model.py  # Comprehensive unit tests
```

### Key Methods
- `fit(X, y, sample_weight=None)`: Train model with optional sample weights
- `predict(X)`: Predict cause of death
- `predict_proba(X)`: Return probability distributions
- `get_feature_importance()`: Extract and visualize feature importance
- `cross_validate()`: Perform k-fold CV with stratification

### Dependencies
- xgboost>=1.7.0
- scikit-learn>=1.0.0
- numpy, pandas (existing)
- optuna (for hyperparameter tuning)

## Validation Criteria
1. Unit tests achieve >95% coverage
2. Model trains successfully on PHMRC data
3. CSMF accuracy metric properly calculated
4. Hyperparameter tuning improves baseline performance
5. Feature importance output is interpretable
6. Handles edge cases (single class, missing features)

## Related Tasks
- Depends on: [IM-003] VADataProcessor, [IM-007] Numeric encoding
- Enables: [IM-036] Multiple model training pipeline
- Related to: [IM-046] Random Forest, [IM-047] Logistic Regression

## Notes
- XGBoost chosen for its excellent performance on tabular data
- Native missing value handling aligns well with VA data characteristics
- Tree-based model provides interpretability through feature importance
- Consider early stopping to prevent overfitting on smaller datasets