# Next Steps: IM-048 - Implement CategoricalNB Baseline Model

## FEATURE:

Implement a CategoricalNB (Categorical Naive Bayes) baseline model for VA cause-of-death prediction as the final ML baseline model. This model will complete the ML baseline suite alongside XGBoost, Random Forest, and Logistic Regression, providing a probabilistic approach that can handle categorical features natively and perform well with missing data common in VA datasets.

## CURRENT STATE:

### Completed Components:
- ✅ XGBoost baseline model with hyperparameter tuning (IM-045, IM-053)
- ✅ Random Forest baseline model with hyperparameter tuning (IM-046, IM-053)  
- ✅ Logistic Regression baseline model with hyperparameter tuning (IM-047, IM-053)
- ✅ VADataProcessor with numeric encoding support
- ✅ Ray-based distributed comparison framework (IM-051)
- ✅ Bootstrap confidence intervals for metrics (IM-052)
- ✅ Comprehensive hyperparameter tuning infrastructure (IM-053)

### Current ML Model Suite:
Based on recent performance with hyperparameter tuning:
- **XGBoost**: ~81.5% in-domain CSMF accuracy (tuned: ~94.6%)
- **Random Forest**: Strong ensemble performance with feature importance
- **Logistic Regression**: Linear baseline with regularization options
- **CategoricalNB**: Missing - needed to complete ML baseline suite

## IMPLEMENTATION PLAN:

### 1. CategoricalNB Model Architecture

**Key Characteristics:**
- Handles categorical features natively (no need for extensive preprocessing)
- Robust to missing data (common in VA questionnaires)
- Probabilistic output suitable for CSMF accuracy calculation
- Fast training and inference
- Natural handling of class imbalance through priors

**Implementation Requirements:**
```python
class CategoricalNBVAModel:
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """
        VA-specific Categorical Naive Bayes model
        
        Args:
            alpha: Additive smoothing parameter
            fit_prior: Whether to learn class prior probabilities
            class_prior: Prior probabilities of classes
        """
```

### 2. Data Preprocessing Strategy

**Challenge**: CategoricalNB expects integer-encoded categorical features
**Solution**: Minimal preprocessing pipeline
```python
# Current VADataProcessor outputs:
# - Numeric encoding: continuous + binary features
# - Need: categorical integer encoding for CategoricalNB

def prepare_categorical_features(self, data):
    """Convert VA data to categorical integer format"""
    # Handle missing values: encode as separate category
    # Convert Y/N/./DK to 0/1/2/3 integer categories
    # Preserve symptom-level categorical structure
```

### 3. Integration with Existing Framework

**File Structure:**
```
baseline/models/
├── __init__.py
├── xgboost_model.py          # ✅ Completed
├── random_forest_model.py    # ✅ Completed  
├── logistic_regression_model.py  # ✅ Completed
└── categorical_nb_model.py   # 🚧 To implement
```

**Interface Consistency:**
- Implement sklearn-compatible interface (fit, predict, predict_proba)
- Support hyperparameter tuning via existing infrastructure
- Include feature importance (via log probabilities)
- CSMF accuracy evaluation compatibility

### 4. Implementation Steps

#### Step 4.1: Create CategoricalNB Model Class
- [ ] Implement `CategoricalNBVAModel` with sklearn interface
- [ ] Add categorical data preprocessing methods
- [ ] Handle missing value encoding strategies
- [ ] Include probability calibration options

#### Step 4.2: Add Hyperparameter Search Space
- [ ] Define search space in `model_comparison/hyperparameter_tuning/search_spaces.py`
- [ ] Key parameters: alpha (smoothing), class_prior handling
- [ ] Integration with Ray Tune infrastructure

#### Step 4.3: Data Pipeline Integration
- [ ] Extend VADataProcessor with categorical encoding option
- [ ] Ensure compatibility with existing numeric encoding
- [ ] Handle missing data appropriately (encode as separate category)

#### Step 4.4: Feature Importance Implementation
- [ ] Implement feature importance via log probability ratios
- [ ] Create visualization compatibility with existing models
- [ ] Document interpretation guidelines for Naive Bayes coefficients

#### Step 4.5: Testing and Validation
- [ ] Unit tests for model class (target: >95% coverage)
- [ ] Integration tests with VA34 dataset
- [ ] Performance benchmarking against other models
- [ ] Cross-validation and hyperparameter tuning validation

### 5. Expected Outcomes

**Performance Expectations:**
- **Baseline Performance**: 70-80% in-domain CSMF accuracy (before tuning)
- **Tuned Performance**: 75-85% in-domain CSMF accuracy
- **Out-domain Performance**: Expected to generalize well due to probabilistic nature
- **Speed**: Very fast training and inference (similar to Logistic Regression)

**Advantages of CategoricalNB:**
- Native categorical feature handling
- Robust to missing data
- Interpretable probability estimates
- Good baseline for class-imbalanced datasets
- Minimal preprocessing requirements

## TECHNICAL CONSIDERATIONS:

### Data Preprocessing for CategoricalNB:
```python
# VA data transformation strategy
categorical_encoding = {
    'Y': 0,   # Yes
    'N': 1,   # No  
    '.': 2,   # Missing/Don't know
    'DK': 2,  # Don't know (same as missing)
}

# Handle edge cases in PHMRC data
def encode_categorical_features(self, data):
    # Convert all VA symptoms to integer categories
    # Preserve original meaning while making NB-compatible
    # Handle site-specific encoding variations
```

### Hyperparameter Search Space:
```python
categorical_nb_space = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Smoothing parameter
    'fit_prior': [True, False],           # Learn class priors
    'class_prior': [None, 'balanced']     # Prior strategy
}
```

### Missing Data Strategy:
- **Approach 1**: Encode missing as separate category (recommended)
- **Approach 2**: Imputation before encoding
- **Rationale**: VA data has meaningful missingness patterns

## DEPENDENCIES:

- scikit-learn CategoricalNB (already available)
- Existing VADataProcessor (needs categorical encoding extension)
- Hyperparameter tuning infrastructure (IM-053)
- Model comparison framework (IM-051)

## SUCCESS CRITERIA:

1. CategoricalNB model implements consistent sklearn-compatible interface
2. Model handles VA categorical data appropriately with missing values
3. Integration with hyperparameter tuning infrastructure works seamlessly
4. Performance competitive with other baseline models (70-85% CSMF accuracy)
5. Fast training and inference suitable for large-scale experiments
6. Unit test coverage > 95% for new code
7. Documentation includes interpretation guidelines for NB coefficients

## NEXT ACTIONS:

1. **Implement CategoricalNBVAModel class** in `baseline/models/categorical_nb_model.py`
2. **Extend VADataProcessor** with categorical encoding methods
3. **Add hyperparameter search space** to existing tuning infrastructure
4. **Create comprehensive unit tests** following existing model patterns
5. **Run initial experiments** on VA34 subset to validate implementation
6. **Integration testing** with distributed comparison framework
7. **Performance benchmarking** against existing ML baselines
8. **Documentation and examples** for usage and interpretation

## NOTES:

- CategoricalNB may perform better than expected due to natural fit with VA categorical data
- Consider ensemble methods combining CategoricalNB with other models
- Document any site-specific encoding challenges discovered
- Monitor for potential issues with rare categories or extreme class imbalance
- This completes the ML baseline model suite, enabling milestone MS-004 completion

## VALIDATION CHECKLIST:

- [ ] Model trains successfully on all VA34 sites
- [ ] Predictions produce valid probability distributions
- [ ] CSMF accuracy calculation works correctly
- [ ] Hyperparameter tuning improves performance
- [ ] Feature importance provides interpretable results
- [ ] Integration tests pass with existing framework
- [ ] Performance is competitive with other baselines
- [ ] Missing data handling is robust and documented