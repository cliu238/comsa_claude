# Next Steps: Task IM-047 - Implement Logistic Regression baseline model

## Task Overview
- **Task ID**: IM-047
- **Priority**: Medium
- **Dependencies**: VADataProcessor, numeric encoding
- **Description**: Implement Logistic Regression baseline model with Multinomial and L1/L2 regularization

## Current Context
We have successfully implemented:
1. XGBoost baseline model (IM-045) with sklearn interface and CSMF accuracy
2. Random Forest baseline model (IM-046) with feature importance analysis
3. InSilicoVA model (IM-013) with Docker integration
4. Model comparison framework (IM-035) for systematic evaluation

## Implementation Plan

### 1. Model Architecture
- Create `baseline/models/logistic_regression_model.py` with sklearn-compatible interface
- Create `baseline/models/logistic_regression_config.py` for configuration management
- Support multinomial logistic regression for multi-class VA prediction
- Implement L1, L2, and ElasticNet regularization options

### 2. Key Features
- **Regularization Options**:
  - L1 (Lasso) for feature selection
  - L2 (Ridge) for coefficient stability
  - ElasticNet (combined L1+L2)
- **Solver Options**:
  - 'lbfgs' for L2 only
  - 'liblinear' for small datasets
  - 'saga' for all penalties and large datasets
- **Multi-class Handling**:
  - Multinomial for true multi-class
  - OvR (One-vs-Rest) option for comparison
- **Class Balancing**:
  - Support class_weight='balanced' for imbalanced VA data
  - Custom class weights option

### 3. Model Interface (following XGBoost/RandomForest pattern)
```python
class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, config: Optional[LogisticRegressionConfig] = None)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel"
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray
    def get_feature_importance(self) -> pd.DataFrame  # Based on coefficients
    def calculate_csmf_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]
```

### 4. Configuration Schema
```python
class LogisticRegressionConfig(BaseModel):
    penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0  # Inverse regularization strength
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: Union[str, Dict[int, float], None] = "balanced"
    random_state: int = 42
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs"
    max_iter: int = 1000
    multi_class: Literal["auto", "ovr", "multinomial"] = "multinomial"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None  # For elasticnet only
```

### 5. Testing Requirements
- Unit tests with >95% coverage
- Test all regularization options
- Test with imbalanced data
- Test feature importance extraction
- Test CSMF accuracy calculation
- Test cross-validation functionality
- Integration with model comparison framework

### 6. Integration Points
- Add to `baseline/models/__init__.py`
- Update model comparison scripts to include logistic regression
- Ensure compatibility with numeric encoding from VADataProcessor
- Add to parallel execution framework (Ray/Prefect)

### 7. Documentation
- Comprehensive docstrings following Google style
- Example usage in module docstring
- Performance characteristics documentation
- Comparison with other baseline models

## Success Criteria
1. ✅ Sklearn-compatible interface matching XGBoost/RandomForest pattern
2. ✅ Support for L1, L2, and ElasticNet regularization
3. ✅ Feature importance based on coefficients
4. ✅ CSMF accuracy metric implementation
5. ✅ Class balancing for imbalanced VA data
6. ✅ >95% test coverage
7. ✅ Integration with model comparison framework
8. ✅ Performance on par with or better than basic ML models
9. ✅ Clear documentation and examples

## Expected Outcomes
- Fast, interpretable baseline model
- Feature selection capability through L1 regularization
- Coefficient-based feature importance for interpretability
- Suitable for datasets where linear relationships dominate
- Complement to tree-based models (XGBoost, Random Forest)

## Next Task After Completion
After IM-047, the next logical task would be:
- IM-048: Implement CategoricalNB baseline model
- IM-049: Implement InterVA model integration (high priority)