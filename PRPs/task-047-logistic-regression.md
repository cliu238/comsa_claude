name: "Logistic Regression Baseline Model Implementation PRP"
description: |

## Purpose
Implement a Logistic Regression baseline model for VA cause-of-death prediction following established patterns from XGBoost and Random Forest implementations, providing comprehensive context for one-pass implementation success with focus on interpretability and regularization options.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Implement a production-ready Logistic Regression baseline model with sklearn-compatible interface, coefficient-based feature importance, multiple regularization options (L1, L2, ElasticNet), and CSMF accuracy metrics for verbal autopsy cause-of-death prediction.

## Why
- **Business value**: Fast, interpretable linear model for understanding feature-outcome relationships
- **Feature selection**: L1 regularization enables automatic feature selection
- **Integration**: Complements tree-based models with linear assumptions
- **Interpretability**: Coefficients provide direct feature importance interpretation
- **Baseline comparison**: Essential fast baseline for model evaluation framework
- **Class balancing**: Built-in support for imbalanced VA data

## What
Create LogisticRegressionModel class with:
- Pydantic configuration for type-safe parameters (penalty, solver, regularization)
- sklearn estimator interface (fit, predict, predict_proba)
- Coefficient-based feature importance extraction
- CSMF accuracy calculation matching other models
- Cross-validation with stratification
- Support for L1, L2, ElasticNet regularization
- Multinomial and OvR multi-class strategies
- Class imbalance handling via class_weight='balanced'

### Success Criteria
- [ ] Model trains on VA data with numeric encoding
- [ ] All regularization options (L1, L2, ElasticNet, None) work correctly
- [ ] Feature importance from coefficients provides interpretable insights
- [ ] CSMF accuracy matches InSilicoVA/XGBoost/RandomForest implementation
- [ ] Integrates with model comparison framework
- [ ] Test coverage >95%
- [ ] All validation tests pass
- [ ] Performance on par with or better than basic ML models

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  why: Official API documentation for LogisticRegression parameters and methods
  
- url: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  why: Solver compatibility matrix, regularization details, multiclass strategies
  
- file: baseline/models/xgboost_model.py
  why: Pattern to follow for model implementation, sklearn interface, CSMF calculation
  
- file: baseline/models/random_forest_model.py
  why: Alternative pattern for sklearn interface, feature importance methods
  
- file: baseline/models/xgboost_config.py
  why: Pattern for Pydantic configuration with validation
  
- file: baseline/models/random_forest_config.py
  why: Additional configuration pattern example
  
- file: tests/baseline/test_xgboost_model.py
  why: Test patterns for model validation, cross-validation, metrics

- doc: https://scikit-learn.org/stable/modules/linear_model.html#solvers
  section: Solver Compatibility Table
  critical: liblinear supports L1 but not multinomial, lbfgs/newton-cg support multinomial but not L1, saga supports all
```

### Current Codebase tree
```bash
baseline/
├── models/
│   ├── __init__.py
│   ├── insilico_model.py
│   ├── model_config.py
│   ├── model_validator.py
│   ├── hyperparameter_tuning.py
│   ├── xgboost_config.py
│   ├── xgboost_model.py
│   ├── random_forest_config.py
│   └── random_forest_model.py
├── data/
│   ├── data_loader.py
│   ├── data_loader_preprocessor.py
│   └── data_splitter.py
└── utils/
    ├── __init__.py
    └── logging_config.py

tests/
└── baseline/
    ├── test_xgboost_model.py
    ├── test_random_forest_model.py
    └── test_data_splitter.py
```

### Desired Codebase tree with files to be added
```bash
baseline/
├── models/
│   ├── __init__.py                    # UPDATE: Export LogisticRegressionModel
│   ├── logistic_regression_config.py  # NEW: Pydantic configuration
│   ├── logistic_regression_model.py   # NEW: Main model implementation
│   ├── insilico_model.py
│   ├── xgboost_config.py
│   ├── xgboost_model.py
│   ├── random_forest_config.py
│   └── random_forest_model.py

tests/
└── baseline/
    ├── test_logistic_regression_model.py  # NEW: Comprehensive tests
    ├── test_xgboost_model.py
    └── test_random_forest_model.py
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Solver-penalty compatibility in scikit-learn LogisticRegression
# - 'liblinear': supports L1, L2 but NOT multinomial
# - 'lbfgs', 'newton-cg', 'newton-cholesky': support L2, None but NOT L1
# - 'saga': supports ALL penalties (L1, L2, elasticnet, None) AND multinomial
# - 'sag': supports L2, None but NOT L1

# CRITICAL: multi_class parameter is deprecated in sklearn 1.5+
# - Use OneVsRestClassifier explicitly for OvR if needed
# - Multinomial is default for 3+ classes

# CRITICAL: penalty='none' string is deprecated, use None instead

# GOTCHA: class_weight='balanced' uses sklearn.utils.class_weight.compute_sample_weight
# This is crucial for imbalanced VA data

# GOTCHA: Feature importance from coefficients requires special handling:
# - For binary: single coefficient vector
# - For multiclass: shape is (n_classes, n_features)
# - Need to aggregate across classes for overall importance

# Pattern: VADataProcessor outputs numeric encoding for ML models
# - All categorical features are converted to numeric
# - Label encoding is handled by the model itself
```

## Implementation Blueprint

### Data models and structure

Create the core configuration model with comprehensive validation:
```python
# logistic_regression_config.py structure:
- Use Pydantic BaseModel
- Support all scikit-learn LogisticRegression parameters
- Add custom validators for solver-penalty compatibility
- Include sensible defaults for VA data (C=1.0, solver='saga', penalty='l2')
- Validate l1_ratio only when penalty='elasticnet'
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Create logistic_regression_config.py
CREATE baseline/models/logistic_regression_config.py:
  - MIRROR pattern from: baseline/models/xgboost_config.py
  - ADD all LogisticRegression parameters with Pydantic fields
  - IMPLEMENT solver-penalty compatibility validator
  - IMPLEMENT l1_ratio validator for elasticnet
  - USE Field with descriptions and constraints
  - ENSURE model_config has validate_assignment=True

Task 2: Create logistic_regression_model.py
CREATE baseline/models/logistic_regression_model.py:
  - MIRROR pattern from: baseline/models/xgboost_model.py
  - INHERIT from BaseEstimator, ClassifierMixin
  - IMPLEMENT __init__ with config parameter
  - IMPLEMENT get_params/set_params for sklearn compatibility
  - IMPLEMENT fit method with sample_weight support
  - IMPLEMENT predict/predict_proba methods
  - IMPLEMENT get_feature_importance from coefficients
  - IMPLEMENT calculate_csmf_accuracy (copy exact formula)
  - IMPLEMENT cross_validate with StratifiedKFold
  - ADD proper logging throughout

Task 3: Update __init__.py
MODIFY baseline/models/__init__.py:
  - ADD import for LogisticRegressionModel and LogisticRegressionConfig
  - FOLLOW existing import pattern

Task 4: Create comprehensive tests
CREATE tests/baseline/test_logistic_regression_model.py:
  - MIRROR pattern from: tests/baseline/test_xgboost_model.py
  - TEST default configuration
  - TEST custom configuration with all penalties
  - TEST solver-penalty validation
  - TEST model fitting and prediction
  - TEST feature importance extraction
  - TEST CSMF accuracy calculation
  - TEST cross-validation
  - TEST class balancing
  - TEST edge cases (single class, missing data)
  - USE pytest fixtures for sample data
  - ENSURE >95% coverage

Task 5: Integration testing
MODIFY model_comparison/experiments/site_comparison.py (if needed):
  - ADD LogisticRegression to model comparison
  - ENSURE compatibility with existing framework
```

### Per task pseudocode as needed

```python
# Task 1: Configuration implementation
class LogisticRegressionConfig(BaseModel):
    # Core parameters with sklearn defaults
    penalty: Optional[Literal["l1", "l2", "elasticnet", None]] = Field(
        default="l2", description="Regularization penalty"
    )
    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength")
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = Field(
        default="saga", description="Optimization algorithm"
    )
    
    @field_validator("solver")
    def validate_solver_penalty_compatibility(cls, solver, info):
        # PATTERN: Access other fields via info.data
        penalty = info.data.get("penalty", "l2")
        
        # CRITICAL: Check compatibility matrix
        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            raise ValueError(f"Solver '{solver}' does not support L1 penalty")
        # ... more validation
        
        return solver

# Task 2: Model implementation key methods
class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        # PATTERN: Validate inputs like XGBoost
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # PATTERN: Store feature names for later
        self.feature_names_ = X.columns.tolist()
        
        # PATTERN: Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        
        # GOTCHA: Convert config to sklearn parameters
        sklearn_params = self._get_sklearn_params()
        
        # Create and fit model
        self.model_ = LogisticRegression(**sklearn_params)
        self.model_.fit(X.values, y_encoded)
        
        self._is_fitted = True
        return self
        
    def get_feature_importance(self) -> pd.DataFrame:
        # GOTCHA: Handle multiclass coefficients
        coef = self.model_.coef_
        
        if coef.shape[0] == 1:  # Binary case
            importance = np.abs(coef[0])
        else:  # Multiclass - aggregate across classes
            # PATTERN: Use mean absolute coefficient across classes
            importance = np.mean(np.abs(coef), axis=0)
            
        # PATTERN: Return sorted DataFrame like other models
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
```

### Integration Points
```yaml
MODEL_COMPARISON:
  - file: model_comparison/experiments/site_comparison.py
  - pattern: Add LogisticRegressionModel to model_configs dict
  
HYPERPARAMETER_TUNING:
  - Consider adding LogisticRegressionHyperparameterTuner later
  - Use GridSearchCV with C values: [0.001, 0.01, 0.1, 1, 10, 100]
  
PARALLEL_EXECUTION:
  - Compatible with Ray/Prefect through sklearn interface
  - No special handling needed
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
cd /Users/ericliu/projects5/context-engineering-intro
poetry run ruff check baseline/models/logistic_regression_*.py --fix
poetry run mypy baseline/models/logistic_regression_*.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# Key test cases to implement:
def test_solver_penalty_compatibility():
    """Test that invalid solver-penalty combinations raise errors"""
    with pytest.raises(ValueError, match="does not support L1"):
        LogisticRegressionConfig(penalty="l1", solver="lbfgs")
        
def test_multinomial_solver_compatibility():
    """Test multinomial support across solvers"""
    # liblinear should work with warn about OvR
    config = LogisticRegressionConfig(solver="liblinear")
    model = LogisticRegressionModel(config)
    # Should not raise error, might warn
    
def test_feature_importance_multiclass():
    """Test feature importance aggregation for multiclass"""
    model = LogisticRegressionModel()
    model.fit(X_multiclass, y_multiclass)
    importance = model.get_feature_importance()
    assert len(importance) == n_features
    assert importance["importance"].min() >= 0  # Absolute values
    
def test_csmf_accuracy_calculation():
    """Test CSMF accuracy matches other implementations"""
    # Create known distribution
    y_true = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    y_pred = np.array(["A"] * 45 + ["B"] * 35 + ["C"] * 20)
    
    model = LogisticRegressionModel()
    csmf_acc = model.calculate_csmf_accuracy(y_true, y_pred)
    
    # Should match XGBoost implementation exactly
    assert 0 <= csmf_acc <= 1
```

```bash
# Run and iterate until passing:
cd /Users/ericliu/projects5/context-engineering-intro
poetry run pytest tests/baseline/test_logistic_regression_model.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with real VA data
cd /Users/ericliu/projects5/context-engineering-intro
poetry run python -c "
from baseline.models import LogisticRegressionModel
from baseline.data import VADataProcessor, DataConfig
import pandas as pd

# Load sample data
config = DataConfig(
    data_path='data/raw/PHMRC/phmrc_adult_cleaned.csv',
    openva_encoding=False  # Use numeric for ML
)
processor = VADataProcessor(config)
data = processor.load_and_process()

# Split features and labels
X = data.drop(columns=['va34'])
y = data['va34']

# Train model
model = LogisticRegressionModel()
model.fit(X.iloc[:100], y.iloc[:100])

# Test predictions
predictions = model.predict(X.iloc[100:110])
print(f'Predictions: {predictions}')

# Test feature importance
importance = model.get_feature_importance()
print(f'Top 5 features: {importance.head()}')
"

# Expected: Successful training and reasonable outputs
```

## Final validation Checklist
- [ ] All tests pass: `poetry run pytest tests/baseline/test_logistic_regression_model.py -v`
- [ ] No linting errors: `poetry run ruff check baseline/models/logistic_regression_*.py`
- [ ] No type errors: `poetry run mypy baseline/models/logistic_regression_*.py`
- [ ] Coverage >95%: `poetry run pytest tests/baseline/test_logistic_regression_model.py --cov=baseline.models.logistic_regression_model --cov-report=term-missing`
- [ ] All regularization options work (L1, L2, ElasticNet, None)
- [ ] Feature importance is interpretable
- [ ] CSMF accuracy calculation matches other models
- [ ] Integration with model comparison framework works

---

## Anti-Patterns to Avoid
- ❌ Don't create new patterns when XGBoost/RandomForest patterns work
- ❌ Don't skip solver-penalty validation - it will fail at runtime
- ❌ Don't ignore class imbalance - use class_weight='balanced'
- ❌ Don't use deprecated multi_class parameter
- ❌ Don't hardcode solver choice - make it configurable
- ❌ Don't forget to handle multiclass coefficients in feature importance

## Confidence Score
**Score: 9/10**

High confidence due to:
- Clear patterns from XGBoost and RandomForest implementations
- Comprehensive scikit-learn documentation available
- Well-defined solver compatibility matrix
- Established test patterns and validation methods
- Simple linear model with fewer hyperparameters than tree-based models

Minor complexity in:
- Solver-penalty compatibility validation
- Multiclass coefficient aggregation for feature importance