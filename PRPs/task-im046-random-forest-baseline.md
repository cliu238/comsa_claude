name: "Random Forest Baseline Model Implementation PRP"
description: |

## Purpose
Implement a Random Forest baseline model for VA cause-of-death prediction following established patterns from XGBoost and InSilicoVA implementations, providing comprehensive context for one-pass implementation success.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Implement a production-ready Random Forest baseline model with sklearn-compatible interface, feature importance analysis, and CSMF accuracy metrics for verbal autopsy cause-of-death prediction.

## Why
- **Business value**: Provides interpretable baseline model for medical professionals
- **Integration**: Complements XGBoost with different ensemble approach
- **Feature insights**: Native feature importance for understanding key symptoms
- **Robustness**: Less prone to overfitting than boosting methods
- **Comparison value**: Essential baseline for model evaluation framework

## What
Create RandomForestModel class with:
- Pydantic configuration for type-safe parameters
- sklearn estimator interface (fit, predict, predict_proba)
- Multiple feature importance methods (MDI, permutation)
- CSMF accuracy calculation matching other models
- Cross-validation with stratification
- Class imbalance handling via class_weight

### Success Criteria
- [x] Model trains on VA data with numeric encoding
- [x] Feature importance provides interpretable insights
- [x] CSMF accuracy matches InSilicoVA/XGBoost implementation
- [x] Integrates with model comparison framework
- [x] Test coverage >90%
- [x] All validation tests pass

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  why: Official API documentation for RandomForestClassifier parameters and methods
  
- url: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
  why: Feature importance comparison - MDI vs permutation importance
  
- file: baseline/models/xgboost_model.py
  why: Pattern to follow for model implementation, sklearn interface, CSMF calculation
  
- file: baseline/models/xgboost_config.py
  why: Pattern for Pydantic configuration with validation
  
- file: tests/baseline/test_xgboost_model.py
  why: Test patterns for model validation, cross-validation, metrics

- doc: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
  section: Random Forests and Feature Importance
  critical: max_features='sqrt' is optimal for classification
```

### Current Codebase tree
```bash
baseline/
├── models/
│   ├── __init__.py
│   ├── insilico_model.py
│   ├── xgboost_config.py
│   ├── xgboost_model.py
│   └── hyperparameter_tuning.py
├── data/
│   ├── data_loader.py
│   └── data_splitter.py
└── utils/
    ├── __init__.py
    └── logging_config.py

tests/
└── baseline/
    ├── test_xgboost_model.py
    └── test_data_splitter.py
```

### Desired Codebase tree with files to be added
```bash
baseline/
├── models/
│   ├── __init__.py              # UPDATE: Export RandomForestModel
│   ├── random_forest_config.py  # NEW: Pydantic configuration
│   ├── random_forest_model.py   # NEW: Main model implementation
│   ├── insilico_model.py
│   ├── xgboost_config.py
│   └── xgboost_model.py

tests/
└── baseline/
    ├── test_random_forest_model.py  # NEW: Comprehensive tests
    ├── test_xgboost_model.py
    └── test_data_splitter.py
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: scikit-learn RandomForest doesn't support sample_weight in predict
# Must use class_weight='balanced' for imbalanced data

# CRITICAL: feature_importances_ gives MDI (biased for high-cardinality)
# Use permutation_importance for unbiased estimates

# CRITICAL: max_features='sqrt' is default and optimal for classification
# Don't use 'auto' - it's deprecated

# CRITICAL: n_jobs=-1 uses all cores but can cause memory issues
# Monitor memory usage for large datasets

# CRITICAL: RandomForest in sklearn>=1.4 supports missing values
# But our VADataProcessor already handles missing data encoding
```

## Implementation Blueprint

### Data models and structure

Create the core configuration model for type safety:
```python
# random_forest_config.py structure:
from pydantic import BaseModel, Field, field_validator

class RandomForestConfig(BaseModel):
    n_estimators: int = Field(default=100, ge=1, le=5000)
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: str = Field(default="sqrt")
    bootstrap: bool = Field(default=True)
    class_weight: Union[str, Dict[int, float], None] = Field(default="balanced")
    n_jobs: int = Field(default=-1)
    random_state: int = Field(default=42)
    oob_score: bool = Field(default=False)
    
    @field_validator("max_features")
    def validate_max_features(cls, v: str) -> str:
        valid = ["sqrt", "log2", None]
        # Also accept float or int but we use str default
```

### List of tasks to be completed in order

```yaml
Task 1:
CREATE baseline/models/random_forest_config.py:
  - MIRROR pattern from: baseline/models/xgboost_config.py
  - ADD RandomForest-specific parameters
  - INCLUDE validators for max_features, class_weight

Task 2:
CREATE baseline/models/random_forest_model.py:
  - MIRROR pattern from: baseline/models/xgboost_model.py
  - IMPLEMENT RandomForestModel(BaseEstimator, ClassifierMixin)
  - ADD fit() with class_weight support
  - ADD predict() and predict_proba()
  - COPY calculate_csmf_accuracy() method exactly
  - ADD get_feature_importance() with MDI and permutation
  - ADD cross_validate() with stratification

Task 3:
MODIFY baseline/models/__init__.py:
  - FIND pattern: "from .xgboost_model import XGBoostModel"
  - ADD after: "from .random_forest_model import RandomForestModel"
  - UPDATE __all__ list

Task 4:
CREATE tests/baseline/test_random_forest_model.py:
  - MIRROR structure from: tests/baseline/test_xgboost_model.py
  - TEST configuration validation
  - TEST model fitting and prediction
  - TEST feature importance methods
  - TEST CSMF accuracy calculation
  - TEST cross-validation
  - ADD edge cases for Random Forest specifics

Task 5:
UPDATE README.md:
  - FIND section on models
  - ADD Random Forest usage example
  - ADD feature importance interpretation guide
```

### Per task pseudocode

```python
# Task 2: Main model implementation structure
class RandomForestModel(BaseEstimator, ClassifierMixin):
    def __init__(self, config: Optional[RandomForestConfig] = None):
        self.config = config or RandomForestConfig()
        self.model_: Optional[RandomForestClassifier] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        # PATTERN: Validate inputs (see xgboost_model.py:119-124)
        # PATTERN: Encode labels (see xgboost_model.py:131-134)
        # CRITICAL: Create RandomForestClassifier with config params
        self.model_ = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            # ... other params
            class_weight=self.config.class_weight,  # Handles imbalance
            random_state=self.config.random_state
        )
        # Fit model
        self.model_.fit(X.values, y_encoded)
        
    def get_feature_importance(self, importance_type: str = "mdi") -> pd.DataFrame:
        # importance_type: "mdi" or "permutation"
        if importance_type == "mdi":
            importance = self.model_.feature_importances_
        else:
            # Use sklearn.inspection.permutation_importance
            # GOTCHA: This needs X and y data
            pass
```

### Integration Points
```yaml
DATABASE:
  - No database changes needed
  
CONFIG:
  - Pattern follows XGBoostConfig exactly
  - No settings.py changes needed
  
MODEL_REGISTRY:
  - add to: baseline/models/__init__.py
  - pattern: "from .random_forest_model import RandomForestModel"
  
EXPERIMENTS:
  - Works with: model_comparison/experiments/site_comparison.py
  - No changes needed - uses same interface as XGBoost
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
cd /Users/ericliu/projects5/context-engineering-intro
source venv_linux/bin/activate

# Check new files
ruff check baseline/models/random_forest_config.py baseline/models/random_forest_model.py --fix
mypy baseline/models/random_forest_config.py baseline/models/random_forest_model.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# Key test cases for test_random_forest_model.py:

def test_default_config():
    """Test default configuration values."""
    config = RandomForestConfig()
    assert config.n_estimators == 100
    assert config.max_features == "sqrt"
    assert config.class_weight == "balanced"

def test_model_fit_predict():
    """Test basic fit and predict functionality."""
    model = RandomForestModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset(set(y_train))

def test_feature_importance_mdi():
    """Test MDI feature importance."""
    model = RandomForestModel()
    model.fit(X_train, y_train)
    importance = model.get_feature_importance("mdi")
    assert len(importance) == X_train.shape[1]
    assert importance["importance"].sum() == pytest.approx(1.0)

def test_csmf_accuracy():
    """Test CSMF accuracy calculation."""
    model = RandomForestModel()
    y_true = pd.Series(["A", "A", "B", "B", "C"])
    y_pred = np.array(["A", "A", "B", "C", "C"])
    csmf_acc = model.calculate_csmf_accuracy(y_true, y_pred)
    assert 0 <= csmf_acc <= 1
```

```bash
# Run tests
poetry run pytest tests/baseline/test_random_forest_model.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with real VA data
cd /Users/ericliu/projects5/context-engineering-intro
poetry run python -c "
from baseline.data.data_loader import VADataProcessor
from baseline.models.random_forest_model import RandomForestModel

# Load data
processor = VADataProcessor()
data = processor.load_data('Adult')

# Prepare features
X = data.drop(['cause34', 'sid'], axis=1)
y = data['cause34']

# Train model
model = RandomForestModel()
model.fit(X[:100], y[:100])  # Small subset for testing

# Get predictions
predictions = model.predict(X[100:110])
print(f'Predictions: {predictions}')

# Get feature importance
importance = model.get_feature_importance('mdi')
print(f'Top 5 features: {importance.head()}')
"

# Expected: Successful training and predictions
```

## Final Validation Checklist
- [ ] All tests pass: `poetry run pytest tests/baseline/ -v`
- [ ] No linting errors: `ruff check baseline/models/random_forest*.py`
- [ ] No type errors: `mypy baseline/models/random_forest*.py`
- [ ] Model trains on VA data successfully
- [ ] Feature importance returns meaningful results
- [ ] CSMF accuracy calculation works correctly
- [ ] Cross-validation produces stable results
- [ ] Integration with comparison framework verified

---

## Anti-Patterns to Avoid
- ❌ Don't modify CSMF accuracy calculation - copy exactly from XGBoost
- ❌ Don't use sample_weight in RandomForest - use class_weight
- ❌ Don't use 'auto' for max_features - it's deprecated
- ❌ Don't implement custom cross-validation - use sklearn's
- ❌ Don't forget to encode labels before training
- ❌ Don't return raw sklearn objects - wrap in DataFrames

## Confidence Score: 9/10
High confidence due to:
- Clear pattern to follow (XGBoost implementation)
- Well-documented sklearn API
- Established test patterns
- No external dependencies or complex integrations
- Straightforward sklearn usage

Minor uncertainty only in permutation importance implementation details.