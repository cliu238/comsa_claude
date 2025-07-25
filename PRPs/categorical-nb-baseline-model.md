name: "CategoricalNB Baseline Model Implementation"
description: |

## Purpose
Implement a CategoricalNB (Categorical Naive Bayes) baseline model for VA cause-of-death prediction as the final ML baseline model, completing the ML baseline suite alongside XGBoost, Random Forest, and Logistic Regression.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Implement a complete CategoricalNB baseline model that integrates seamlessly with the existing VA cause-of-death prediction framework, providing a probabilistic approach that handles categorical features natively and performs well with missing data common in VA datasets.

## Why
- **Completes ML baseline suite**: Provides the final piece (CategoricalNB) to complete comprehensive ML baseline comparison
- **Native categorical handling**: Eliminates need for extensive preprocessing, working directly with categorical VA data
- **Missing data robustness**: VA questionnaires commonly have missing data which CategoricalNB can handle through categorical encoding strategies
- **Probabilistic interpretability**: Provides interpretable probability estimates suitable for CSMF accuracy calculation
- **Performance expectations**: Expected 70-85% CSMF accuracy, competitive with other baseline models

## What
Implement a complete CategoricalNB model following the established patterns in the codebase:
- Sklearn-compatible interface (fit, predict, predict_proba)
- Pydantic configuration class for hyperparameters
- Integration with existing hyperparameter tuning infrastructure
- Categorical data preprocessing pipeline
- Feature importance via log probability ratios
- CSMF accuracy evaluation compatibility
- Comprehensive unit tests (>95% coverage)

### Success Criteria
- [ ] CategoricalNB model implements consistent sklearn-compatible interface
- [ ] Model handles VA categorical data appropriately with missing values
- [ ] Integration with hyperparameter tuning infrastructure works seamlessly
- [ ] Performance competitive with other baseline models (70-85% CSMF accuracy)
- [ ] Fast training and inference suitable for large-scale experiments
- [ ] Unit test coverage > 95% for new code
- [ ] Documentation includes interpretation guidelines for NB coefficients

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html
  why: Official API documentation for CategoricalNB parameters and methods
  
- url: https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes
  why: Theory and implementation details of categorical naive bayes algorithm
  
- file: /Users/ericliu/projects5/context-engineering-intro/baseline/models/logistic_regression_model.py
  why: Pattern to follow for sklearn-compatible interface and CSMF accuracy calculation
  
- file: /Users/ericliu/projects5/context-engineering-intro/baseline/models/random_forest_model.py
  why: Feature importance implementation patterns and cross-validation structure
  
- file: /Users/ericliu/projects5/context-engineering-intro/baseline/models/xgboost_model.py
  why: Advanced parameter handling and configuration patterns
  
- file: /Users/ericliu/projects5/context-engineering-intro/baseline/models/logistic_regression_config.py
  why: Pydantic config validation patterns and hyperparameter definitions
  
- file: /Users/ericliu/projects5/context-engineering-intro/baseline/data/data_loader_preprocessor.py
  why: Understanding of VA data preprocessing and categorical encoding strategies
  
- file: /Users/ericliu/projects5/context-engineering-intro/tests/baseline/test_logistic_regression_model.py
  why: Testing patterns, edge cases, and validation approaches
  
- doc: https://scikit-learn.org/stable/modules/impute.html
  section: Missing categorical data handling with SimpleImputer
  critical: CategoricalNB expects integer-encoded categorical features without NaN values
</yaml>

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
baseline/
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py          # ✅ Completed
│   ├── random_forest_model.py    # ✅ Completed  
│   ├── logistic_regression_model.py  # ✅ Completed
│   ├── xgboost_config.py
│   ├── random_forest_config.py
│   ├── logistic_regression_config.py
│   ├── model_config.py
│   ├── hyperparameter_tuning.py
│   └── model_validator.py
├── data/
│   ├── data_loader_preprocessor.py  # VADataProcessor with categorical encoding
│   └── data_splitter.py
└── config/
    └── data_config.py

tests/baseline/
├── test_xgboost_model.py
├── test_random_forest_model.py
├── test_logistic_regression_model.py
└── test_model_config.py
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
baseline/models/
├── categorical_nb_model.py       # 🚧 Main CategoricalNB model implementation
├── categorical_nb_config.py      # 🚧 Pydantic configuration for hyperparameters

tests/baseline/
├── test_categorical_nb_model.py  # 🚧 Comprehensive unit tests for model

# Integration points:
model_comparison/hyperparameter_tuning/
├── search_spaces.py              # 🔄 Add CategoricalNB search space
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: CategoricalNB expects integer-encoded categorical features (0, 1, 2, ...)
# Example: CategoricalNB does NOT handle NaN values directly - must preprocess first
# Example: VA data encoding: Y=0, N=1, .=2, DK=2 (missing as separate category)

# CRITICAL: Follow existing sklearn interface pattern
# All models must implement: fit(), predict(), predict_proba(), get_feature_importance()
# Models must inherit from BaseEstimator, ClassifierMixin

# CRITICAL: CSMF accuracy calculation pattern must match exactly
# Use existing calculate_csmf_accuracy() method from other models - DO NOT reimplement

# CRITICAL: Config validation with Pydantic v2
# Use Field() with proper validation, follow existing config patterns
# Include model_validator for complex parameter relationships

# CRITICAL: Handle single class edge case
# VA data may have training sets with only one class - implement graceful handling
# Return dummy predictions that always predict the single class

# CRITICAL: VADataProcessor categorical encoding
# VA data uses mixed encoding: Y/N/./DK -> need consistent integer mapping
# Missing values should be encoded as separate category (not dropped/imputed)

# CRITICAL: Ray/Prefect integration
# Model must work with existing distributed comparison framework
# Use standard pickling, avoid complex internal state
```

## Implementation Blueprint

### Data models and structure

Create the core configuration model using Pydantic v2 patterns:
```python
# categorical_nb_config.py
class CategoricalNBConfig(BaseModel):
    """Configuration for CategoricalNB model following established patterns."""
    
    # Core hyperparameters for CategoricalNB
    alpha: float = Field(default=1.0, gt=0, description="Additive smoothing parameter")
    fit_prior: bool = Field(default=True, description="Whether to learn class prior probabilities")
    class_prior: Optional[np.ndarray] = Field(default=None, description="Prior probabilities of classes")
    force_alpha: bool = Field(default=False, description="Force alpha to be exactly as specified")
    
    # Performance parameters (following established pattern)
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    # Validation following LogisticRegressionConfig pattern
    @field_validator("alpha")
    def validate_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("alpha must be positive")
        return v
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Create CategoricalNB Configuration Class
MODIFY: None needed, create new file
CREATE: baseline/models/categorical_nb_config.py
  - MIRROR pattern from: baseline/models/logistic_regression_config.py
  - MODIFY hyperparameters specific to CategoricalNB (alpha, fit_prior, class_prior, force_alpha)
  - KEEP validation pattern identical (field_validator, model_config)
  - PRESERVE Pydantic v2 compatibility

Task 2: Implement CategoricalNB Model Class
CREATE: baseline/models/categorical_nb_model.py
  - MIRROR pattern from: baseline/models/logistic_regression_model.py
  - MODIFY to use sklearn.naive_bayes.CategoricalNB as underlying model
  - KEEP interface methods identical (fit, predict, predict_proba, cross_validate)
  - PRESERVE CSMF accuracy calculation method exactly
  - HANDLE single class edge case following LogisticRegressionModel pattern

Task 3: Implement Categorical Data Preprocessing Methods
MODIFY: baseline/models/categorical_nb_model.py
  - ADD method: _prepare_categorical_features()
  - INJECT categorical integer encoding logic
  - HANDLE missing values as separate category (encode . and DK as same value)
  - PRESERVE existing VADataProcessor integration

Task 4: Implement Feature Importance via Log Probabilities
MODIFY: baseline/models/categorical_nb_model.py
  - ADD method: get_feature_importance()
  - INJECT log probability ratio calculation for feature importance
  - PRESERVE DataFrame return format matching other models
  - HANDLE multiclass aggregation similar to LogisticRegressionModel

Task 5: Create Comprehensive Unit Tests
CREATE: tests/baseline/test_categorical_nb_model.py
  - MIRROR pattern from: tests/baseline/test_logistic_regression_model.py
  - MODIFY test cases for CategoricalNB-specific behavior
  - KEEP test structure identical (config tests, model tests, edge cases)
  - ADD categorical data specific test cases

Task 6: Integrate with Hyperparameter Tuning Infrastructure
MODIFY: baseline/models/hyperparameter_tuning.py
  - ADD CategoricalNBHyperparameterTuner class
  - MIRROR pattern from: XGBoostHyperparameterTuner
  - MODIFY search space for CategoricalNB parameters
  - PRESERVE Optuna integration and metric optimization

Task 7: Update Model Registry and Imports
MODIFY: baseline/models/__init__.py
  - ADD imports for CategoricalNBModel and CategoricalNBConfig
  - PRESERVE existing imports and structure

Task 8: Integration Testing with VA Data Pipeline
MODIFY: None - create integration test
CREATE: example script for CategoricalNB usage
  - MIRROR pattern from: existing example_usage.py files
  - TEST integration with VADataProcessor categorical encoding
  - VALIDATE CSMF accuracy calculation on VA34 data
```

### Per task pseudocode as needed added to each task

```python
# Task 2: CategoricalNB Model Implementation
class CategoricalNBModel(BaseEstimator, ClassifierMixin):
    def __init__(self, config: Optional[CategoricalNBConfig] = None):
        # PATTERN: Follow exact initialization pattern from LogisticRegressionModel
        self.config = config or CategoricalNBConfig()
        self.model_: Optional[CategoricalNB] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        # ... (follow existing pattern exactly)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CategoricalNBModel":
        # PATTERN: Input validation following LogisticRegressionModel
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # CRITICAL: Categorical encoding - VA data preprocessing
        X_categorical = self._prepare_categorical_features(X)
        
        # GOTCHA: Handle single class case like LogisticRegressionModel
        if len(self.classes_) < 2:
            self._single_class = True
            self._single_class_label = self.classes_[0]
            return self
        
        # PATTERN: Use sklearn CategoricalNB with config parameters
        self.model_ = CategoricalNB(
            alpha=self.config.alpha,
            fit_prior=self.config.fit_prior,
            class_prior=self.config.class_prior,
            force_alpha=self.config.force_alpha
        )
        self.model_.fit(X_categorical, y_encoded)
    
    def _prepare_categorical_features(self, X: pd.DataFrame) -> np.ndarray:
        # CRITICAL: VA-specific categorical encoding
        # Y -> 0, N -> 1, . -> 2, DK -> 2 (missing as separate category)
        # PATTERN: Handle mixed encoding from VADataProcessor
        categorical_mapping = {
            'Y': 0, 'y': 0, 'Yes': 0, 'yes': 0, 1: 0, '1': 0,
            'N': 1, 'n': 1, 'No': 1, 'no': 1, 0: 1, '0': 1,
            '.': 2, 'DK': 2, 'dk': 2, np.nan: 2, 'nan': 2, '': 2
        }
        # Transform and ensure integer values for CategoricalNB
    
    def get_feature_importance(self) -> pd.DataFrame:
        # PATTERN: Follow RandomForestModel return format
        # CRITICAL: Use log probability ratios for NB feature importance
        # log(P(feature=value|class=c)) - log(P(feature=value|class=other))
        # Aggregate across classes for multiclass case
```

### Integration Points
```yaml
CONFIGURATION:
  - add to: baseline/models/categorical_nb_config.py
  - pattern: "Follow LogisticRegressionConfig validation patterns"
  
HYPERPARAMETER_TUNING:
  - add to: baseline/models/hyperparameter_tuning.py
  - pattern: "CategoricalNBHyperparameterTuner following XGBoostHyperparameterTuner"
  - search_space: "{alpha: [0.1, 0.5, 1.0, 2.0, 5.0], fit_prior: [True, False]}"
  
DATA_PREPROCESSING:
  - integrate with: baseline/data/data_loader_preprocessor.py
  - pattern: "Use categorical encoding option in VADataProcessor"
  - ensure: "Missing value encoding as separate category"
  
MODEL_COMPARISON:
  - integrate with: model_comparison framework
  - pattern: "Standard sklearn interface ensures compatibility"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check baseline/models/categorical_nb_model.py --fix
ruff check baseline/models/categorical_nb_config.py --fix
mypy baseline/models/categorical_nb_model.py
mypy baseline/models/categorical_nb_config.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE tests/baseline/test_categorical_nb_model.py with these test cases:

def test_config_default_values():
    """Test default configuration values match expected CategoricalNB params."""
    config = CategoricalNBConfig()
    assert config.alpha == 1.0
    assert config.fit_prior is True
    assert config.class_prior is None

def test_model_fit_predict_basic():
    """Basic functionality works with categorical data."""
    # Use sample VA-like categorical data
    X = pd.DataFrame({
        'symptom1': ['Y', 'N', '.', 'Y', 'DK'],
        'symptom2': ['N', 'Y', 'Y', '.', 'N']
    })
    y = pd.Series(['cause1', 'cause2', 'cause1', 'cause2', 'cause1'])
    
    model = CategoricalNBModel()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in ['cause1', 'cause2'] for pred in predictions)

def test_categorical_encoding():
    """Test VA categorical encoding works correctly."""
    model = CategoricalNBModel()
    X = pd.DataFrame({'feature': ['Y', 'N', '.', 'DK', np.nan]})
    encoded = model._prepare_categorical_features(X)
    # Y=0, N=1, .=2, DK=2, nan=2
    expected = np.array([[0], [1], [2], [2], [2]])
    assert np.array_equal(encoded, expected)

def test_single_class_handling():
    """Handles single class gracefully like LogisticRegressionModel."""
    X = pd.DataFrame({'feature': ['Y', 'N', 'Y']})
    y = pd.Series(['single_cause'] * 3)
    
    model = CategoricalNBModel()
    model.fit(X, y)
    predictions = model.predict(X)
    assert all(pred == 'single_cause' for pred in predictions)

def test_csmf_accuracy_calculation():
    """CSMF accuracy calculation matches existing implementation."""
    model = CategoricalNBModel()
    y_true = pd.Series(['A', 'A', 'B', 'B', 'C'])
    y_pred = np.array(['A', 'B', 'B', 'B', 'C'])
    csmf_acc = model.calculate_csmf_accuracy(y_true, y_pred)
    assert 0 <= csmf_acc <= 1

def test_feature_importance():
    """Feature importance returns proper DataFrame format."""
    X = pd.DataFrame({
        'symptom1': ['Y', 'N', 'Y', 'N'],
        'symptom2': ['Y', 'Y', 'N', 'N']
    })
    y = pd.Series(['cause1', 'cause2', 'cause1', 'cause2'])
    
    model = CategoricalNBModel()
    model.fit(X, y)
    importance_df = model.get_feature_importance()
    
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert len(importance_df) == 2  # Two features
    assert importance_df['importance'].dtype in [np.float64, float]
```

```bash
# Run and iterate until passing:
pytest tests/baseline/test_categorical_nb_model.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with real VA data pipeline
cd /Users/ericliu/projects5/context-engineering-intro
python -c "
from baseline.models.categorical_nb_model import CategoricalNBModel
from baseline.data.data_loader_preprocessor import VADataProcessor
from baseline.config.data_config import DataConfig

# Test integration with VA data processing
config = DataConfig()
processor = VADataProcessor(config)
# Load and process VA data...

print('CategoricalNB integration test passed')
"

# Expected: No errors, model trains and predicts successfully
# If error: Check logs for stack trace and data compatibility issues
```

## Final validation Checklist
- [ ] All tests pass: `pytest tests/baseline/test_categorical_nb_model.py -v`
- [ ] No linting errors: `ruff check baseline/models/categorical_nb_*.py`
- [ ] No type errors: `mypy baseline/models/categorical_nb_*.py`
- [ ] Integration test successful: Model works with VADataProcessor
- [ ] CSMF accuracy calculation matches other models exactly
- [ ] Feature importance provides interpretable results
- [ ] Single class edge case handled gracefully
- [ ] Categorical encoding handles all VA data formats (Y/N/./DK/missing)
- [ ] Performance competitive with other baselines (70-85% CSMF accuracy)

---

## Anti-Patterns to Avoid
- ❌ Don't reimplement CSMF accuracy calculation - use existing method exactly
- ❌ Don't skip categorical encoding validation - CategoricalNB requires integer categories
- ❌ Don't handle missing values by dropping - encode as separate category
- ❌ Don't ignore single class edge case - implement graceful handling
- ❌ Don't use different sklearn interface patterns - follow existing models exactly
- ❌ Don't hardcode categorical mappings - make them configurable/extensible
- ❌ Don't skip hyperparameter tuning integration - include search space definition

## Expected Performance & Validation

### Performance Expectations
- **Baseline Performance**: 70-80% in-domain CSMF accuracy (before tuning)
- **Tuned Performance**: 75-85% in-domain CSMF accuracy  
- **Out-domain Performance**: Expected to generalize well due to probabilistic nature
- **Speed**: Very fast training and inference (similar to Logistic Regression)
- **Memory**: Low memory footprint due to simple categorical statistics

### Key Advantages
- Native categorical feature handling (no extensive preprocessing needed)
- Robust to missing data through categorical encoding strategies
- Interpretable probability estimates suitable for medical applications
- Good baseline for class-imbalanced datasets (VA data often has rare causes)
- Minimal preprocessing requirements compared to other ML models

### Integration Validation
- Model pickles/unpickles correctly for Ray distributed execution
- Works seamlessly with existing cross-validation and bootstrapping infrastructure
- Hyperparameter tuning improves performance measurably
- CSMF accuracy calculation produces results comparable to other baseline models
- Feature importance provides medically interpretable insights

### Critical Success Metrics
1. **Accuracy**: Achieves 70-85% CSMF accuracy on VA34 dataset
2. **Speed**: Training + prediction under 60 seconds for full VA34 dataset
3. **Robustness**: Handles all VA data edge cases (missing values, single class, rare causes)
4. **Integration**: Works with all existing infrastructure without modifications
5. **Interpretability**: Feature importance correlates with known medical relationships