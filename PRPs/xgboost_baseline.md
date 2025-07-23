# XGBoost Baseline Model Implementation PRP

## Overview
Implementation of XGBoost baseline model for VA cause-of-death prediction as part of the classical ML models suite. This model will serve as a high-performance baseline for comparison with both traditional VA algorithms (InSilicoVA, InterVA) and other ML approaches.

## Context and Background

### Existing Model Implementation Reference
The codebase already has a well-structured model implementation - InSilicoVA model. Study these files for patterns:
- `baseline/models/insilico_model.py` - sklearn-like interface pattern
- `baseline/models/model_config.py` - Pydantic configuration pattern
- `baseline/models/model_validator.py` - Validation utilities

### Key Integration Points
1. **VADataProcessor Integration**
   - Located at: `baseline/data/data_loader.py`
   - Use numeric encoding output: `processor.to_numeric(encoding_type='standard')`
   - Handles feature exclusion to prevent data leakage
   - Site-based data splitting through VADataSplitter

2. **Existing Dependencies**
   - XGBoost ^2.0.0 already in pyproject.toml
   - Optuna available for hyperparameter tuning
   - All standard ML packages (sklearn, numpy, pandas) available

### XGBoost Documentation References
- Main documentation: https://xgboost.readthedocs.io/en/stable/
- Python API: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Multi-class classification: https://xgboost.readthedocs.io/en/stable/tutorials/multiclass_classification.html
- Handling missing values: https://xgboost.readthedocs.io/en/stable/faq.html#how-to-deal-with-missing-value
- Hyperparameter tuning: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html

## Implementation Blueprint

### File Structure
```
baseline/models/
├── xgboost_model.py          # Main XGBoost implementation
├── xgboost_config.py         # Configuration using Pydantic
├── hyperparameter_tuning.py  # Tuning utilities using Optuna
└── tests/
    └── test_xgboost_model.py # Comprehensive unit tests
```

### Configuration Schema (xgboost_config.py)
```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any

class XGBoostConfig(BaseModel):
    """Configuration for XGBoost model following InSilicoVA pattern."""
    
    # Model parameters
    n_estimators: int = Field(default=100, ge=1, description="Number of boosting rounds")
    max_depth: int = Field(default=6, ge=1, le=20, description="Maximum tree depth")
    learning_rate: float = Field(default=0.3, gt=0, le=1, description="Learning rate")
    subsample: float = Field(default=1.0, gt=0, le=1, description="Subsample ratio")
    colsample_bytree: float = Field(default=1.0, gt=0, le=1, description="Column subsample ratio")
    
    # Multi-class specific
    objective: str = Field(default="multi:softprob", description="Objective function")
    num_class: Optional[int] = Field(default=None, description="Number of classes (auto-detected)")
    
    # Class imbalance handling
    scale_pos_weight: Optional[Dict[int, float]] = Field(default=None, description="Class weights")
    
    # Performance
    tree_method: str = Field(default="hist", description="Tree construction algorithm")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")
    n_jobs: int = Field(default=-1, description="Number of parallel threads")
    
    # Regularization
    reg_alpha: float = Field(default=0, ge=0, description="L1 regularization")
    reg_lambda: float = Field(default=1, ge=0, description="L2 regularization")
    
    # Missing values
    missing: float = Field(default=float('nan'), description="Value to treat as missing")
    
    # Early stopping
    early_stopping_rounds: Optional[int] = Field(default=10, description="Early stopping rounds")
    
    @field_validator('tree_method')
    def validate_tree_method(cls, v):
        valid_methods = ['auto', 'exact', 'approx', 'hist', 'gpu_hist']
        if v not in valid_methods:
            raise ValueError(f"tree_method must be one of {valid_methods}")
        return v
```

### Main Model Implementation (xgboost_model.py)
```python
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, Any, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from baseline.models.xgboost_config import XGBoostConfig
import logging

logger = logging.getLogger(__name__)

class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost model for VA cause-of-death prediction.
    
    Follows sklearn interface pattern like InSilicoVA model.
    """
    
    def __init__(self, config: Optional[XGBoostConfig] = None):
        """Initialize XGBoost model with configuration."""
        self.config = config or XGBoostConfig()
        self.model_: Optional[xgb.Booster] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_names_: Optional[list] = None
        self.classes_: Optional[np.ndarray] = None
        self._is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[list] = None) -> 'XGBoostModel':
        """Fit XGBoost model following InSilicoVA pattern."""
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
            
        # Store feature names
        self.feature_names_ = X.columns.tolist()
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        
        # Update num_class if not set
        if self.config.num_class is None:
            self.config.num_class = len(self.classes_)
            
        # Create DMatrix with missing value handling
        dtrain = xgb.DMatrix(
            X.values, 
            label=y_encoded,
            feature_names=self.feature_names_,
            weight=sample_weight,
            missing=self.config.missing
        )
        
        # Prepare parameters
        params = self._get_xgb_params()
        
        # Prepare eval list if provided
        evals = []
        if eval_set is not None:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                y_eval_encoded = self.label_encoder_.transform(y_eval)
                deval = xgb.DMatrix(
                    X_eval.values,
                    label=y_eval_encoded,
                    feature_names=self.feature_names_,
                    missing=self.config.missing
                )
                evals.append((deval, f'eval_{i}'))
        
        # Train model
        callbacks = []
        if self.config.early_stopping_rounds and evals:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.config.early_stopping_rounds,
                    save_best=True
                )
            )
            
        self.model_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals or [(dtrain, 'train')],
            callbacks=callbacks,
            verbose_eval=True
        )
        
        self._is_fitted = True
        logger.info(f"XGBoost model trained with {self.model_.num_boosted_rounds()} rounds")
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cause of death."""
        self._check_is_fitted()
        proba = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability distribution over causes."""
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        # Create DMatrix
        dtest = xgb.DMatrix(
            X.values,
            feature_names=self.feature_names_,
            missing=self.config.missing
        )
        
        # Get predictions
        proba = self.model_.predict(dtest)
        
        # Ensure 2D array
        if proba.ndim == 1:
            proba = proba.reshape(-1, self.config.num_class)
            
        return proba
        
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance scores."""
        self._check_is_fitted()
        
        importance = self.model_.get_score(importance_type=importance_type)
        
        # Create DataFrame
        df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ])
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
        
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, stratified: bool = True) -> Dict[str, Any]:
        """Perform cross-validation with stratification."""
        if stratified:
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
            
        scores = {
            'csmf_accuracy': [],
            'cod_accuracy': [],
            'train_score': [],
            'val_score': []
        }
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone model for this fold
            model = XGBoostModel(config=self.config)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            # Calculate metrics
            y_pred = model.predict(X_val)
            
            # CSMF accuracy
            csmf_acc = self.calculate_csmf_accuracy(y_val, y_pred)
            scores['csmf_accuracy'].append(csmf_acc)
            
            # COD accuracy
            cod_acc = (y_val == y_pred).mean()
            scores['cod_accuracy'].append(cod_acc)
            
        return {k: np.mean(v) for k, v in scores.items()}
        
    def calculate_csmf_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate CSMF accuracy following InSilicoVA implementation."""
        # Get true and predicted fractions
        true_fractions = y_true.value_counts(normalize=True)
        pred_fractions = pd.Series(y_pred).value_counts(normalize=True)
        
        # Align categories
        all_categories = set(true_fractions.index) | set(pred_fractions.index)
        true_fractions = true_fractions.reindex(all_categories, fill_value=0)
        pred_fractions = pred_fractions.reindex(all_categories, fill_value=0)
        
        # Calculate CSMF accuracy
        diff = np.abs(true_fractions - pred_fractions).sum()
        min_frac = true_fractions.min()
        csmf_accuracy = 1 - diff / (2 * (1 - min_frac))
        
        return max(0, csmf_accuracy)  # Ensure non-negative
        
    def _get_xgb_params(self) -> Dict[str, Any]:
        """Convert config to XGBoost parameters."""
        params = {
            'objective': self.config.objective,
            'num_class': self.config.num_class,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'tree_method': self.config.tree_method,
            'device': self.config.device,
            'nthread': self.config.n_jobs,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'missing': self.config.missing,
            'eval_metric': 'mlogloss'
        }
        
        # Handle class weights for imbalanced data
        if self.config.scale_pos_weight is not None:
            # XGBoost doesn't directly support per-class weights in multi-class
            # We'll need to use sample weights instead
            logger.warning("scale_pos_weight not directly supported for multi-class; use sample_weight in fit()")
            
        return params
        
    def _check_is_fitted(self):
        """Check if model is fitted."""
        if not self._is_fitted or self.model_ is None:
            raise ValueError("Model must be fitted before prediction")
```

### Hyperparameter Tuning (hyperparameter_tuning.py)
```python
import optuna
from optuna import Trial
from typing import Dict, Any, Optional
import numpy as np
from baseline.models.xgboost_model import XGBoostModel
from baseline.models.xgboost_config import XGBoostConfig
import logging

logger = logging.getLogger(__name__)

class XGBoostHyperparameterTuner:
    """Hyperparameter tuning for XGBoost using Optuna."""
    
    def __init__(self, base_config: Optional[XGBoostConfig] = None):
        self.base_config = base_config or XGBoostConfig()
        
    def objective(self, trial: Trial, X, y, cv=5) -> float:
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters
        config_dict = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }
        
        # Update config
        config = XGBoostConfig(**{**self.base_config.model_dump(), **config_dict})
        
        # Train and evaluate
        model = XGBoostModel(config=config)
        cv_results = model.cross_validate(X, y, cv=cv)
        
        # Return negative CSMF accuracy (Optuna minimizes)
        return -cv_results['csmf_accuracy']
        
    def tune(self, X, y, n_trials=100, cv=5) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X, y, cv),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"Best CSMF accuracy: {-study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': -study.best_value,
            'study': study
        }
```

## Critical Implementation Details

### 1. Multi-class Classification Setup
XGBoost requires specific configuration for multi-class:
```python
params = {
    'objective': 'multi:softprob',  # Returns probability matrix
    'num_class': n_classes,         # Must be specified
    'eval_metric': 'mlogloss'       # Multi-class log loss
}
```

### 2. Missing Value Handling
XGBoost has native support for missing values:
```python
# Specify missing value indicator
dtrain = xgb.DMatrix(X, label=y, missing=np.nan)
# XGBoost learns optimal default directions for missing values
```

### 3. Class Imbalance Handling
For multi-class problems, use sample weights:
```python
# Calculate class weights
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y)
model.fit(X, y, sample_weight=sample_weights)
```

### 4. CSMF Accuracy Calculation
Follow the InSilicoVA implementation pattern:
```python
# Formula: 1 - sum(|pred_fraction - true_fraction|) / (2 * (1 - min(true_fraction)))
```

### 5. Feature Importance
XGBoost provides multiple importance types:
- 'gain': Average gain when feature is used
- 'weight': Number of times feature appears in trees
- 'cover': Average coverage of feature

## Task List (in order)

1. Create `baseline/models/xgboost_config.py` with Pydantic configuration
2. Create `baseline/models/xgboost_model.py` with main implementation
3. Create `baseline/models/hyperparameter_tuning.py` for Optuna integration
4. Create `baseline/models/__init__.py` exports
5. Create comprehensive tests in `tests/baseline/test_xgboost_model.py`
6. Update `baseline/models/__init__.py` to export new classes
7. Run validation and fix any issues

## Validation Gates

### Syntax and Style Check
```bash
# Format code
poetry run black baseline/models/xgboost*.py tests/baseline/test_xgboost*.py

# Lint code
poetry run ruff check --fix baseline/models/xgboost*.py tests/baseline/test_xgboost*.py

# Type check
poetry run mypy baseline/models/xgboost*.py
```

### Unit Tests
```bash
# Run XGBoost-specific tests with coverage
poetry run pytest tests/baseline/test_xgboost_model.py -v --cov=baseline.models.xgboost_model --cov=baseline.models.xgboost_config --cov=baseline.models.hyperparameter_tuning --cov-report=term-missing

# Ensure >95% coverage
poetry run pytest tests/baseline/test_xgboost_model.py --cov=baseline.models --cov-fail-under=95
```

### Integration Test
```bash
# Test with actual VA data pipeline
poetry run python -c "
from baseline.data.data_loader import VADataProcessor
from baseline.models.xgboost_model import XGBoostModel

# Load sample data
processor = VADataProcessor()
df = processor.load_data('data/sample_data.csv')
X, y = processor.prepare_for_ml(df)

# Train model
model = XGBoostModel()
model.fit(X.iloc[:100], y.iloc[:100])
print('CSMF Accuracy:', model.calculate_csmf_accuracy(y.iloc[100:150], model.predict(X.iloc[100:150])))
"
```

## Common Gotchas and Solutions

1. **XGBoost Version Compatibility**: Ensure using XGBoost >= 2.0.0 for latest features
2. **Memory Usage**: Use `tree_method='hist'` for better memory efficiency
3. **GPU Support**: Set `device='cuda'` only if CUDA is available
4. **Early Stopping**: Requires validation set to work properly
5. **Reproducibility**: Set random seeds in both XGBoost params and data splitting

## Testing Patterns Reference

Follow the patterns from `tests/baseline/test_insilico_model.py`:
- Use fixtures for mock data and configurations
- Test initialization, fitting, prediction separately
- Include edge cases (empty data, single class, etc.)
- Mock external dependencies
- Test all public methods
- Verify error messages for invalid inputs

## External Resources

- XGBoost Python API Docs: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Multi-class Classification Tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/multiclass_classification.html
- Handling Missing Values: https://xgboost.readthedocs.io/en/stable/faq.html#how-to-deal-with-missing-value
- Optuna Documentation: https://optuna.readthedocs.io/en/stable/

## Quality Score: 9/10

This PRP provides comprehensive context including:
- Complete code structure with working implementations
- Integration patterns from existing codebase
- XGBoost-specific configurations and gotchas
- Executable validation gates
- Clear task ordering
- Testing patterns that match the codebase style

The implementation should succeed in one pass with this level of detail.