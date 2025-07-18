name: "Baseline Data Splitting Module - VA Pipeline Context-Rich PRP"
description: |

## Purpose
Implementation of a simple, robust data splitting module for the baseline VA pipeline that handles site-based splitting, train/test splits, and imbalanced class scenarios while following KISS principles.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a clean, simple data splitting module for the baseline VA pipeline that segments datasets by site and provides train/test splits with proper handling of imbalanced classes and small class sizes, following existing project patterns and KISS principles.

## Why
- **Simplifies complex splitting logic**: Replace over-complicated examples with clean, maintainable code
- **Enables ML pipeline workflows**: Provides foundation for baseline VA model evaluation
- **Handles edge cases gracefully**: Properly manages imbalanced classes and small sample sizes
- **Follows project conventions**: Integrates seamlessly with existing baseline module architecture
- **Supports future analysis**: Designed for extensibility while maintaining simplicity

## What
A data splitting module that provides:
- Site-based data splitting with configurable strategies
- Train/test splits with stratified sampling
- Imbalanced class handling with minimum sample validation
- Cross-validation support for model evaluation
- Integration with existing baseline configuration system
- Comprehensive error handling and validation

### Success Criteria
- [ ] Clean, simple API with single method for splitting
- [ ] Three core splitting strategies implemented (train/test, cross-site, stratified)
- [ ] Integration with existing DataConfig system
- [ ] Minimum sample size validation with graceful error handling
- [ ] >90% test coverage with comprehensive edge case testing
- [ ] Example usage script demonstrating all functionality
- [ ] Follows KISS principle - no over-engineering

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  why: Core splitting functionality with stratify parameter for maintaining class distribution
  
- url: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
  why: Cross-validation with stratified sampling, minimum sample requirements
  
- url: https://scikit-learn.org/stable/modules/cross_validation.html
  why: Comprehensive cross-validation strategies and best practices
  
- url: https://pandas.pydata.org/docs/user_guide/groupby.html
  why: Site-based grouping and stratified sampling patterns
  
- file: baseline/config/data_config.py
  why: Configuration pattern to follow, Pydantic validation approach
  
- file: baseline/data/data_loader_preprocessor.py
  why: Data processing architecture pattern, error handling, logging setup
  
- file: tests/baseline/test_data_loader.py
  why: Test patterns, fixture usage, mocking strategies
  
- file: examples/dataset_splitting_v2.py
  why: Reference implementation to simplify (shows over-complicated approach to avoid)
  
- file: examples/cross_site_splitting.py
  why: Cross-site splitting logic reference (simplify the approach)
  
- file: baseline/DATA_SPLITTING_FEATURE.md
  why: Complete feature requirements and considerations
  
- doc: https://github.com/scikit-learn/scikit-learn/issues/1017
  critical: Minimum sample size requirements - need at least 2 samples per class for stratified splitting
  
- doc: https://stackoverflow.com/questions/43179429/scikit-learn-error-the-least-populated-class-in-y-has-only-1-member
  critical: How to handle "least populated class" errors with single-instance classes
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
baseline/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── data_config.py        # Pydantic configuration system
├── data/
│   ├── __init__.py
│   └── data_loader_preprocessor.py  # Core data processing patterns
├── example_usage.py          # Usage demonstration patterns
└── tests/
    ├── __init__.py
    └── test_data_loader.py    # Test patterns and fixture usage
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
baseline/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── data_config.py        # EXTEND with splitting parameters
├── data/
│   ├── __init__.py
│   ├── data_loader_preprocessor.py
│   └── data_splitter.py      # NEW: Main splitting functionality
├── utils/
│   ├── __init__.py           # NEW: Utilities package
│   ├── split_validator.py    # NEW: Validation utilities for splits
│   └── class_validator.py    # NEW: Small class handling utilities
├── example_usage.py
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_data_splitter.py     # NEW: Comprehensive splitting tests
    └── test_split_validators.py  # NEW: Validation utility tests
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Project uses Poetry for dependency management - use "poetry add" not pip
# CRITICAL: All functions must be under 350 lines as per CLAUDE.md
# CRITICAL: Use venv_linux virtual environment for Python commands
# CRITICAL: Follow KISS principle - prioritize simplicity over complexity
# CRITICAL: DataConfig uses Pydantic with field validators - follow existing patterns
# CRITICAL: VADataProcessor pattern: config injection, private methods, comprehensive logging
# CRITICAL: Tests use pytest with class-based organization, MagicMock for external dependencies
# CRITICAL: All functions require Google-style docstrings with type hints
# CRITICAL: Save results in results/baseline/ directory with timestamped filenames
# CRITICAL: Use pandas for data manipulation, scikit-learn for ML utilities
# CRITICAL: Minimum 2 samples per class for stratified splitting in scikit-learn
# CRITICAL: StratifiedKFold requires at least n_splits samples per class
# CRITICAL: Classes with single instances will cause "least populated class" errors
# CRITICAL: Use Field validation in Pydantic for parameter validation
# CRITICAL: Error handling pattern: log before raising, use specific exception types
# CRITICAL: Import organization: standard library -> third-party -> local imports
```

## Implementation Blueprint

### Data models and structure

Create the core data models to ensure type safety and consistency:
```python
# Extend DataConfig with splitting parameters
class SplitConfig(BaseModel):
    split_strategy: Literal["train_test", "cross_site", "stratified_site"] = "train_test"
    test_size: float = Field(default=0.3, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    site_column: str = Field(default="site")
    label_column: str = Field(default="va34")
    train_sites: Optional[List[str]] = Field(default=None)
    test_sites: Optional[List[str]] = Field(default=None)
    min_samples_per_class: int = Field(default=5, ge=1)
    handle_small_classes: Literal["error", "warn", "exclude"] = "warn"

# Core splitting result structure
class SplitResult(BaseModel):
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: Dict[str, Any]

# Validation result structure
class ValidationResult(BaseModel):
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    class_distribution: Dict[str, int]
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Extend DataConfig with splitting parameters
MODIFY baseline/config/data_config.py:
  - FIND pattern: "class DataConfig(BaseModel):"
  - ADD splitting parameters using Field validation
  - PRESERVE existing field patterns and validators
  - FOLLOW existing validation patterns

Task 2: Create validation utilities
CREATE baseline/utils/__init__.py:
  - STANDARD package initialization with docstring
  
CREATE baseline/utils/class_validator.py:
  - IMPLEMENT minimum sample size validation
  - HANDLE single-instance classes gracefully
  - FOLLOW existing error handling patterns
  - MIRROR validation patterns from config/data_config.py

CREATE baseline/utils/split_validator.py:
  - IMPLEMENT site validation logic
  - VALIDATE split configurations
  - HANDLE edge cases (empty sites, missing columns)
  - FOLLOW existing logging patterns

Task 3: Create core splitting functionality
CREATE baseline/data/data_splitter.py:
  - MIRROR class structure from data_loader_preprocessor.py
  - IMPLEMENT VADataSplitter class following existing patterns
  - PRESERVE error handling and logging approach
  - KEEP methods under 350 lines each

Task 4: Create comprehensive unit tests
CREATE tests/baseline/test_data_splitter.py:
  - MIRROR test structure from test_data_loader.py
  - IMPLEMENT test classes following existing patterns
  - USE pytest fixtures for sample data
  - INCLUDE minimum 3 test cases per function

CREATE tests/baseline/test_split_validators.py:
  - TEST validation utilities comprehensively
  - FOLLOW existing test patterns
  - INCLUDE edge case testing

Task 5: Create example usage script
CREATE baseline/example_splitting.py:
  - MIRROR pattern from example_usage.py
  - DEMONSTRATE all splitting strategies
  - FOLLOW existing example patterns
  - INCLUDE error handling examples

Task 6: Update package initialization
MODIFY baseline/__init__.py:
  - ADD data_splitter imports
  - FOLLOW existing import patterns
  - PRESERVE existing exports
```

### Per task pseudocode as needed added to each task

```python
# Task 1: Extend DataConfig
class DataConfig(BaseModel):
    # ... existing fields ...
    
    # NEW: Splitting parameters
    split_strategy: Literal["train_test", "cross_site", "stratified_site"] = "train_test"
    test_size: float = Field(default=0.3, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    site_column: str = Field(default="site")
    label_column: str = Field(default="va34")
    min_samples_per_class: int = Field(default=5, ge=1)
    handle_small_classes: Literal["error", "warn", "exclude"] = "warn"
    
    @field_validator("test_size")
    def validate_test_size(cls, v):
        # PATTERN: Follow existing validation patterns
        if not 0.1 <= v <= 0.5:
            raise ValueError("test_size must be between 0.1 and 0.5")
        return v

# Task 2: Class validator utility
class ClassValidator:
    """Utility for validating class distributions and handling small classes."""
    
    def __init__(self, min_samples_per_class: int = 5):
        self.min_samples_per_class = min_samples_per_class
    
    def validate_class_distribution(self, y: pd.Series) -> ValidationResult:
        # PATTERN: Count class occurrences
        class_counts = y.value_counts()
        
        # PATTERN: Identify small classes
        small_classes = class_counts[class_counts < self.min_samples_per_class]
        
        # PATTERN: Generate warnings/errors based on findings
        warnings = []
        errors = []
        
        if len(small_classes) > 0:
            # CRITICAL: Handle single-instance classes
            single_instance = class_counts[class_counts == 1]
            if len(single_instance) > 0:
                errors.append(f"Classes with single instance: {single_instance.index.tolist()}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            class_distribution=class_counts.to_dict()
        )

# Task 3: Core splitting functionality
class VADataSplitter:
    """Data splitter for VA datasets with site-based and stratified splitting."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.class_validator = ClassValidator(config.min_samples_per_class)
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, data: pd.DataFrame) -> SplitResult:
        """Main splitting method following the existing processor pattern."""
        # PATTERN: Input validation first
        self._validate_input_data(data)
        
        # PATTERN: Strategy dispatch
        if self.config.split_strategy == "train_test":
            return self._train_test_split(data)
        elif self.config.split_strategy == "cross_site":
            return self._cross_site_split(data)
        elif self.config.split_strategy == "stratified_site":
            return self._stratified_site_split(data)
    
    def _train_test_split(self, data: pd.DataFrame) -> SplitResult:
        """Simple train/test split with stratification."""
        # PATTERN: Validate classes before splitting
        validation = self.class_validator.validate_class_distribution(data[self.config.label_column])
        
        if not validation.is_valid:
            # PATTERN: Handle validation errors
            if self.config.handle_small_classes == "error":
                raise ValueError(f"Class validation failed: {validation.errors}")
            elif self.config.handle_small_classes == "warn":
                self.logger.warning(f"Small classes detected: {validation.warnings}")
        
        # CRITICAL: Use stratify parameter to maintain class distribution
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                data.drop(columns=[self.config.label_column]),
                data[self.config.label_column],
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=data[self.config.label_column]  # CRITICAL: Maintain distribution
            )
        except ValueError as e:
            # GOTCHA: Handle "least populated class" errors
            if "least populated class" in str(e):
                self.logger.warning("Falling back to non-stratified split due to small classes")
                X_train, X_test, y_train, y_test = train_test_split(
                    data.drop(columns=[self.config.label_column]),
                    data[self.config.label_column],
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=None  # Fallback to simple split
                )
            else:
                raise
        
        # PATTERN: Reconstruct DataFrames
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # PATTERN: Generate metadata
        metadata = {
            "split_strategy": self.config.split_strategy,
            "test_size": self.config.test_size,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "train_class_distribution": train_data[self.config.label_column].value_counts().to_dict(),
            "test_class_distribution": test_data[self.config.label_column].value_counts().to_dict()
        }
        
        return SplitResult(train=train_data, test=test_data, metadata=metadata)

# Task 4: Test patterns
class TestVADataSplitter:
    """Test cases for VADataSplitter following existing patterns."""
    
    @pytest.fixture
    def sample_va_data(self):
        """Create sample VA data for testing."""
        return pd.DataFrame({
            "site": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "va34": [1, 1, 2, 2, 1, 1, 2, 2],
            "symptom1": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
            "symptom2": ["no", "yes", "no", "yes", "no", "yes", "no", "yes"]
        })
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        return DataConfig(
            data_path="dummy.csv",
            split_strategy="train_test",
            test_size=0.3,
            random_state=42
        )
    
    def test_train_test_split_success(self, mock_config, sample_va_data):
        """Test successful train/test split."""
        # PATTERN: Create splitter and execute
        splitter = VADataSplitter(mock_config)
        result = splitter.split_data(sample_va_data)
        
        # PATTERN: Verify results
        assert isinstance(result, SplitResult)
        assert len(result.train) + len(result.test) == len(sample_va_data)
        assert result.metadata["split_strategy"] == "train_test"
    
    def test_small_class_handling(self, mock_config, sample_va_data):
        """Test handling of classes with insufficient samples."""
        # PATTERN: Create problematic data
        small_class_data = sample_va_data.copy()
        small_class_data.loc[0, "va34"] = 999  # Single instance class
        
        # PATTERN: Test error handling
        splitter = VADataSplitter(mock_config)
        with pytest.raises(ValueError, match="least populated class"):
            splitter.split_data(small_class_data)
```

### Integration Points
```yaml
CONFIG:
  - extend: baseline/config/data_config.py
  - pattern: "Field validation with descriptive error messages"
  
IMPORTS:
  - add to: baseline/__init__.py
  - pattern: "from baseline.data.data_splitter import VADataSplitter"
  
RESULTS:
  - save to: results/baseline/splits/
  - pattern: "Timestamped directories with metadata files"
  
TESTS:
  - mirror: tests/baseline/test_data_loader.py
  - pattern: "Class-based test organization with fixtures"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check baseline/data/data_splitter.py --fix
ruff check baseline/utils/ --fix
mypy baseline/data/data_splitter.py
mypy baseline/utils/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE comprehensive tests with these patterns:
def test_train_test_split_success():
    """Test basic train/test splitting functionality."""
    # Test happy path with balanced classes
    
def test_stratified_splitting():
    """Test stratified splitting preserves class distribution."""
    # Verify class ratios maintained
    
def test_small_class_error_handling():
    """Test handling of classes with insufficient samples."""
    # Test single-instance class scenario
    
def test_site_based_splitting():
    """Test site-based splitting functionality."""
    # Test cross-site and same-site scenarios
    
def test_config_validation():
    """Test configuration parameter validation."""
    # Test invalid test_size, missing columns, etc.
```

```bash
# Run and iterate until passing:
poetry run pytest tests/baseline/test_data_splitter.py -v
poetry run pytest tests/baseline/test_split_validators.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with real data processing pipeline
cd baseline/
poetry run python example_splitting.py

# Expected: Successful splits with metadata output
# Test all three splitting strategies
# Verify results saved to results/baseline/splits/
```

## Final validation Checklist
- [ ] All tests pass: `poetry run pytest tests/baseline/test_data_splitter.py -v`
- [ ] No linting errors: `ruff check baseline/data/data_splitter.py baseline/utils/`
- [ ] No type errors: `mypy baseline/data/data_splitter.py baseline/utils/`
- [ ] Example script runs successfully: `poetry run python baseline/example_splitting.py`
- [ ] All splitting strategies work correctly
- [ ] Small class handling works as expected
- [ ] Configuration validation prevents invalid inputs
- [ ] Metadata generation includes all required fields
- [ ] Integration with existing DataConfig works
- [ ] Test coverage >90% for all new files

---

## Anti-Patterns to Avoid
- ❌ Don't over-engineer - follow KISS principle strictly
- ❌ Don't ignore "least populated class" errors - handle gracefully
- ❌ Don't skip class distribution validation
- ❌ Don't hardcode column names - use configuration
- ❌ Don't create complex splitting logic when simple works
- ❌ Don't forget to generate comprehensive metadata
- ❌ Don't skip comprehensive error handling for edge cases
- ❌ Don't use different patterns from existing baseline code

---

## PRP Confidence Score: 9/10

**High confidence for one-pass implementation success due to:**
- Comprehensive existing codebase pattern research
- Detailed external documentation references
- Clear implementation blueprint with pseudocode
- Known gotchas and edge cases documented
- Executable validation loops provided
- Specific test patterns identified
- Integration points clearly defined

**Minor risk factors:**
- Complex edge case handling for small classes
- Integration with existing configuration system complexity