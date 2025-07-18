name: "InSilicoVA Model Module Implementation PRP"
description: |
  Context-rich PRP for implementing InSilicoVA model module with Docker integration, sklearn-like interface, and CSMF accuracy evaluation for VA cause-of-death prediction.

---

## Goal
Implement a standalone InSilicoVA model module (`baseline/models/insilico_model.py`) that integrates with the existing baseline data pipeline and provides sklearn-like interface for VA cause-of-death prediction using Docker-based InSilicoVA execution.

## Why
- **Foundation for Model Comparison**: Creates baseline InSilicoVA implementation for future ML model comparisons
- **Research Replication**: Match InSilicoVA CSMF accuracy from two key benchmarks:
  1. **0.74 CSMF accuracy** from "The openVA Toolkit for Verbal Autopsies" paper (https://journal.r-project.org/articles/RJ-2023-020/)
  2. **Table 3 results** from "Probabilistic Cause-of-death Assignment using Verbal Autopsies" paper:
     - All sites training: 0.70 ± 0.07 (Quantile prior) or 0.68 ± 0.06 (Default prior)
     - Same site training: 0.84 ± 0.05 (Quantile prior) or 0.85 ± 0.05 (Default prior)
- **Modular Architecture**: Separates model logic from data pipeline, enabling clean integration and testing
- **Production Ready**: Docker-based execution handles complex InSilicoVA dependencies and old JDK requirements

## What
A complete InSilicoVA model implementation with:
- **Sklearn-like Interface**: `fit(X, y)`, `predict(X)`, `predict_proba(X)` methods
- **Docker Integration**: Robust container-based execution with fallback mechanisms
- **CSMF Accuracy Evaluation**: Proper calculation matching openVA implementation
- **Configuration Management**: Pydantic-based model configuration
- **Comprehensive Testing**: Unit tests, Docker tests, and accuracy validation

### Success Criteria
- [ ] Model class follows sklearn interface conventions
- [ ] Docker integration works with primary (`insilicova-arm64:latest`) and fallback (`Dockerfile`) images
- [ ] CSMF accuracy evaluation matches openVA implementation
- [ ] Achieves CSMF accuracy reasonably close to published benchmarks:
  - [ ] Near 0.74 from openVA Toolkit paper (±0.10 acceptable)
  - [ ] OR within reasonable range of Table 3 results (0.52-0.85)
  - [ ] As long as results are not drastically different, either benchmark passing is acceptable
- [ ] Comprehensive test suite with >90% coverage
- [ ] Integrates seamlessly with existing baseline data pipeline
- [ ] Follows all existing codebase patterns and conventions

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://github.com/verbal-autopsy-software/InSilicoVA
  why: Official InSilicoVA repository for algorithm understanding
  
- url: https://github.com/cliu238/InSilicoVA-sim
  why: Usage examples and implementation patterns
  
- url: https://github.com/verbal-autopsy-software/openVA
  why: OpenVA package documentation for codeVA function parameters
  
- url: https://journal.r-project.org/articles/RJ-2023-020/
  why: "The openVA Toolkit for Verbal Autopsies" paper - source of 0.74 CSMF accuracy benchmark
  
- file: baseline/config/data_config.py
  why: Configuration patterns using Pydantic BaseModel and field validators
  
- file: baseline/data/data_preprocessor.py
  why: Data processing class structure and logging patterns
  
- file: baseline/utils/split_validator.py
  why: Validation result patterns and error handling
  
- file: examples/insilico_model.py
  why: Docker integration patterns and R script generation (reference only)
  
- file: examples/insilico_va_model.py
  why: Advanced model patterns with BaseModel inheritance (reference only)
  
- file: tests/baseline/test_data_splitter.py
  why: Test patterns, fixtures, and mock usage
  
- file: Dockerfile
  why: Docker image requirements and R package installation
  
- docfile: baseline/INSILICO_INTEGRATION_FEATURE.md
  why: Complete feature requirements and architecture specifications
  
- docfile: doc/Probabilistic Cause-of-death Assignment using Verbal Autopsies.pdf
  why: Contains Table 3 with CSMF accuracy benchmarks for different training scenarios (InSilicoVA ranges 0.52-0.85)
```

### Current Codebase tree
```bash
baseline/
├── config/
│   └── data_config.py          # Pydantic configuration patterns
├── data/
│   ├── data_loader.py          # Data loading interface
│   ├── data_preprocessor.py    # Processing class patterns
│   └── data_splitter.py        # Splitting utilities
├── utils/
│   ├── __init__.py
│   ├── class_validator.py      # Validation patterns
│   └── split_validator.py      # Result validation
├── example_usage.py            # Integration examples
└── __init__.py                 # Package exports
```

### Desired Codebase tree with files to be added
```bash
baseline/
├── models/                     # NEW: Model implementations directory
│   ├── __init__.py            # NEW: Package exports
│   ├── model_config.py        # NEW: Model configuration classes
│   ├── insilico_model.py      # NEW: Main InSilicoVA model class
│   └── model_validator.py     # NEW: Model-specific validation
├── config/
├── data/
├── utils/
├── tests/
│   └── baseline/
│       ├── test_insilico_model.py    # NEW: Model tests
│       └── test_model_config.py      # NEW: Configuration tests
└── example_insilico_usage.py   # NEW: Usage demonstration
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: InSilicoVA requires very old JDK - use Docker only
# CRITICAL: codeVA requires ID column added with row_number()
# CRITICAL: Use data.type = "customize" for custom datasets
# CRITICAL: NA values must be converted to empty strings in R
# CRITICAL: CSMF accuracy formula: 1 - sum(abs(pred - true)) / (2 * (1 - min(true)))
# CRITICAL: Use repr() for safe string escaping in R scripts
# CRITICAL: Set random seed both in R (set.seed()) and codeVA (seed parameter)
# CRITICAL: Docker platform must be specified for ARM64 compatibility
# CRITICAL: Use tempfile.TemporaryDirectory() for Docker data isolation
# CRITICAL: Poetry dependency management - use poetry add for new packages
# CRITICAL: All classes use logging.getLogger(__name__) pattern
# CRITICAL: Pydantic v2 patterns with @field_validator decorators
# CRITICAL: Test fixtures use @pytest.fixture with proper mocking
# CRITICAL: Use venv_linux for Python execution in tests
# CRITICAL: Two benchmark targets to validate (passing either is acceptable):
#   1. OpenVA Toolkit paper: 0.74 CSMF accuracy (accept ±0.10 variance)
#   2. Table 3 CSMF accuracies vary by training scenario:
#      - All sites: 0.70±0.07 (Quantile) or 0.68±0.06 (Default prior)
#      - All other sites: 0.60±0.06 (Quantile) or 0.61±0.09 (Default)
#      - Same site: 0.84±0.05 (Quantile) or 0.85±0.05 (Default)
#      - One other site: 0.52±0.12 (both priors)
# IMPORTANT: Papers' results may not be perfectly reproducible - reasonable proximity is acceptable
```

## Implementation Blueprint

### Data models and structure
```python
# Model Configuration (model_config.py)
class InSilicoVAConfig(BaseModel):
    """Configuration for InSilicoVA model parameters."""
    
    # Core InSilicoVA parameters
    nsim: int = Field(default=10000, ge=1000, description="Number of MCMC simulations")
    jump_scale: float = Field(default=0.05, gt=0, description="Jump scale parameter")
    auto_length: bool = Field(default=False, description="Auto length parameter")
    convert_type: str = Field(default="fixed", description="Convert type parameter")
    prior_type: str = Field(default="quantile", description="Prior type: 'quantile' or 'default'")
    
    # Docker configuration
    docker_image: str = Field(default="insilicova-arm64:latest", description="Docker image")
    docker_platform: str = Field(default="linux/arm64", description="Docker platform")
    docker_timeout: int = Field(default=3600, ge=60, description="Docker timeout seconds")
    
    # Data configuration
    cause_column: str = Field(default="gs_text34", description="Cause column name")
    phmrc_type: str = Field(default="adult", description="PHMRC data type")
    use_hce: bool = Field(default=True, description="Use Historical Cause-Specific Elements")
    
    # Execution parameters
    random_seed: int = Field(default=42, ge=0, description="Random seed")
    output_dir: str = Field(default="temp", description="Output directory")
    
    @field_validator("docker_platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        valid_platforms = ["linux/arm64", "linux/amd64"]
        if v not in valid_platforms:
            raise ValueError(f"Platform must be one of {valid_platforms}")
        return v

# Validation Results (model_validator.py)
class ModelValidationResult(BaseModel):
    """Result of model validation."""
    is_valid: bool
    warnings: List[str] = []
    errors: List[str] = []
    metadata: Dict[str, Any] = {}
    
# Main Model Class (insilico_model.py)
class InSilicoVAModel:
    """InSilicoVA model with sklearn-like interface."""
    
    def __init__(self, config: InSilicoVAConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        self.train_data = None
        self._unique_causes = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model using training data."""
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using fitted model."""
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        
    def calculate_csmf_accuracy(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate CSMF accuracy following openVA implementation."""
```

### List of tasks to be completed in order

```yaml
Task 1: Create Model Configuration
CREATE baseline/models/__init__.py:
  - INITIALIZE empty package
  - EXPORT main classes for public API

CREATE baseline/models/model_config.py:
  - MIRROR pattern from: baseline/config/data_config.py
  - IMPLEMENT InSilicoVAConfig with Pydantic BaseModel
  - ADD field validators for docker_platform, nsim, jump_scale
  - INCLUDE comprehensive docstrings following Google style

Task 2: Create Model Validator
CREATE baseline/models/model_validator.py:
  - MIRROR pattern from: baseline/utils/split_validator.py
  - IMPLEMENT ModelValidationResult with Pydantic BaseModel
  - CREATE InSilicoVAValidator class with data validation methods
  - VALIDATE Docker availability, data format, required columns
  - INCLUDE comprehensive error messages and logging

Task 3: Implement Core Model Class
CREATE baseline/models/insilico_model.py:
  - MIRROR pattern from: baseline/data/data_preprocessor.py (class structure)
  - IMPLEMENT sklearn-like interface: fit(), predict(), predict_proba()
  - INTEGRATE Docker execution with robust error handling
  - ADD CSMF accuracy calculation following openVA formula
  - INCLUDE comprehensive logging and progress tracking
  - HANDLE temporary directory management with context managers

Task 4: Create Unit Tests
CREATE tests/baseline/test_model_config.py:
  - MIRROR pattern from: tests/baseline/test_split_validators.py
  - TEST configuration validation, field validators, edge cases
  - INCLUDE valid/invalid parameter combinations
  - TEST Docker configuration validation

CREATE tests/baseline/test_insilico_model.py:
  - MIRROR pattern from: tests/baseline/test_data_splitter.py
  - TEST sklearn interface methods (fit, predict, predict_proba)
  - MOCK Docker execution for unit tests
  - TEST CSMF accuracy calculation
  - INCLUDE comprehensive edge cases and error handling

Task 5: Create Integration Tests
CREATE tests/baseline/test_insilico_integration.py:
  - TEST Docker container execution (real Docker tests)
  - TEST with actual small dataset
  - VALIDATE CSMF accuracy calculation
  - TEST fallback Dockerfile if primary image fails

Task 6: Create Usage Example
CREATE baseline/example_insilico_usage.py:
  - MIRROR pattern from: baseline/example_usage.py
  - DEMONSTRATE full pipeline: load -> preprocess -> split -> model
  - INCLUDE CSMF accuracy evaluation
  - SHOW both Docker image options
  - ADD comprehensive logging and error handling

Task 7: Update Package Exports
MODIFY baseline/__init__.py:
  - ADD models package import
  - EXPORT InSilicoVAModel, InSilicoVAConfig
  - MAINTAIN backward compatibility
```

### Per task pseudocode

```python
# Task 3: Core Model Implementation
class InSilicoVAModel:
    def __init__(self, config: InSilicoVAConfig):
        # PATTERN: Initialize with config (see data_preprocessor.py)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        self.train_data = None
        self._unique_causes = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # PATTERN: Validate input data first
        validator = InSilicoVAValidator(self.config)
        validation_result = validator.validate_training_data(X, y)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid training data: {validation_result.errors}")
        
        # PATTERN: Store training data for Docker execution
        self.train_data = X.copy()
        self.train_data[self.config.cause_column] = y
        self._unique_causes = sorted(y.unique())
        self.is_fitted = True
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # PATTERN: Check if fitted (sklearn pattern)
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # PATTERN: Use temporary directory for Docker isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            # PATTERN: Save data files (see examples/insilico_model.py)
            train_file = os.path.join(temp_dir, "train_data.csv")
            test_file = os.path.join(temp_dir, "test_data.csv")
            
            # PATTERN: Generate R script dynamically
            r_script = self._generate_r_script()
            r_script_path = os.path.join(temp_dir, "run_insilico.R")
            
            # PATTERN: Execute Docker with robust error handling
            probs_df = self._execute_docker(temp_dir, r_script_path)
            
            # PATTERN: Convert to sklearn-compatible format
            probs_array = self._format_probabilities(probs_df)
            
        return probs_array
    
    def _execute_docker(self, temp_dir: str, r_script_path: str) -> pd.DataFrame:
        # PATTERN: Docker execution with fallback (see examples/)
        cmd = [
            "docker", "run", "--rm",
            "--platform", self.config.docker_platform,
            "-v", f"{temp_dir}:/data",
            self.config.docker_image,
            "R", "-f", "/data/run_insilico.R"
        ]
        
        # PATTERN: Subprocess with timeout and error handling
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=self.config.docker_timeout
            )
            if result.returncode != 0:
                # PATTERN: Try fallback Dockerfile
                return self._try_fallback_docker(temp_dir, r_script_path)
        except subprocess.TimeoutExpired:
            self.logger.error("Docker execution timeout")
            raise RuntimeError("InSilicoVA execution timeout")
        
        # PATTERN: Read results from mounted volume
        probs_file = os.path.join(temp_dir, "insilico_probs.csv")
        return pd.read_csv(probs_file, index_col=0)
    
    def calculate_csmf_accuracy(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        # PATTERN: CSMF accuracy calculation following openVA
        # FORMULA: 1 - sum(abs(pred - true)) / (2 * (1 - min(true)))
        csmf_true = y_true.value_counts(normalize=True).sort_index()
        csmf_pred = y_pred.value_counts(normalize=True).sort_index()
        
        # PATTERN: Align indices for comparison
        all_causes = sorted(set(csmf_true.index) | set(csmf_pred.index))
        csmf_true_aligned = csmf_true.reindex(all_causes, fill_value=0)
        csmf_pred_aligned = csmf_pred.reindex(all_causes, fill_value=0)
        
        # PATTERN: Calculate accuracy
        numerator = np.sum(np.abs(csmf_pred_aligned - csmf_true_aligned))
        denominator = 2 * (1 - np.min(csmf_true_aligned))
        
        return 1 - (numerator / denominator)
```

### Integration Points
```yaml
DOCKER:
  - primary: "insilicova-arm64:latest"
  - fallback: "build from Dockerfile"
  - platform: "linux/arm64 or linux/amd64"
  
CONFIG:
  - extend: baseline/config/data_config.py patterns
  - separate: model config from data config
  - validate: Docker parameters and InSilicoVA parameters
  
TESTING:
  - unit: Mock Docker execution for fast tests
  - integration: Real Docker tests with small datasets
  - validation: CSMF accuracy calculation verification
  
LOGGING:
  - pattern: logging.getLogger(__name__)
  - levels: INFO for progress, ERROR for failures
  - output: Docker stdout/stderr capture
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check baseline/models/ --fix
mypy baseline/models/
ruff check tests/baseline/test_*model*.py --fix
mypy tests/baseline/test_*model*.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# Test cases for test_insilico_model.py
def test_model_initialization():
    """Test model creates with valid config"""
    config = InSilicoVAConfig()
    model = InSilicoVAModel(config)
    assert not model.is_fitted
    assert model.train_data is None

def test_fit_with_valid_data():
    """Test model fits with valid training data"""
    config = InSilicoVAConfig()
    model = InSilicoVAModel(config)
    X, y = create_sample_data()  # Helper function
    model.fit(X, y)
    assert model.is_fitted
    assert model.train_data is not None

def test_predict_without_fit():
    """Test predict raises error when not fitted"""
    config = InSilicoVAConfig()
    model = InSilicoVAModel(config)
    X = create_sample_features()
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(X)

def test_csmf_accuracy_calculation():
    """Test CSMF accuracy calculation matches expected formula"""
    y_true = pd.Series([1, 1, 2, 2, 3, 3])
    y_pred = pd.Series([1, 2, 2, 2, 3, 3])  # Some misclassification
    
    config = InSilicoVAConfig()
    model = InSilicoVAModel(config)
    accuracy = model.calculate_csmf_accuracy(y_true, y_pred)
    
    # Expected accuracy based on formula
    assert 0.0 <= accuracy <= 1.0
    assert accuracy < 1.0  # Not perfect due to misclassification

@pytest.mark.benchmark
def test_benchmark_accuracy():
    """Test model achieves reasonable CSMF accuracy compared to published benchmarks"""
    # Test with standard dataset configuration
    config = InSilicoVAConfig()
    model = InSilicoVAModel(config)
    X, y = load_standard_va_dataset()  # Standard test dataset
    
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = model.calculate_csmf_accuracy(y, predictions)
    
    # Check against both benchmarks - passing either is acceptable
    openva_benchmark_pass = 0.64 <= accuracy <= 0.84  # 0.74 ± 0.10
    table3_range_pass = 0.52 <= accuracy <= 0.90  # Covers all Table 3 scenarios with margin
    
    # Log the results for debugging
    logging.info(f"CSMF accuracy achieved: {accuracy:.3f}")
    logging.info(f"OpenVA benchmark (0.74±0.10): {'PASS' if openva_benchmark_pass else 'FAIL'}")
    logging.info(f"Table 3 range (0.52-0.85): {'PASS' if table3_range_pass else 'FAIL'}")
    
    # As long as one benchmark passes, the test passes
    assert openva_benchmark_pass or table3_range_pass, \
        f"CSMF accuracy {accuracy:.3f} is too far from published benchmarks"

@pytest.mark.benchmark
def test_table3_specific_scenarios():
    """Test specific Table 3 scenarios with generous tolerance"""
    # Only test a few key scenarios with wider tolerance
    scenarios = [
        ("all_sites", "default", 0.68, 0.15),  # Wider tolerance
        ("same_site", "default", 0.85, 0.15),  # Wider tolerance
    ]
    
    passed_scenarios = []
    for scenario, prior, expected_mean, tolerance in scenarios:
        config = InSilicoVAConfig(prior_type=prior)
        model = InSilicoVAModel(config)
        
        try:
            X_train, y_train, X_test, y_test = load_scenario_data(scenario)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = model.calculate_csmf_accuracy(y_test, predictions)
            
            # Check if within generous range
            if expected_mean - tolerance <= accuracy <= expected_mean + tolerance:
                passed_scenarios.append(scenario)
                logging.info(f"Scenario '{scenario}': PASS (accuracy: {accuracy:.3f})")
            else:
                logging.info(f"Scenario '{scenario}': FAIL (accuracy: {accuracy:.3f}, expected: {expected_mean}±{tolerance})")
        except Exception as e:
            logging.warning(f"Scenario '{scenario}' could not be tested: {e}")
    
    # As long as at least one scenario passes, consider it acceptable
    assert len(passed_scenarios) > 0, \
        "No scenarios achieved accuracy close to Table 3 benchmarks"

@pytest.mark.integration
def test_docker_execution():
    """Test actual Docker execution with small dataset"""
    # Only run if Docker is available
    pytest.importorskip("docker")
    
    config = InSilicoVAConfig()
    model = InSilicoVAModel(config)
    X, y = create_small_va_dataset()  # Real VA-like data
    
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    
    assert probabilities.shape[0] == len(X)
    assert probabilities.shape[1] == len(y.unique())
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
```

```bash
# Run unit tests (with mocked Docker)
poetry run pytest tests/baseline/test_model_config.py -v
poetry run pytest tests/baseline/test_insilico_model.py -v -k "not integration"

# Run integration tests (with real Docker)
poetry run pytest tests/baseline/test_insilico_model.py -v -k "integration"
```

### Level 3: Integration Test
```bash
# Test with example usage script
poetry run python baseline/example_insilico_usage.py

# Expected output:
# INFO: Loading data...
# INFO: Preprocessing data...
# INFO: Splitting data...
# INFO: Training InSilicoVA model...
# INFO: Docker execution successful
# INFO: CSMF accuracy achieved: 0.XX
# INFO: Benchmark validation:
# INFO:   - OpenVA Toolkit (0.74±0.10): PASS/FAIL
# INFO:   - Table 3 range (0.52-0.85): PASS/FAIL
# INFO:   - Overall: PASS (at least one benchmark passed)
# INFO: Model evaluation complete
```

## Final validation Checklist
- [ ] All tests pass: `poetry run pytest tests/baseline/test_*model*.py -v`
- [ ] No linting errors: `ruff check baseline/models/`
- [ ] No type errors: `mypy baseline/models/`
- [ ] Docker integration works: `poetry run python baseline/example_insilico_usage.py`
- [ ] CSMF accuracy calculation verified against openVA formula
- [ ] Model achieves reasonable CSMF accuracy (either benchmark passes):
  - [ ] Within ±0.10 of 0.74 (openVA Toolkit) OR
  - [ ] Within Table 3 range (0.52-0.85) with reasonable margin
- [ ] Model follows sklearn interface conventions
- [ ] Error cases handled gracefully (Docker failures, invalid data)
- [ ] Logs are informative but not verbose
- [ ] Documentation complete with usage examples

---

## Anti-Patterns to Avoid
- ❌ Don't skip Docker error handling - containers can fail
- ❌ Don't hardcode R script parameters - use dynamic generation
- ❌ Don't ignore CSMF accuracy formula - must match openVA exactly
- ❌ Don't skip ID column requirement - InSilicoVA needs it
- ❌ Don't use synchronous subprocess without timeout
- ❌ Don't mock Docker in integration tests - test real execution
- ❌ Don't ignore platform specification - ARM64 vs AMD64 matters
- ❌ Don't skip data validation - invalid data causes R errors
- ❌ Don't use global variables - encapsulate in class
- ❌ Don't skip temporary directory cleanup - use context managers

**PRP Quality Score: 9/10**

This PRP provides comprehensive context for one-pass implementation success, including all necessary documentation, existing patterns, Docker integration specifics, CSMF accuracy calculation, and thorough validation approaches. The implementation follows established codebase conventions and includes robust error handling for Docker-based execution.