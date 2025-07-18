name: "VA Data Processing Pipeline Implementation"
description: |

## Purpose
Implement a baseline Verbal Autopsy (VA) data processing pipeline that integrates with the JH-DSAI/va-data repository for standardized PHMRC data processing, supporting both standard ML algorithms and InSilicoVA.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Build a modular, stage-based data processing pipeline for VA data that:
- Loads and validates PHMRC CSV data using the va-data submodule
- Converts categorical to numeric format for standard ML algorithms
- Supports OpenVA encoding for InSilicoVA compatibility
- Handles VA-specific data patterns and transformations
- Provides robust error handling and progress logging

## Why
- Standardizes VA data processing across the COMSA project
- Enables reproducible research with validated data transformations
- Supports multiple VA algorithms (openVA, InSilicoVA, InterVA) through consistent preprocessing
- Provides clean numeric data for ML model training

## What
A data processing pipeline that loads raw PHMRC CSV files, validates them using PHMRCData class, applies appropriate transformations based on configuration, and outputs clean data ready for ML models.

### Success Criteria
- [ ] Successfully loads PHMRC adult/child/neonate CSV files
- [ ] Validates data integrity with pandera schemas
- [ ] Applies OpenVA encoding when needed for InSilicoVA
- [ ] Converts categorical to numeric for ML compatibility
- [ ] Handles missing values appropriately
- [ ] Supports site-based stratification
- [ ] All unit tests pass
- [ ] Processed data saved to results/baseline/

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- file: examples/data_validation.py
  why: Reference implementation showing PHMRCData usage patterns and preprocessing logic
  
- repo: https://github.com/JH-DSAI/va-data
  path: va_data/phmrc_data.py
  why: PHMRCData class implementation with validate() and xform() methods
  
- repo: https://github.com/JH-DSAI/ml_pipeline
  path: ml_pipeline/stages/data_validation.py
  why: Stage-based architecture pattern to follow
  
- file: data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv
  why: Example PHMRC data file structure to process

- doc: va-data README.md dependencies
  packages: pandas, pandera, pyyaml, matplotlib
  why: Required dependencies for va-data submodule functionality
```

### Current Codebase tree
```bash
context-engineering-intro/
├── baseline/
│   └── data/                    # Currently empty, needs implementation
├── data/
│   └── raw/
│       └── PHMRC/              # Contains PHMRC CSV files
├── examples/
│   └── data_validation.py      # Reference implementation
├── pyproject.toml              # Poetry configuration
├── tests/                      # Test directory
└── results/                    # Output directory
```

### Desired Codebase tree with files
```bash
context-engineering-intro/
├── baseline/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader_preprocessor.py    # Main data processing module
│   └── config/
│       ├── __init__.py
│       └── data_config.py                  # Configuration management
├── va-data/                                # Git submodule (to be added)
├── tests/
│   └── baseline/
│       ├── __init__.py
│       └── test_data_loader.py            # Unit tests
└── results/
    └── baseline/
        └── processed_data/                 # Output directory
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: va-data requires specific import order for transform registration
import va_data.va_data_core.phmrc_plugins  # Must import before PHMRCData
from va_data.va_data_core import PHMRCData

# GOTCHA: PHMRCData.validate() with strict='filter' drops invalid columns
# Use nullable=False to drop rows with null values during validation

# GOTCHA: OpenVA encoding uses specific mappings for InSilicoVA
repldict = {1: "Y", 0: "", 2: "."}  # Must use these exact mappings

# GOTCHA: Pandas future.no_silent_downcasting warning
# Always use: with pd.option_context("future.no_silent_downcasting", True):

# GOTCHA: Poetry is used for dependencies, not pip
# Use: poetry add <package> instead of pip install
```

## Implementation Blueprint

### Data models and structure

```python
# Configuration model using pydantic
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class DataConfig(BaseModel):
    """Configuration for VA data processing."""
    data_path: str = Field(..., description="Path to PHMRC CSV file")
    output_dir: str = Field("results/baseline/", description="Output directory")
    openva_encoding: bool = Field(False, description="Apply OpenVA encoding for InSilicoVA")
    drop_columns: Optional[List[str]] = Field(default_factory=list)
    stratify_by_site: bool = Field(True, description="Enable site-based stratification")
```

### List of tasks to be completed in order

```yaml
Task 1: Add va-data as git submodule
EXECUTE:
  - Command: git submodule add https://github.com/JH-DSAI/va-data
  - Purpose: Add va-data repository as submodule for PHMRCData access
  - Validation: Check va-data directory exists

Task 2: Create baseline package structure
CREATE baseline/__init__.py:
  - Empty file to make baseline a package
CREATE baseline/data/__init__.py:
  - Empty file to make data subpackage
CREATE baseline/config/__init__.py:
  - Empty file to make config subpackage

Task 3: Create data configuration module
CREATE baseline/config/data_config.py:
  - MIRROR pattern from: ml_pipeline/config/settings.py
  - IMPLEMENT: DataConfig pydantic model
  - ADD: Path validation and default values
  - INCLUDE: Logging configuration

Task 4: Create data loader/preprocessor module
CREATE baseline/data/data_loader_preprocessor.py:
  - COPY patterns from: examples/data_validation.py
  - ENHANCE with: Progress logging, error handling
  - IMPLEMENT: Standard ML preprocessing and OpenVA encoding
  - ADD: Site stratification support

Task 5: Create unit tests
CREATE tests/baseline/__init__.py:
  - Empty file for test package
CREATE tests/baseline/test_data_loader.py:
  - MIRROR test patterns from: ml_pipeline/tests/test_data_validation.py
  - TEST: Happy path, validation errors, missing files
  - TEST: Standard ML preprocessing and OpenVA encoding modes

Task 6: Update Poetry dependencies
MODIFY pyproject.toml:
  - ADD: pandera, pyyaml dependencies for va-data
  - RUN: poetry install

Task 7: Create example usage script
CREATE baseline/example_usage.py:
  - Demonstrate loading PHMRC adult data
  - Show both standard ML and OpenVA encoding outputs
  - Save results to appropriate directories
```

### Per task pseudocode

```python
# Task 4: data_loader_preprocessor.py pseudocode
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# CRITICAL: Import order matters
import va_data.va_data_core.phmrc_plugins
from va_data.va_data_core import PHMRCData
from baseline.config.data_config import DataConfig

logger = logging.getLogger(__name__)

class VADataProcessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self._setup_output_dirs()
    
    def load_and_process(self) -> pd.DataFrame:
        # PATTERN: Stage-based processing with logging
        logger.info(f"Loading PHMRC data from {self.config.data_path}")
        
        # Initialize PHMRCData (see examples/data_validation.py:48)
        va_data = PHMRCData(self.config.data_path)
        
        # Validate with appropriate settings
        df = va_data.validate(nullable=False, drop=self.config.drop_columns)
        logger.info(f"Validated {len(df)} records")
        
        # Apply OpenVA transformation
        df = va_data.xform("openva")
        
        # Apply preprocessing based on encoding needs
        if self.config.openva_encoding:
            # GOTCHA: Use exact mapping for InSilicoVA
            repldict = {1: "Y", 0: "", 2: "."}
            cols = df.columns.difference(["site", "va34", "cod5"])
            # Handle pandas warning
            with pd.option_context("future.no_silent_downcasting", True):
                df = df.replace({c: repldict for c in cols})
        else:
            # Convert to numeric for ML models
            df = self._convert_categorical_to_numeric(df)
        
        # Save processed data
        self._save_results(df)
        return df

# Task 5: test_data_loader.py structure
import pytest
from baseline.data.data_loader_preprocessor import VADataProcessor
from baseline.config.data_config import DataConfig

def test_load_phmrc_adult_data():
    """Test loading adult PHMRC data for ML models"""
    config = DataConfig(
        data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        openva_encoding=False
    )
    processor = VADataProcessor(config)
    df = processor.load_and_process()
    
    # Assertions based on expected data
    assert len(df) > 0
    assert "va34" in df.columns
    assert df["va34"].notna().all()
    # Check numeric conversion
    feature_cols = df.columns.difference(["site", "va34", "cod5"])
    for col in feature_cols:
        assert df[col].dtype in ['int64', 'float64']

def test_openva_encoding_mode():
    """Test OpenVA encoding for InSilicoVA"""
    config = DataConfig(
        data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        openva_encoding=True
    )
    processor = VADataProcessor(config)
    df = processor.load_and_process()
    
    # Check OpenVA encoding applied
    feature_cols = df.columns.difference(["site", "va34", "cod5"])
    sample_col = df[feature_cols[0]].dropna()
    # Values should be Y, "", or .
    assert all(val in ["Y", "", "."] for val in sample_col.unique())
```

### Integration Points
```yaml
GIT SUBMODULE:
  - command: "git submodule add https://github.com/JH-DSAI/va-data"
  - update: "git submodule update --init --recursive"
  
POETRY:
  - add to: pyproject.toml
  - packages: "pandera>=0.23.1", "pyyaml>=6.0.2"
  - command: "poetry install"
  
OUTPUT:
  - create: results/baseline/processed_data/
  - format: "{dataset}_{encoding_type}_{timestamp}.csv"
  - encoding_type: "numeric" or "openva"
  - metadata: Save config as JSON alongside data
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
poetry run black baseline/  # Auto-format code
poetry run ruff check baseline/ --fix  # Fix linting issues
poetry run mypy baseline/  # Type checking

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```bash
# Ensure va-data submodule is initialized
git submodule update --init --recursive

# Install dependencies
poetry install

# Run tests with coverage
poetry run pytest tests/baseline/ -v --cov=baseline

# Expected: All tests pass with >80% coverage
# If failing: Check import paths, validate test data exists
```

### Level 3: Integration Test
```bash
# Run example usage script
poetry run python baseline/example_usage.py

# Check outputs exist
ls -la results/baseline/processed_data/

# Validate processed data
poetry run python -c "
import pandas as pd
import glob
files = glob.glob('results/baseline/processed_data/adult_*.csv')
for f in files:
    df = pd.read_csv(f)
    print(f'{f}: {len(df)} records with {len(df.columns)} columns')
    print(f'Target distribution: {df["va34"].value_counts().head()}')"

# Expected: Data loaded successfully with appropriate columns
```

## Final Validation Checklist
- [ ] All tests pass: `poetry run pytest tests/baseline/ -v`
- [ ] No linting errors: `poetry run ruff check baseline/`
- [ ] No type errors: `poetry run mypy baseline/`
- [ ] VA-data submodule properly initialized
- [ ] Processed data saved to results/baseline/
- [ ] Both numeric and OpenVA encoding modes work correctly
- [ ] Logs show progress for long-running operations
- [ ] Configuration is documented and validated

---

## Anti-Patterns to Avoid
- ❌ Don't import PHMRCData before phmrc_plugins
- ❌ Don't use pip instead of poetry for dependencies
- ❌ Don't ignore pandas downcasting warnings
- ❌ Don't hardcode paths - use configuration
- ❌ Don't skip validation even for "trusted" data
- ❌ Don't mix numeric and text encodings in single dataset

## Confidence Score: 9/10
High confidence due to:
- Clear reference implementation in examples/data_validation.py
- Access to both required repositories (va-data and ml_pipeline)
- Simplified to just two output formats (numeric for ML, OpenVA for InSilicoVA)
- Existing test patterns to follow
- Removed complexity of Table 3 compatibility

Minor uncertainty:
- Exact va-data submodule integration steps