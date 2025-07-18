name: "AP-Only InSilicoVA Evaluation - R Journal 2023 Replication"
description: |

## Goal
Implement Andhra Pradesh (AP)-only testing methodology to match R Journal 2023 experimental setup and validate our InSilicoVA implementation against published benchmarks. This will enable fair comparison with the literature-reported 0.74 CSMF accuracy by using identical experimental conditions.

## Why
- **Literature Validation**: Our current 0.791 CSMF accuracy uses mixed-site testing (easier) vs R Journal 2023's 0.74 using AP-only testing (harder) - results aren't comparable
- **Research Credibility**: Need fair benchmark comparison using identical experimental conditions for scientific validity  
- **Implementation Verification**: Confirm our model matches published performance when tested under the same conditions
- **Comprehensive Evaluation**: Provide both within-distribution (current) and cross-site generalization (new) performance metrics

## What
Create an AP-only evaluation script that replicates the exact R Journal 2023 methodology:
- **Training**: 5 sites excluding Andhra Pradesh (Mexico, Dar, UP, Bohol, Pemba) - ~6,287 samples
- **Testing**: Andhra Pradesh only - ~1,554 samples  
- **Model Config**: Research-grade InSilicoVA parameters matching literature
- **Output**: Direct CSMF accuracy comparison with 0.74 benchmark

### Success Criteria
- [ ] AP-only testing produces CSMF accuracy within ±0.05 of R Journal 2023 benchmark (0.74)
- [ ] Script automatically configures cross-site split with correct site assignments
- [ ] Results are saved with clear methodology documentation
- [ ] Both experimental setups (mixed-site vs AP-only) are documented and compared
- [ ] Implementation validated as research-grade against published literature

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://journal.r-project.org/articles/RJ-2023-020/
  why: Original R Journal 2023 paper with exact methodology we're replicating
  section: InSilicoVA results section, training/test site configuration
  critical: AP-only testing uses geographic generalization (harder) vs our mixed-site (easier)

- file: baseline/data/data_splitter.py
  why: Already implements _cross_site_split() method with train_sites/test_sites support
  pattern: Lines 180-221 show exact implementation we need to use
  critical: Method validates sites exist and splits data correctly

- file: baseline/config/data_config.py  
  why: Already has train_sites and test_sites parameters for cross-site strategy
  pattern: Lines 43-48 show configuration parameters
  critical: Use split_strategy="cross_site" with specific site assignments

- file: baseline/run_full_insilico.py
  why: Existing pattern for full InSilicoVA evaluation with result saving
  pattern: Complete evaluation pipeline with proper result documentation
  critical: Mirror this structure but with cross-site configuration

- file: EXPERIMENTAL_SETUP_COMPARISON.md
  why: Detailed analysis of methodology differences and requirements
  section: R Journal 2023 Methodology section
  critical: Exact site assignments and expected outcomes documented
```

### Current Codebase Tree (key sections)
```bash
baseline/
├── config/
│   └── data_config.py          # Has train_sites/test_sites support
├── data/
│   ├── data_loader_preprocessor.py  # Data processing pipeline
│   └── data_splitter.py             # Cross-site splitting already implemented
├── models/
│   ├── insilico_model.py            # InSilicoVA implementation
│   └── model_config.py              # Model configuration
└── run_full_insilico.py             # Current mixed-site evaluation script

tests/baseline/
├── test_data_splitter.py       # Test patterns for cross-site splitting
└── test_insilico_model.py      # Model testing patterns

results/insilico_full_run/      # Current results (mixed-site)
├── benchmark_results.csv
├── predictions.csv
└── full_results.json
```

### Desired Codebase Tree with New Files
```bash
baseline/
└── run_ap_only_insilico.py     # NEW: AP-only evaluation script

results/
└── ap_only_insilico/           # NEW: AP-only results directory
    ├── benchmark_results.csv   # AP-only CSMF accuracy results
    ├── predictions.csv         # Individual predictions
    ├── full_results.json       # Complete experiment results
    └── methodology_comparison.md # Comparison with mixed-site results
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Site names in PHMRC data are specific strings
PHMRC_SITES = ["Mexico", "Dar", "AP", "UP", "Bohol", "Pemba"]
# - Must use exact case-sensitive site names
# - "Dar" = Dar es Salaam, "AP" = Andhra Pradesh, "UP" = Uttar Pradesh

# CRITICAL: Cross-site split requires specific configuration
config = DataConfig(
    split_strategy="cross_site",  # NOT "train_test"
    train_sites=["Mexico", "Dar", "UP", "Bohol", "Pemba"],  # 5 sites
    test_sites=["AP"],  # 1 site only
    # test_size is IGNORED in cross_site strategy
)

# CRITICAL: InSilicoVA Docker timeout needs adjustment for cross-site
# - Cross-site has different data size than mixed-site
# - AP-only test set is ~1,554 samples vs 1,517 in mixed-site

# CRITICAL: Feature exclusion must work with cross-site splits
# - Same _get_label_equivalent_columns() logic applies
# - Ensure no data leakage in training sites

# GOTCHA: CSMF accuracy expected to be LOWER than 0.791
# - Geographic generalization is harder than within-distribution
# - 0.74 ± 0.05 is expected range based on R Journal 2023
```

## Implementation Blueprint

### Data Models and Structure
The existing data models are sufficient:
```python
# DataConfig already supports cross-site parameters
DataConfig(
    split_strategy="cross_site",
    train_sites=List[str],  # Existing field
    test_sites=List[str],   # Existing field
)

# InSilicoVAConfig unchanged - same model parameters
InSilicoVAConfig(
    nsim=5000,  # Match current setup
    docker_timeout=3600,  # 1 hour for full dataset
)

# SplitResult unchanged - same output structure
SplitResult(train=DataFrame, test=DataFrame, metadata=dict)
```

### List of Tasks to Complete (in order)

```yaml
Task 1: CREATE baseline/run_ap_only_insilico.py
  - COPY from: baseline/run_full_insilico.py 
  - MODIFY: Configuration section to use cross_site strategy
  - PRESERVE: All existing result saving and logging functionality
  - MODIFY: Output directory to "results/ap_only_insilico/"

Task 2: UPDATE configuration in new script
  - FIND: data_config = DataConfig(...) section
  - REPLACE: split_strategy="train_test" with split_strategy="cross_site"
  - ADD: train_sites=["Mexico", "Dar", "UP", "Bohol", "Pemba"]
  - ADD: test_sites=["AP"]
  - REMOVE: test_size parameter (unused in cross_site)

Task 3: UPDATE logging and documentation
  - MODIFY: Log messages to indicate "AP-only evaluation"
  - UPDATE: Step descriptions to mention cross-site methodology
  - ADD: Expected sample sizes based on site-specific splits

Task 4: ADD methodology comparison functionality
  - CREATE: Function to compare AP-only vs mixed-site results
  - READ: Previous mixed-site results from results/insilico_full_run/
  - GENERATE: Comparative analysis in results

Task 5: CREATE unit tests for AP-only evaluation
  - MIRROR: Existing test patterns from tests/baseline/test_insilico_model.py
  - TEST: Cross-site configuration validation
  - TEST: Site assignment correctness
  - MOCK: InSilicoVA execution for faster testing

Task 6: VALIDATE end-to-end execution
  - RUN: AP-only evaluation script
  - VERIFY: CSMF accuracy within expected range (0.74 ± 0.05)
  - COMPARE: Results with mixed-site methodology
  - DOCUMENT: Findings and validation status
```

### Task 1 Pseudocode: Create AP-Only Evaluation Script
```python
#!/usr/bin/env python
"""Run InSilicoVA with AP-only testing to match R Journal 2023 methodology."""

def main():
    # PATTERN: Mirror run_full_insilico.py structure exactly
    logger.info("=== InSilicoVA AP-Only Evaluation (R Journal 2023 Replication) ===")
    
    # CRITICAL: Use cross_site strategy, not train_test
    data_config = DataConfig(
        data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        output_dir="results/ap_only_insilico/",  # Different output dir
        openva_encoding=True,
        split_strategy="cross_site",  # KEY CHANGE
        train_sites=["Mexico", "Dar", "UP", "Bohol", "Pemba"],  # 5 sites
        test_sites=["AP"],  # AP only
        random_state=42,
        label_column="va34",
        site_column="site"
    )
    
    # PATTERN: Same data processing pipeline
    processor = VADataProcessor(data_config)
    processed_data = processor.load_and_process()
    
    # PATTERN: Same splitting (but different strategy)
    splitter = VADataSplitter(data_config)
    split_result = splitter.split_data(processed_data)
    
    # VERIFY: Expected sample sizes from R Journal 2023
    expected_train_size = 6287  # Approximately
    expected_test_size = 1554   # Approximately
    
    # PATTERN: Same feature exclusion and model training
    # PATTERN: Same result saving structure
    
    # NEW: Add methodology comparison
    compare_with_mixed_site_results()
```

### Integration Points
```yaml
CONFIG:
  - no changes: All parameters already exist in DataConfig
  
DIRECTORIES:
  - create: results/ap_only_insilico/ (automatic via DataConfig.output_dir)
  
VALIDATION:
  - verify: Site names match PHMRC data exactly
  - check: Cross-site split produces expected sample sizes
  - validate: CSMF accuracy within literature range

DOCUMENTATION:
  - update: ACTUAL_PERFORMANCE_RESULTS.md with AP-only results
  - create: Comparative analysis between methodologies
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
poetry run ruff check baseline/run_ap_only_insilico.py --fix
poetry run mypy baseline/run_ap_only_insilico.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE test_ap_only_evaluation.py with these test cases:
def test_ap_only_config_validation():
    """Cross-site configuration is valid"""
    config = DataConfig(
        data_path="va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
        split_strategy="cross_site",
        train_sites=["Mexico", "Dar", "UP", "Bohol", "Pemba"],
        test_sites=["AP"]
    )
    assert config.split_strategy == "cross_site"
    assert "AP" not in config.train_sites
    assert "AP" in config.test_sites

def test_site_assignments():
    """Site assignments match R Journal 2023"""
    expected_train = {"Mexico", "Dar", "UP", "Bohol", "Pemba"}
    expected_test = {"AP"}
    
    # Test using mock data with all sites
    mock_data = create_mock_phmrc_data_with_all_sites()
    config = create_ap_only_config()
    splitter = VADataSplitter(config)
    result = splitter.split_data(mock_data)
    
    train_sites = set(result.train['site'].unique())
    test_sites = set(result.test['site'].unique())
    
    assert train_sites == expected_train
    assert test_sites == expected_test

def test_expected_sample_sizes():
    """Sample sizes approximately match R Journal 2023"""
    # Test with real data (if available) or realistic mock
    result = run_ap_only_split_with_real_data()
    
    # R Journal 2023 reported sizes (approximate)
    assert 6200 <= len(result.train) <= 6400  # ~6,287
    assert 1500 <= len(result.test) <= 1600   # ~1,554
```

```bash
# Run and iterate until passing:
poetry run pytest tests/test_ap_only_evaluation.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test the full evaluation pipeline
poetry run python baseline/run_ap_only_insilico.py

# Expected outputs:
# - CSMF accuracy between 0.69-0.79 (R Journal 2023: 0.74)
# - Training samples: ~6,287
# - Test samples: ~1,554
# - Results saved to results/ap_only_insilico/

# Validate results structure:
ls results/ap_only_insilico/
# Expected: benchmark_results.csv, predictions.csv, full_results.json

# Check CSMF accuracy:
poetry run python -c "
import pandas as pd
results = pd.read_csv('results/ap_only_insilico/benchmark_results.csv')
csmf_acc = results[results['metric'] == 'CSMF_accuracy']['value'].iloc[0]
print(f'AP-only CSMF accuracy: {csmf_acc:.3f}')
assert 0.69 <= csmf_acc <= 0.79, f'Unexpected CSMF accuracy: {csmf_acc}'
"
```

## Final Validation Checklist
- [ ] All tests pass: `poetry run pytest tests/ -v`
- [ ] No linting errors: `poetry run ruff check baseline/`  
- [ ] No type errors: `poetry run mypy baseline/`
- [ ] AP-only evaluation runs successfully: `poetry run python baseline/run_ap_only_insilico.py`
- [ ] CSMF accuracy within expected range: 0.69-0.79 (target: 0.74)
- [ ] Sample sizes match R Journal 2023: ~6,287 train, ~1,554 test
- [ ] Results properly saved and documented
- [ ] Methodology comparison generated and accurate
- [ ] EXPERIMENTAL_SETUP_COMPARISON.md updated with findings

---

## Expected Outcomes

### Primary Success Metrics
- **CSMF Accuracy**: 0.74 ± 0.05 (matching R Journal 2023)
- **Implementation Validation**: Confirms our model is research-grade
- **Methodology Clarity**: Clear comparison between evaluation approaches

### Secondary Benefits
- **Literature Credibility**: Fair benchmark comparison with published work
- **Research Completeness**: Both within-distribution and cross-site evaluation
- **Future Research**: Foundation for geographic generalization studies

## Anti-Patterns to Avoid
- ❌ Don't modify existing mixed-site evaluation - create separate script
- ❌ Don't use test_size parameter with cross_site strategy - it's ignored  
- ❌ Don't expect same CSMF accuracy as mixed-site - geographic generalization is harder
- ❌ Don't hardcode site names without validating against actual data
- ❌ Don't skip sample size validation - it confirms correct implementation

---

**Confidence Score: 9/10**  
*High confidence due to:*
- *Existing cross-site infrastructure already implemented*
- *Clear methodology from R Journal 2023 paper*  
- *Straightforward configuration changes needed*
- *Comprehensive validation strategy defined*
- *Well-understood expected outcomes*