# Context Engineering Template

A comprehensive template for getting started with Context Engineering - the discipline of engineering context for AI coding assistants so they have the information necessary to get the job done end to end.

> **Context Engineering is 10x better than prompt engineering and 100x better than vibe coding.**

## 🚀 Quick Start

```bash
# 1. Clone this template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up your project rules (optional - template provided)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## 📚 Table of Contents

- [What is Context Engineering?](#what-is-context-engineering)
- [Template Structure](#template-structure)
- [Step-by-Step Guide](#step-by-step-guide)
- [Docker Setup for InSilicoVA](#docker-setup-for-insilicova)
- [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
- [The PRP Workflow](#the-prp-workflow)
- [Using Examples Effectively](#using-examples-effectively)
- [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering represents a paradigm shift from traditional prompt engineering:

### Prompt Engineering vs Context Engineering

**Prompt Engineering:**
- Focuses on clever wording and specific phrasing
- Limited to how you phrase a task
- Like giving someone a sticky note

**Context Engineering:**
- A complete system for providing comprehensive context
- Includes documentation, examples, rules, patterns, and validation
- Like writing a full screenplay with all the details

### Why Context Engineering Matters

1. **Reduces AI Failures**: Most agent failures aren't model failures - they're context failures
2. **Ensures Consistency**: AI follows your project patterns and conventions
3. **Enables Complex Features**: AI can handle multi-step implementations with proper context
4. **Self-Correcting**: Validation loops allow AI to fix its own mistakes

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates comprehensive PRPs
│   │   └── execute-prp.md     # Executes PRPs to implement features
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global rules for AI assistant
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

This template doesn't focus on RAG and tools with context engineering because I have a LOT more in store for that soon. ;)

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file contains project-wide rules that the AI assistant will follow in every conversation. The template includes:

- **Project awareness**: Reading planning docs, checking tasks
- **Code structure**: File size limits, module organization
- **Testing requirements**: Unit test patterns, coverage expectations
- **Style conventions**: Language preferences, formatting rules
- **Documentation standards**: Docstring formats, commenting practices

**You can use the provided template as-is or customize it for your project.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe what you want to build:

```markdown
## FEATURE:
[Describe what you want to build - be specific about functionality and requirements]

## EXAMPLES:
[List any example files in the examples/ folder and explain how they should be used]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or MCP server resources]

## OTHER CONSIDERATIONS:
[Mention any gotchas, specific requirements, or things AI assistants commonly miss]
```

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints that include:

- Complete context and documentation
- Implementation steps with validation
- Error handling patterns
- Test requirements

They are similar to PRDs (Product Requirements Documents) but are crafted more specifically to instruct an AI coding assistant.

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

This command will:
1. Read your feature request
2. Research the codebase for patterns
3. Search for relevant documentation
4. Create a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:
1. Read all context from the PRP
2. Create a detailed implementation plan
3. Execute each step with validation
4. Run tests and fix any issues
5. Ensure all success criteria are met

## Docker Setup for InSilicoVA

This project includes a complete Docker environment for running InSilicoVA verbal autopsy evaluations. The Docker setup is essential for reproducible research and ensures all R packages and Java dependencies are properly configured.

### 🐳 Quick Docker Setup

```bash
# 1. Build and test your Docker setup
./build-docker.sh

# 2. Build the image manually (if needed)
# For Apple Silicon (M1/M2):
docker build -t insilicova-arm64:latest --platform linux/arm64 .

# For Intel/AMD:
docker build -t insilicova-amd64:latest --platform linux/amd64 .

# 3. Run InSilicoVA evaluations
poetry run python baseline/run_ap_only_insilico.py
poetry run python baseline/run_full_insilico.py
```

### 📋 What's Included

- **Base**: Ubuntu 22.04 with R 4.4.3
- **Java 11**: Required for InSilicoVA backend
- **R Packages**: openVA, InSilicoVA, dplyr, and analysis tools
- **Validated**: Against R Journal 2023 benchmarks

### 🔧 Key Files

- `Dockerfile` - Complete Docker image definition
- `DOCKER_USAGE.md` - Detailed usage guide
- `build-docker.sh` - Automated build and testing script
- **Working SHA**: `sha256:61df64731dec9b9e188b2b5a78000dd81e63caf78957257872b9ac1ba947efa4`

### 📊 Research Validation

The Docker setup has been validated against:
- **R Journal 2023** InSilicoVA methodology
- **PHMRC dataset** (7,582 adult samples)
- **AP-only testing**: 0.695 CSMF accuracy
- **Cross-platform** compatibility (ARM64/AMD64)

For complete Docker documentation, see `DOCKER_USAGE.md`.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive
- ❌ "Build a web scraper"
- ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage the examples/ folder
- Place relevant code patterns in `examples/`
- Reference specific files and patterns to follow
- Explain what aspects should be mimicked

**DOCUMENTATION**: Include all relevant resources
- API documentation URLs
- Library guides
- MCP server documentation
- Database schemas

**OTHER CONSIDERATIONS**: Capture important details
- Authentication requirements
- Rate limits or quotas
- Common pitfalls
- Performance requirements

## The PRP Workflow

### How /generate-prp Works

The command follows this process:

1. **Research Phase**
   - Analyzes your codebase for patterns
   - Searches for similar implementations
   - Identifies conventions to follow

2. **Documentation Gathering**
   - Fetches relevant API docs
   - Includes library documentation
   - Adds gotchas and quirks

3. **Blueprint Creation**
   - Creates step-by-step implementation plan
   - Includes validation gates
   - Adds test requirements

4. **Quality Check**
   - Scores confidence level (1-10)
   - Ensures all context is included

### How /execute-prp Works

1. **Load Context**: Reads the entire PRP
2. **Plan**: Creates detailed task list using TodoWrite
3. **Execute**: Implements each component
4. **Validate**: Runs tests and linting
5. **Iterate**: Fixes any issues found
6. **Complete**: Ensures all requirements met

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Using Examples Effectively

The `examples/` folder is **critical** for success. AI coding assistants perform much better when they can see patterns to follow.

### What to Include in Examples

1. **Code Structure Patterns**
   - How you organize modules
   - Import conventions
   - Class/function patterns

2. **Testing Patterns**
   - Test file structure
   - Mocking approaches
   - Assertion styles

3. **Integration Patterns**
   - API client implementations
   - Database connections
   - Authentication flows

4. **CLI Patterns**
   - Argument parsing
   - Output formatting
   - Error handling

### Example Structure

```
examples/
├── README.md           # Explains what each example demonstrates
├── cli.py             # CLI implementation pattern
├── agent/             # Agent architecture patterns
│   ├── agent.py      # Agent creation pattern
│   ├── tools.py      # Tool implementation pattern
│   └── providers.py  # Multi-provider pattern
└── tests/            # Testing patterns
    ├── test_agent.py # Unit test patterns
    └── conftest.py   # Pytest configuration
```

## Best Practices

### 1. Be Explicit in INITIAL.md
- Don't assume the AI knows your preferences
- Include specific requirements and constraints
- Reference examples liberally

### 2. Provide Comprehensive Examples
- More examples = better implementations
- Show both what to do AND what not to do
- Include error handling patterns

### 3. Use Validation Gates
- PRPs include test commands that must pass
- AI will iterate until all validations succeed
- This ensures working code on first try

### 4. Leverage Documentation
- Include official API docs
- Add MCP server resources
- Reference specific documentation sections

### 5. Customize CLAUDE.md
- Add your conventions
- Include project-specific rules
- Define coding standards

## VA Data Processing Pipeline

### Overview

The `baseline` module provides a standardized pipeline for processing PHMRC (Population Health Metrics Research Consortium) Verbal Autopsy data. It supports two output formats:

1. **Numeric format** - For standard ML algorithms (scikit-learn, XGBoost, etc.)
2. **OpenVA format** - For InSilicoVA R package compatibility

### Installation

```bash
# 1. Initialize the va-data submodule
git submodule update --init --recursive

# 2. Install dependencies with Poetry
poetry install
```

### Usage

Run the example script to process PHMRC adult data:

```bash
poetry run python baseline/example_usage.py
```

This will:
- Load and validate PHMRC data using the PHMRCData class
- Apply OpenVA transformations
- Generate both numeric and OpenVA encoded outputs
- Save results to `results/baseline/processed_data/`

### Configuration

The pipeline uses a Pydantic-based configuration system:

```python
from baseline.config.data_config import DataConfig

config = DataConfig(
    data_path="data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv",
    output_dir="results/baseline/",
    openva_encoding=False,  # True for InSilicoVA, False for ML
    drop_columns=[],        # Optional columns to exclude
    stratify_by_site=True   # Enable site-based stratification
)
```

### Output Files

The pipeline generates two files per processing run in `results/baseline/processed_data/`:

1. **CSV Data Files**
   - `adult_numeric_YYYYMMDD_HHMMSS.csv` - Numeric encoding for ML (0/1 values)
   - `adult_openva_YYYYMMDD_HHMMSS.csv` - OpenVA encoding for InSilicoVA ("Y"/""/".") 

2. **Metadata JSON Files**
   - Contains processing configuration, timestamp, data shape, column names, and cause-of-death distribution
   - Example: `adult_numeric_20250717_163737.metadata.json`

### Testing

Run unit tests with coverage:

```bash
poetry run pytest tests/baseline/ -v --cov=baseline
```

The core modules achieve >96% test coverage.

## Updated Project Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates comprehensive PRPs
│   │   └── execute-prp.md     # Executes PRPs to implement features
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   ├── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
│   └── va_data_processing_pipeline.md  # VA pipeline PRP
├── baseline/                  # VA data processing module
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── data_config.py    # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader_preprocessor.py  # Core processing logic
│   ├── models/               # ML model implementations
│   │   ├── __init__.py
│   │   ├── insilico_model.py      # InSilicoVA wrapper
│   │   ├── xgboost_model.py       # XGBoost implementation
│   │   ├── xgboost_config.py      # XGBoost configuration
│   │   ├── random_forest_model.py  # Random Forest implementation
│   │   └── random_forest_config.py # Random Forest configuration
│   └── example_usage.py      # Usage demonstration
├── examples/                  # Your code examples
│   └── data_validation.py    # Reference VA data processing
├── tests/
│   ├── __init__.py
│   └── baseline/
│       ├── __init__.py
│       └── test_data_loader.py  # Unit tests
├── results/
│   └── baseline/
│       ├── processed_data/   # Output CSV and metadata files
│       └── logs/            # Processing logs
├── va-data/                 # Git submodule for VA data utilities
├── conftest.py             # Pytest configuration
├── pyproject.toml          # Poetry dependencies
├── CLAUDE.md               # Global rules for AI assistant
├── INITIAL.md              # Template for feature requests
└── README.md               # This file
```

## VA34 Model Comparison Results

The VA34 site-based model comparison has been fully integrated into the distributed comparison framework. All model comparisons now use the unified `run_distributed_comparison.py` script which supports parallel execution with Ray and Prefect orchestration.

### Available Models

- **InSilicoVA**: Bayesian probabilistic model with epidemiological priors
- **XGBoost**: Gradient boosting model with excellent accuracy
- **Random Forest**: Ensemble model with robust feature importance analysis
- **Logistic Regression**: Fast linear model with L1/L2 regularization options
- **CategoricalNB**: Naive Bayes model optimized for categorical VA data with robust handling of missing values
- **TabICL**: Foundation model using in-context learning for tabular data (best for few-shot scenarios)

### Latest Performance Results

Based on the latest implementation with all fixes applied:

**Model Performance (CSMF Accuracy)**:
- **XGBoost**: 76.5% ± 10.9% (best overall performance)
- **Logistic Regression**: 75.9% ± 9.3% (excellent performance with fast training)
- **Random Forest**: 67.9% ± 11.5% (good balance of speed and accuracy)
- **InSilico**: 62.6% ± 17.7% (best cross-site generalization despite lower average)
- **CategoricalNB**: 55.4% ± 12.1% (fast training, handles categorical features natively)

### Key Findings (Real VA Data)

After fixing data leakage issues and data format compatibility, the experiment shows realistic performance:

**XGBoost Performance:**
- **In-domain CSMF accuracy**: 81.5% (training and testing on same site)
- **Out-domain CSMF accuracy**: 43.8% (training on one site, testing on another)
- **Generalization gap**: 37.7% performance drop
- **Training size impact**: Performance improves from 71.6% (25% data) to 81.8% (100% data)

**InSilicoVA Performance:**
- **In-domain CSMF accuracy**: 80.0% (training and testing on same site)
- **Out-domain CSMF accuracy**: 46.1% (training on one site, testing on another)
- **Generalization gap**: 33.9% performance drop (better generalization than XGBoost)
- **Training size impact**: Performance improves from 74.3% (25% data) to 79.7% (100% data)

**Key observations:**
- InSilicoVA shows better cross-site generalization despite similar in-domain performance
- Both models benefit from more training data
- Significant variation in cross-site performance depending on site pairs

### Output Files

Results are saved to the specified output directory:
- `full_results.csv` - All experimental results with confidence intervals
- `in_domain_results.csv` - Same-site train/test results
- `out_domain_results.csv` - Cross-site train/test results  
- `training_size_results.csv` - Impact of training data size
- `summary_statistics.csv` - Aggregated statistics
- Visualization plots (if --no-plots not specified)


## Available ML Models

The baseline module provides several machine learning models for VA cause-of-death prediction, all following a consistent sklearn-compatible interface:

### XGBoost Model
High-performance gradient boosting model with excellent accuracy:
```python
from baseline.models import XGBoostModel, XGBoostConfig

# Initialize with custom configuration
config = XGBoostConfig(n_estimators=200, max_depth=8, learning_rate=0.1)
model = XGBoostModel(config=config)

# Train and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance("gain")
```

### Random Forest Model
Robust ensemble model with built-in feature importance analysis:
```python
from baseline.models import RandomForestModel, RandomForestConfig

# Initialize with balanced class weights
config = RandomForestConfig(
    n_estimators=100,
    class_weight="balanced",  # Handle imbalanced VA data
    max_features="sqrt"       # Optimal for high-dimensional data
)
model = RandomForestModel(config=config)

# Train and evaluate
model.fit(X_train, y_train)
csmf_accuracy = model.calculate_csmf_accuracy(y_test, model.predict(X_test))

# Get feature importance (MDI or permutation)
mdi_importance = model.get_feature_importance("mdi")
perm_importance = model.get_feature_importance("permutation", X_test, y_test)

# Cross-validation
cv_results = model.cross_validate(X_train, y_train, cv=5)
print(f"CSMF accuracy: {cv_results['csmf_accuracy_mean']:.3f} ± {cv_results['csmf_accuracy_std']:.3f}")
```

### InSilicoVA Model
Bayesian probabilistic model with epidemiological priors:
```python
from baseline.models import InSilicoVAModel

# Requires OpenVA format data and Docker
model = InSilicoVAModel()
model.fit(X_train_openva, y_train)
predictions = model.predict(X_test_openva)
```

### Logistic Regression Model
Fast, interpretable linear model with regularization options:
```python
from baseline.models import LogisticRegressionModel, LogisticRegressionConfig

# Initialize with L1 regularization for feature selection
config = LogisticRegressionConfig(
    penalty="l1",
    solver="saga",  # Supports all penalty types
    C=0.1,          # Stronger regularization
    class_weight="balanced"
)
model = LogisticRegressionModel(config=config)

# Train and evaluate
model.fit(X_train, y_train)
csmf_accuracy = model.calculate_csmf_accuracy(y_test, model.predict(X_test))

# Get coefficient-based feature importance
importance = model.get_feature_importance()
print(f"Top 5 features: {importance.head()}")

# L1 regularization creates sparse models
zero_features = (importance['importance'] == 0).sum()
print(f"Features eliminated by L1: {zero_features}")
```

### CategoricalNB Model
Naive Bayes model specifically designed for categorical VA data:
```python
from baseline.models import CategoricalNBModel, CategoricalNBConfig

# Initialize with optimized parameters for VA data
config = CategoricalNBConfig(
    alpha=1.0,          # Laplace smoothing parameter
    fit_prior=True,     # Learn class priors from data
    force_alpha=True    # Apply same smoothing to all features
)
model = CategoricalNBModel(config=config)

# Train and predict - handles Y/N/. encoding automatically
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate CSMF accuracy
csmf_accuracy = model.calculate_csmf_accuracy(y_test, predictions)

# Get feature importance based on log probability ratios
importance = model.get_feature_importance()
print(f"Most discriminative features:\n{importance.head(10)}")

# Cross-validation with stratification
cv_results = model.cross_validate(X_train, y_train, cv=5)
print(f"CV CSMF: {cv_results['csmf_accuracy_mean']:.3f} ± {cv_results['csmf_accuracy_std']:.3f}")
```

### TabICL Model
Foundation model using in-context learning for tabular data. TabICL leverages pre-trained knowledge and excels in few-shot scenarios (<100 samples):

```python
from baseline.models import TabICLModel, TabICLConfig

# Initialize with optimized configuration for 34 VA classes
config = TabICLConfig(
    n_estimators=48,        # Optimal for 34 classes
    softmax_temperature=0.5, # Lower for sharper predictions
    use_hierarchical=True,   # Essential for >10 classes
    batch_size=4,           # Reduced for memory efficiency
    offload_to_cpu=True     # Handle memory constraints
)
model = TabICLModel(config=config)

# Train and predict - automatically handles 1-34 label encoding
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate CSMF accuracy
csmf_accuracy = model.calculate_csmf_accuracy(y_test, predictions)
```

#### Comparing TabICL with XGBoost

**Full comparison with both models:**
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites AP Mexico \
    --models xgboost tabicl \
    --training-sizes 1.0 \
    --n-bootstrap 10 \
    --n-workers 4 \
    --output-dir results/xgboost_tabicl_comparison
```

**Few-shot comparison (where TabICL should excel):**
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites AP Mexico \
    --models xgboost tabicl \
    --training-sizes 0.05 0.1 0.5 1.0 \
    --n-bootstrap 5 \
    --n-workers 4 \
    --output-dir results/fewshot_comparison
```

**Quick test (minimal bootstrap for testing):**
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites AP \
    --models xgboost tabicl \
    --training-sizes 1.0 \
    --n-bootstrap 2 \
    --n-workers 2 \
    --output-dir results/quick_test
```

#### TabICL Performance Notes

**Key Findings:**
- **Speed**: TabICL is 200-300x slower than XGBoost due to its ensemble architecture
- **Accuracy**: On large datasets (1000+ samples), XGBoost outperforms TabICL by 7-9% CSMF accuracy
- **Sweet Spot**: TabICL excels with <100 training samples where pre-trained knowledge provides advantages
- **Memory**: Requires 4GB+ RAM; uses CPU offloading for memory efficiency

**When to Use TabICL:**
- ✅ New deployment sites with minimal local data (<100 samples)
- ✅ Rapid prototyping without hyperparameter tuning
- ✅ Transfer learning from similar medical datasets
- ✅ Research on few-shot learning capabilities

**When to Use XGBoost:**
- ✅ Production VA pipeline with 1000+ samples
- ✅ Speed is critical (real-time processing)
- ✅ Full hyperparameter optimization possible
- ✅ Resource-constrained environments

**Configuration Details:**
TabICL has been configured to handle all 34 VA cause-of-death classes using:
- Hierarchical classification mode (`use_hierarchical=True`)
- Optimized ensemble size (`n_estimators=48`)
- Memory-efficient batch processing (`batch_size=4`)
- Automatic label encoding for 1-34 class labels

### Model Features Comparison

| Feature | XGBoost | Random Forest | Logistic Regression | CategoricalNB | InSilicoVA |
|---------|---------|---------------|-------------------|---------------|------------|
| Training Speed | Fast | Moderate | Very Fast | Very Fast | Slow |
| Prediction Speed | Fast | Fast | Very Fast | Very Fast | Slow |
| Feature Importance | ✓ | ✓ (Better) | ✓ (Coefficients) | ✓ (Log ratios) | ✗ |
| Handles Missing Data | ✓ | ✓ | ✓ | ✓ (Native) | ✓ |
| Cross-site Generalization | Good | Better | Moderate | Moderate | Best |
| Interpretability | Medium | High | Very High | High | High |
| Class Imbalance Handling | Good | Excellent | Good | Moderate | Good |
| Regularization | ✗ | ✗ | L1/L2/ElasticNet | Alpha smoothing | ✗ |
| Feature Selection | ✗ | ✗ | ✓ (L1) | ✗ | ✗ |
| Categorical Features | Encoded | Encoded | Encoded | Native | Native |

All models support:
- CSMF accuracy calculation
- Cross-validation with stratification
- sklearn-compatible interface (fit, predict, predict_proba)
- Integration with the model comparison framework

## Ensemble Model Experiments

Run ensemble experiments using the unified distributed comparison framework. Ensembles are now integrated into `run_distributed_comparison.py` with comprehensive configuration options.

### Usage

```bash
# Run ensemble experiments with soft voting and multiple sizes
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP \
    --models ensemble \
    --ensemble-voting-strategies soft \
    --ensemble-weight-strategies none performance \
    --ensemble-sizes 3 5 \
    --ensemble-base-models all \
    --ensemble-combination-strategy smart \
    --training-sizes 0.7 \
    --n-bootstrap 30 \
    --output-dir results/ensemble_with_names-v2
```

### Ensemble-Specific Parameters

- `--ensemble-voting-strategies`: Voting methods (soft, hard)
- `--ensemble-weight-strategies`: Weight assignment (none, performance)
- `--ensemble-sizes`: Number of models in ensemble (3, 5, 7)
- `--ensemble-base-models`: Base models to include (all, or specific list)
- `--ensemble-combination-strategy`: How to select model combinations (smart, exhaustive)

### Key Findings

Based on comprehensive analysis comparing ensembles with individual models:

**Performance Results (CSMF Accuracy)**:
- **XGBoost (Individual)**: 74.8% ± 9.2% - **Best overall performer**
- **5-Model Ensemble**: 67.0% ± 13.1% - Best ensemble, but underperforms XGBoost by 10.4%
- **3-Model Ensemble**: 59.7% ± 15.0% - Significant underperformance

**Head-to-Head Win Rates** (Ensemble vs Individual Models):
- vs XGBoost: 11-44% win rate (ensembles consistently lose)
- vs Random Forest: 33-44% win rate (mixed results)
- vs Weak models (Categorical NB, Logistic): 67-89% win rate (ensembles usually win)

**Recommendation**: Individual models (especially XGBoost) outperform ensembles while requiring 3-5x less computational resources. Focus on optimizing individual models rather than ensembling.

## Distributed Model Comparison

The project includes a distributed model comparison framework using Ray and Prefect for running large-scale experiments across multiple models and sites. This unified framework replaces the previous separate scripts and provides parallel execution capabilities for all model types.

### Quick Start Commands

#### System Requirements Note
The `--n-workers` parameter should be adjusted based on your system:
- **2-4 cores**: Use `--n-workers 2`
- **8 cores**: Use `--n-workers 4-6`
- **16+ cores**: Use `--n-workers 8-14` (leave some cores for system tasks)

For a quick test with fewer experiments:
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
   --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
   --sites Mexico AP \
   --models xgboost insilico \
   --n-workers 8 \
   --training-sizes 1.0 \
   --n-bootstrap 10 \
   --enable-tuning \
   --tuning-algorithm bayesian \
   --tuning-trials 3 \
   --tuning-cv-folds 3 \
   --memory-per-worker 4GB \
   --track-component-times \
   --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S) \
   --no-plots
```

For all sites with maximum parallelization:
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP Dar Bohol Pemba \
    --models logistic_regression random_forest xgboost categorical_nb insilico \
    --n-workers 4 \
    --training-sizes 1.0 \
    --n-bootstrap 10 \
    --batch-size 10 \
    --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S) \
    --no-plots
```

Full comprehensive run with hyperparameter tuning enabled:
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites AP Bohol Dar Mexico Pemba UP \
    --models xgboost random_forest logistic_regression categorical_nb insilico \
    --n-bootstrap 100 \
    --enable-tuning \
    --tuning-trials 100 \
    --tuning-algorithm bayesian \
    --tuning-metric csmf_accuracy \
    --tuning-cv-folds 5 \
    --tuning-cpus-per-trial 1.0 \
    --n-workers -1 \
    --memory-per-worker 4GB \
    --batch-size 50 \
    --checkpoint-interval 10 \
    --ray-dashboard-port 8265 \
    --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S) \
    --track-component-times \
    --random-seed 42
```

### Command Line Options

- `--data-path`: Path to the VA data CSV file (required)
- `--sites`: List of sites to include in comparison (e.g., Mexico AP UP Dar Bohol Pemba)
- `--models`: Models to compare (choices: logistic_regression, random_forest, xgboost, insilico)
- `--n-workers`: Number of Ray workers for parallel execution (-1 for auto)
- `--training-sizes`: Training data fractions for size experiments (default: 0.25 0.5 0.75 1.0)
- `--n-bootstrap`: Number of bootstrap iterations for confidence intervals (default: 100)
- `--output-dir`: Directory to save results (default: results/distributed_comparison)
- `--no-plots`: Skip generating visualization plots
- `--resume`: Resume from checkpoint if available
- `--clear-checkpoints`: Clear existing checkpoints before starting
- `--track-component-times`: Track separate timing for tuning, training, and inference components

### Execution Time Considerations

**⚠️ Important:** InSilico models can take 30-60 minutes per experiment. With 100 bootstrap iterations and multiple sites/training sizes, the full comparison may take several hours.

**Recommendations:**
1. **Use terminal or screen/tmux**: Claude Code has a 10-minute timeout limit. For long-running experiments, run directly in your terminal:
   ```bash
   # Using screen
   screen -S va_comparison
   poetry run python model_comparison/scripts/run_distributed_comparison.py [options]
   # Detach with Ctrl+A, D
   
   # Using tmux
   tmux new -s va_comparison
   poetry run python model_comparison/scripts/run_distributed_comparison.py [options]
   # Detach with Ctrl+B, D
   ```

2. **Use checkpoints**: The script automatically saves checkpoints every 10 experiments. Use `--resume` to continue from where you left off:
   ```bash
   poetry run python model_comparison/scripts/run_distributed_comparison.py --resume [other options]
   ```

3. **Start with quick tests**: Run with fewer bootstrap iterations (10-20) and fewer training sizes to verify everything works before running the full evaluation.

### Output Files

Results are saved to the specified output directory:
- `va34_comparison_results.csv` - Detailed results for each experiment including:
  - Model performance metrics (CSMF accuracy, COD accuracy)
  - Experiment metadata (sites, training size, execution time)
  - Error messages for failed experiments
- `checkpoints/` - Checkpoint files for resuming interrupted runs

### Model Performance Summary

Based on the implemented fixes, expected performance ranges:
- **Random Forest**: 78-82% CSMF accuracy (in-domain)
- **XGBoost**: 60-74% CSMF accuracy (in-domain)
- **Logistic Regression**: 65-70% CSMF accuracy (in-domain)
- **InSilico**: 70-74% CSMF accuracy (cross-site validation)

### Tracking Computational Performance

The `--track-component-times` flag enables detailed timing breakdown to understand where computational time is spent:

```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP \
    --models logistic_regression xgboost \
    --n-workers 8 \
    --enable-tuning \
    --tuning-trials 10 \
    --track-component-times \
    --output-dir results/timing_analysis
```

When enabled, the output CSV will include additional columns:
- `tuning_time_seconds`: Time spent on hyperparameter optimization (if tuning enabled)
- `training_time_seconds`: Time to fit the model with final parameters
- `inference_time_seconds`: Time for predictions and metric calculation
- `execution_time_seconds`: Total time (maintained for backward compatibility)

**Notes on timing comparisons:**
- **InSilicoVA**: No tuning time (deterministic model), only training and inference
- **XGBoost/Random Forest/Logistic Regression**: Include tuning time when `--enable-tuning` is used
- **Fair comparison**: Compare inference times for deployment scenarios, as tuning is a one-time cost
- **Tuning overhead**: With Bayesian optimization (10 trials × 5-fold CV = 50 model fits), tuning can dominate total time

### Troubleshooting

1. **Memory issues on macOS**: The script automatically limits Ray's object store to 2GB on macOS. If you still encounter issues, reduce `--n-workers`.

2. **InSilico Docker errors**: Ensure Docker is running and the InSilico image is built:
   ```bash
   ./build-docker.sh
   ```

3. **Missing va_data module**: The script automatically adds the va-data directory to Python path. Ensure the submodule is initialized:
   ```bash
   git submodule update --init --recursive
   ```

4. **NaN values in data**: The preprocessing now handles NaN values automatically by converting them to a special category for categorical features and -1 for numeric features.

5. **CategoricalNB index errors**: The model now handles varying numbers of categories per feature between training and test sets by capping unseen categories to the maximum seen during training.

## Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)


For COD5 VS VA34
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP Dar Bohol Pemba \
    --label-type cod5 \
    --models logistic_regression random_forest xgboost categorical_nb insilico \
    --n-workers 4 \
    --training-sizes 1.0 \
    --n-bootstrap 10 \
    --batch-size 10 \
    --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S)_cod5 \
    --no-plots
```

```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP Dar Bohol Pemba \
    --label-type va34 \
    --models logistic_regression random_forest xgboost categorical_nb insilico \
    --n-workers 4 \
    --training-sizes 1.0 \
    --n-bootstrap 10 \
    --batch-size 10 \
    --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S)_va34 \
    --no-plots
```


For COD5 VS VA34
```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP Dar Bohol Pemba \
    --label-type cod5 \
    --models logistic_regression random_forest xgboost categorical_nb insilico \
    --n-workers 4 \
    --training-sizes 1.0 \
    --n-bootstrap 10 \
    --batch-size 10 \
    --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S)_cod5 \
    --no-plots
```

```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP Dar Bohol Pemba \
    --label-type va34 \
    --models logistic_regression random_forest xgboost categorical_nb insilico \
    --n-workers 4 \
    --training-sizes 1.0 \
    --n-bootstrap 10 \
    --batch-size 10 \
    --output-dir results/full_comparison_$(date +%Y%m%d_%H%M%S)_va34 \
    --no-plots
```