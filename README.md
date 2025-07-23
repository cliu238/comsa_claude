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

The VA34 site-based model comparison experiment has been implemented to compare InSilicoVA and XGBoost performance across different VA data collection sites.

### Data Preprocessing

Before running the comparison, you must first preprocess the VA data using the baseline module:

```bash
# Generate OpenVA format data (required for InSilicoVA)
poetry run python baseline/example_usage.py
```

This creates processed data files in `results/baseline/processed_data/` with the appropriate format for both models.

### Running the Experiment

```bash
# Basic usage
poetry run python model_comparison/scripts/run_va34_comparison.py --data-path path/to/va_data.csv

# Full example with all options (Mac users need RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1)
RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1 poetry run python model_comparison/scripts/run_va34_comparison.py \
  --data-path results/baseline/processed_data/adult_openva_20250723_103018.csv \
  --sites AP Bohol Dar Mexico Pemba UP \
  --models xgboost insilico \
  --training-sizes 0.25 0.5 0.75 1.0 \
  --n-bootstrap 100 \
  --parallel \
  --n-workers 8 \
  --batch-size 50 \
  --output-dir results/full_va34_comparison_complete \
  --no-plots
```

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

### Parallel Execution with Prefect and Ray

The comparison scripts now support parallel execution using Prefect orchestration and Ray distributed computing for significant performance improvements:

```bash
# Enable parallel execution with the --parallel flag
poetry run python model_comparison/scripts/run_va34_comparison.py \
  --data-path ./results/baseline/processed_data/adult_openva_20250723_103018.csv \
  --sites AP Bohol Dar \
  --models insilico xgboost \
  --parallel \
  --n-workers 4 \
  --batch-size 50

# For Mac users, set the Ray object store environment variable
RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1 poetry run python model_comparison/scripts/run_va34_comparison.py \
  --data-path ./results/baseline/processed_data/adult_openva_20250723_103018.csv \
  --sites AP Bohol Dar Mexico UP \
  --models xgboost insilico \
  --parallel \
  --n-workers 8 \
  --checkpoint-interval 10 \
  --resume  # Resume from checkpoint if interrupted
```

**Features:**
- 50%+ performance improvement through parallel model training with Ray
- Prefect orchestration for workflow management and monitoring
- Real-time progress monitoring with tqdm
- Checkpoint/resume capability for long experiments
- Ray dashboard at http://localhost:8265 for monitoring
- Memory-efficient data sharing across workers
- Backward compatible (sequential execution by default)
- Automatic handling of data format requirements (numeric for XGBoost, "Y"/"." for InSilicoVA)

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

### Model Features Comparison

| Feature | XGBoost | Random Forest | InSilicoVA |
|---------|---------|---------------|------------|
| Training Speed | Fast | Moderate | Slow |
| Prediction Speed | Fast | Fast | Slow |
| Feature Importance | ✓ | ✓ (Better) | ✗ |
| Handles Missing Data | ✓ | ✓ | ✓ |
| Cross-site Generalization | Good | Better | Best |
| Interpretability | Medium | High | High |
| Class Imbalance Handling | Good | Excellent | Good |

All models support:
- CSMF accuracy calculation
- Cross-validation with stratification
- sklearn-compatible interface (fit, predict, predict_proba)
- Integration with the model comparison framework
## Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)