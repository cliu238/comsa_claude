# Verbal Autopsy Model Comparison Framework

A comprehensive framework for comparing machine learning models on verbal autopsy (VA) cause-of-death prediction tasks using the PHMRC dataset.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-repo/va-model-comparison.git
cd va-model-comparison
git submodule update --init --recursive

# 2. Install dependencies
poetry install

# 3. Run a quick comparison
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP \
    --models xgboost insilico \
    --training-sizes 1.0 \
    --n-workers 4 \
    --output-dir results/quick_test
```

## 📊 Available Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| **XGBoost** | Fast | High (76.5%) | Production pipelines |
| **Random Forest** | Fast | Good (67.9%) | Feature importance analysis |
| **Logistic Regression** | Very Fast | High (75.9%) | Interpretable baselines |
| **CategoricalNB** | Very Fast | Moderate (55.4%) | Quick prototypes |
| **InSilicoVA** | Slow | Good (62.6%) | Cross-site generalization |
| **TabICL** | Very Slow | Varies | Few-shot scenarios (<100 samples) |

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Docker (for InSilicoVA model)
- Poetry for dependency management

### Setup Steps

```bash
# Clone repository with submodules
git clone https://github.com/your-repo/va-model-comparison.git
cd va-model-comparison
git submodule update --init --recursive

# Install Python dependencies
poetry install

# Build Docker image for InSilicoVA (optional)
./build-docker.sh
```

## 💻 Running Experiments

### Basic Model Comparison

Compare models across different sites with multiple random seeds for robustness:

```bash
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --data-path va-data/data/phmrc/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv \
    --sites Mexico AP UP Dar Bohol Pemba \
    --models xgboost random_forest logistic_regression \
    --random-seeds 42 123 456 \
    --training-sizes 0.5 1.0 \
    --n-workers 4 \
    --output-dir results/comparison
```

### Command Line Options

**Required:**
- `--data-path`: Path to VA data CSV file
- `--sites`: Sites to include (e.g., Mexico AP UP Dar Bohol Pemba)

**Model Selection:**
- `--models`: Models to compare (xgboost, random_forest, logistic_regression, categorical_nb, insilico, tabicl)
- `--label-type`: Label system - va34 (34 causes) or cod5 (5 categories), default: va34

**Experiment Configuration:**
- `--training-sizes`: Training data fractions (default: 0.25 0.5 0.75 1.0)
- `--random-seeds`: Seeds for multiple runs (default: 42)

**Parallelization:**
- `--n-workers`: Number of parallel workers (-1 for auto)
- `--memory-per-worker`: Memory per worker (default: 4GB)
- `--batch-size`: Experiments per batch (default: 50)

**Output:**
- `--output-dir`: Results directory (default: results/distributed)
- `--no-plots`: Skip visualization generation

### Advanced Features

For hyperparameter tuning, ensemble experiments, and bootstrap confidence intervals, use the advanced script:

```bash
poetry run python model_comparison/scripts/run_distributed_comparison_advanced.py \
    --data-path data.csv \
    --sites Mexico AP \
    --enable-tuning \
    --enable-ensemble-exploration \
    --n-bootstrap 100
```

## 📁 Data Processing Pipeline

The framework includes a complete VA data processing pipeline:

```python
from baseline.data import DataLoaderPreprocessor
from baseline.config import DataConfig

# Configure data processing
config = DataConfig(
    data_path="path/to/phmrc_data.csv",
    output_dir="results/",
    openva_encoding=False,  # True for InSilicoVA
    stratify_by_site=True
)

# Process data
processor = DataLoaderPreprocessor(config)
X_train, X_test, y_train, y_test = processor.load_and_split_data()
```

### Output Formats

The pipeline generates two encoding formats:
- **Numeric format** (`adult_numeric_*.csv`): For ML models (0/1 values)
- **OpenVA format** (`adult_openva_*.csv`): For InSilicoVA ("Y"/""/".") 

## 🐳 Docker Setup for InSilicoVA

InSilicoVA requires R and Java dependencies. Use the provided Docker setup:

```bash
# Build the Docker image
./build-docker.sh

# The image is automatically used by the InSilicoVA model
poetry run python baseline/run_full_insilico.py
```

### Docker Contents
- Ubuntu 22.04 with R 4.4.3
- Java 11 (required for InSilicoVA)
- R packages: openVA, InSilicoVA, dplyr
- Validated against R Journal 2023 benchmarks

## 📈 Model Usage Examples

### XGBoost
```python
from baseline.models import XGBoostModel, XGBoostConfig

config = XGBoostConfig(n_estimators=200, max_depth=8)
model = XGBoostModel(config=config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Random Forest
```python
from baseline.models import RandomForestModel, RandomForestConfig

config = RandomForestConfig(n_estimators=100, class_weight="balanced")
model = RandomForestModel(config=config)
model.fit(X_train, y_train)
importance = model.get_feature_importance("mdi")
```

### InSilicoVA (requires Docker)
```python
from baseline.models import InSilicoVAModel

model = InSilicoVAModel()
model.fit(X_train_openva, y_train)
predictions = model.predict(X_test_openva)
```

## 🎯 Performance Results

Based on PHMRC adult dataset (7,582 samples across 6 sites):

### In-Domain Performance (same site train/test)
- **XGBoost**: 81.5% CSMF accuracy
- **Logistic Regression**: 80.2% CSMF accuracy  
- **Random Forest**: 78.5% CSMF accuracy
- **InSilicoVA**: 80.0% CSMF accuracy

### Cross-Site Generalization
- **InSilicoVA**: 46.1% (best generalization)
- **XGBoost**: 43.8%
- **Logistic Regression**: 41.2%
- **Random Forest**: 40.5%

## 📊 Output Files

Results are saved to the specified output directory:
- `va34_comparison_results.csv` - Detailed results for each experiment
- `checkpoints/` - Resume capability for interrupted runs
- Visualization plots (unless --no-plots specified)

Each result includes:
- Model performance metrics (CSMF accuracy, COD accuracy)
- Experiment metadata (sites, training size, random seed)
- Execution time per experiment

## ⚡ Performance Tips

### System Requirements
- **2-4 cores**: Use `--n-workers 2`
- **8 cores**: Use `--n-workers 4-6`
- **16+ cores**: Use `--n-workers 8-14`

### Long Running Experiments

For experiments that may take hours:

```bash
# Use screen or tmux
screen -S va_experiment
poetry run python model_comparison/scripts/run_distributed_comparison.py [options]
# Detach with Ctrl+A, D

# Resume from checkpoint if interrupted
poetry run python model_comparison/scripts/run_distributed_comparison.py --resume [options]
```

### Memory Management

On macOS, Ray's object store is automatically limited to 2GB. If you encounter memory issues:
- Reduce `--n-workers`
- Decrease `--batch-size`
- Use `--memory-per-worker 2GB`

## 🧪 Testing

Run the test suite:

```bash
# All tests
poetry run pytest

# Specific module tests
poetry run pytest tests/baseline/ -v

# With coverage
poetry run pytest --cov=baseline --cov-report=html
```

## 📝 Project Structure

```
va-model-comparison/
├── baseline/                  # Core VA processing pipeline
│   ├── config/               # Configuration management
│   ├── data/                 # Data loading and preprocessing
│   └── models/               # Model implementations
├── model_comparison/         # Distributed comparison framework
│   ├── experiments/          # Experiment configurations
│   ├── metrics/              # CSMF and COD metrics
│   ├── orchestration/        # Ray/Prefect orchestration
│   └── scripts/              # CLI scripts
├── va-data/                  # VA data utilities (git submodule)
├── tests/                    # Unit tests
├── results/                  # Output directory
└── pyproject.toml           # Poetry dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `poetry run pytest` to ensure tests pass
5. Submit a pull request

## 📚 References

- [PHMRC Dataset](https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-9-27)
- [InSilicoVA R Package](https://cran.r-project.org/package=InSilicoVA)
- [openVA Project](https://github.com/verbal-autopsy-software/openVA)

## 📄 License

MIT License - see LICENSE file for details