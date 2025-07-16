# Baseline Benchmark

## Overview
This module establishes baseline performance metrics for cause-of-death assignment using the PHMRC gold-standard verbal autopsy dataset. It compares classical ML models with established VA algorithms to create a comprehensive benchmark.

## Dataset
- **Source**: PHMRC (Population Health Metrics Research Consortium) VA gold standard
- **Files**:
  - `data/raw/PHMRC/IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv`
  - `data/raw/PHMRC/IHME_PHMRC_VA_DATA_CHILD_Y2013M09D11_0.csv`
  - `data/raw/PHMRC/IHME_PHMRC_VA_DATA_NEONATE_Y2013M09D11_0.csv`
  - `data/raw/PHMRC/IHME_PHMRC_VA_DATA_CODEBOOK_Y2013M09D11_0.xlsx`

## Models

### Classical ML Models
1. **XGBoost** (`models/xgboost_model.py`)
   - Gradient boosting with tree-based learners
   - Hyperparameters: max_depth, learning_rate, n_estimators
   
2. **Logistic Regression** (`models/logistic_regression_model.py`)
   - Multi-class classification with regularization
   - Hyperparameters: C (regularization), solver
   
3. **Naïve Bayes** 
   - Probabilistic classifier assuming feature independence
   - Variants: Gaussian, Multinomial, Bernoulli
   
4. **Random Forest** (`models/random_forest_model.py`)
   - Ensemble of decision trees
   - Hyperparameters: n_estimators, max_depth, min_samples_split

### VA-Specific Algorithms
1. **InSilicoVA** (`models/insilico_va_model.py`)
   - Bayesian hierarchical model
   - Uses Docker environment: `models/insilico/Dockerfile`
   - Requires OpenVA library
   
2. **InterVA**
   - Rule-based probabilistic algorithm
   - Uses Docker environment with OpenVA

## Data Preprocessing
- **Library**: https://github.com/JH-DSAI/va-data
- **Steps**:
  1. Load raw PHMRC data
  2. Standardize column names
  3. Handle missing values
  4. Create age-group categories
  5. Encode categorical variables
  6. Split by site for stratification

## Training Protocol
1. **Train/Test Split**:
   - 80/20 split
   - Stratified by site AND age-group
   - Random seed for reproducibility
   
2. **Cross-Validation**:
   - 5-fold stratified CV
   - Used for hyperparameter tuning
   - Maintain site/age-group balance

## Evaluation Metrics
1. **CSMF Accuracy** (Cause-Specific Mortality Fraction)
   - Measures accuracy of population-level cause distribution
   - Formula: 1 - sum(|true_csmf - predicted_csmf|) / 2
   
2. **Top-1 COD Accuracy**
   - Individual-level accuracy for top predicted cause
   
3. **Top-3 COD Accuracy**
   - Accuracy when true cause is in top 3 predictions
   
4. **COD5**
   - Accuracy for top 5 most common causes of death

## Implementation Steps
1. Load and preprocess PHMRC data
2. Create stratified train/test splits
3. Train each classical ML model with CV hyperparameter tuning
4. Run InSilicoVA and InterVA algorithms
5. Calculate all metrics for each model
6. Generate comparison tables and plots
7. Export results to `results/baseline/benchmark_results.csv`

## Expected Deliverables
- `benchmark_results.csv` with columns:
  - Model name
  - CSMF accuracy
  - Top-1 accuracy
  - Top-3 accuracy
  - COD5 accuracy
  - Training time
  - Inference time
  - Best hyperparameters (if applicable)

## Code Structure
```
baseline/
├── __init__.py
├── data_preprocessing.py    # PHMRC data loading and preprocessing
├── train_models.py          # Model training orchestration
├── evaluate.py              # Metric calculation
├── utils.py                 # Helper functions
└── config.py               # Configuration settings
```

## Dependencies
- scikit-learn
- xgboost
- pandas
- numpy
- openva (via Docker)
- va-data preprocessing library