# Detailed Ensemble Combination Analysis Report

Generated: 2025-08-05 11:51:13

## Executive Summary

This analysis separates different 3-model and 5-model ensemble combinations to provide detailed insights into which specific model combinations work best.

## Model Combination Performance Rankings

| Rank | Combination | CSMF Accuracy | COD Accuracy | Experiments |
|------|-------------|---------------|--------------|-------------|
| 1 | XGBoost + Random Forest + InSilicoVA + Categorical NB + Logistic Regression | 0.6703 ± 0.1340 | 0.3316 ± 0.0958 | 18 |
| 2 | XGBoost + Random Forest + Categorical NB + Logistic Regression + InSilicoVA | 0.6703 ± 0.1340 | 0.3316 ± 0.0958 | 18 |
| 3 | XGBoost + Categorical NB + InSilicoVA + Random Forest + Logistic Regression | 0.6703 ± 0.1340 | 0.3316 ± 0.0958 | 18 |
| 4 | XGBoost + Categorical NB + InSilicoVA | 0.6427 ± 0.1541 | 0.2959 ± 0.1019 | 18 |
| 5 | XGBoost + Random Forest + Categorical NB | 0.5516 ± 0.1354 | 0.2450 ± 0.1009 | 18 |

## Detailed Combination Analysis

### XGBoost + Random Forest + InSilicoVA + Categorical NB + Logistic Regression (xgb_rf_ins_cnb_lr)

**Performance:**
- CSMF Accuracy: 0.6703 ± 0.1340
- COD Accuracy: 0.3316 ± 0.0958
- Number of experiments: 18

**Head-to-Head vs Individual Models:**

| Individual Model | Win Rate | Mean CSMF Difference | Significance |
|------------------|----------|---------------------|-------------|
| insilico | 83.3% | +0.0697 | ✓ |
| categorical_nb | 88.9% | +0.1989 | ✓ |
| xgboost | 38.9% | -0.0652 | ✓ |
| random_forest | 55.6% | +0.0109 |  |
| logistic_regression | 55.6% | +0.0977 |  |

### XGBoost + Random Forest + Categorical NB + Logistic Regression + InSilicoVA (xgb_rf_cnb_lr_ins)

**Performance:**
- CSMF Accuracy: 0.6703 ± 0.1340
- COD Accuracy: 0.3316 ± 0.0958
- Number of experiments: 18

**Head-to-Head vs Individual Models:**

| Individual Model | Win Rate | Mean CSMF Difference | Significance |
|------------------|----------|---------------------|-------------|
| insilico | 83.3% | +0.0697 | ✓ |
| categorical_nb | 88.9% | +0.1989 | ✓ |
| xgboost | 38.9% | -0.0652 | ✓ |
| random_forest | 55.6% | +0.0109 |  |
| logistic_regression | 55.6% | +0.0977 |  |

### XGBoost + Categorical NB + InSilicoVA + Random Forest + Logistic Regression (xgb_cnb_ins_rf_lr)

**Performance:**
- CSMF Accuracy: 0.6703 ± 0.1340
- COD Accuracy: 0.3316 ± 0.0958
- Number of experiments: 18

**Head-to-Head vs Individual Models:**

| Individual Model | Win Rate | Mean CSMF Difference | Significance |
|------------------|----------|---------------------|-------------|
| insilico | 83.3% | +0.0697 | ✓ |
| categorical_nb | 88.9% | +0.1989 | ✓ |
| xgboost | 38.9% | -0.0652 | ✓ |
| random_forest | 55.6% | +0.0109 |  |
| logistic_regression | 55.6% | +0.0977 |  |

### XGBoost + Categorical NB + InSilicoVA (xgb_cnb_ins)

**Performance:**
- CSMF Accuracy: 0.6427 ± 0.1541
- COD Accuracy: 0.2959 ± 0.1019
- Number of experiments: 18

**Head-to-Head vs Individual Models:**

| Individual Model | Win Rate | Mean CSMF Difference | Significance |
|------------------|----------|---------------------|-------------|
| insilico | 72.2% | +0.0421 | ✓ |
| categorical_nb | 88.9% | +0.1713 | ✓ |
| xgboost | 38.9% | -0.0928 | ✓ |
| random_forest | 50.0% | -0.0167 |  |
| logistic_regression | 55.6% | +0.0701 |  |

### XGBoost + Random Forest + Categorical NB (xgb_rf_cnb)

**Performance:**
- CSMF Accuracy: 0.5516 ± 0.1354
- COD Accuracy: 0.2450 ± 0.1009
- Number of experiments: 18

**Head-to-Head vs Individual Models:**

| Individual Model | Win Rate | Mean CSMF Difference | Significance |
|------------------|----------|---------------------|-------------|
| insilico | 38.9% | -0.0491 |  |
| categorical_nb | 66.7% | +0.0801 | ✓ |
| xgboost | 11.1% | -0.1839 | ✓ |
| random_forest | 22.2% | -0.1079 | ✓ |
| logistic_regression | 33.3% | -0.0211 |  |

## Key Insights

### Best Performing Combination
**XGBoost + Random Forest + InSilicoVA + Categorical NB + Logistic Regression** achieves the highest CSMF accuracy at 0.6703 ± 0.1340

### Model Synergies
Analysis of which model combinations create synergistic effects:

**High-Performing Combinations:**
- XGBoost + Random Forest + InSilicoVA + Categorical NB + Logistic Regression: 0.6703 CSMF accuracy

**Common Patterns in High-Performing Combinations:**
- XGBoost appears in 1/1 top combinations
- Random Forest appears in 1/1 top combinations
- InSilicoVA appears in 1/1 top combinations
- Categorical NB appears in 1/1 top combinations
- Logistic Regression appears in 1/1 top combinations


## Recommendations

1. **Best Overall Combination**: Use XGBoost + Random Forest + InSilicoVA + Categorical NB + Logistic Regression for maximum ensemble performance (0.6703 CSMF accuracy)

2. **Avoid Low-Performing Combinations**: Avoid XGBoost + Random Forest + Categorical NB (0.5516 CSMF accuracy)

3. **Individual vs Ensemble Trade-offs**: Consider computational cost vs performance improvement when choosing between individual models and ensemble combinations

4. **Site-Specific Performance**: Performance varies by train/test site combination - consider site-specific model selection for optimal results

