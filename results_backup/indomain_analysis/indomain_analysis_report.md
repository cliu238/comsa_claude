# In-Domain Model Performance Analysis Report
## Executive Summary
### Key Findings:
1. **Best performing models vary by classification complexity**
   - COD5: XGBOOST achieves 69.6% COD accuracy
   - VA34: XGBOOST achieves 46.7% COD accuracy

2. **Significant performance range across models**
   - COD5: CATEGORICAL_NB (40.5%) to XGBOOST (69.6%)
   - VA34: CATEGORICAL_NB (19.4%) to XGBOOST (46.7%)

3. **Performance degradation from COD5 to VA34 affects all models**
   - CATEGORICAL_NB: 21.1% drop
   - RANDOM_FOREST: 22.5% drop
   - XGBOOST: 22.8% drop
   - LOGISTIC_REGRESSION: 34.0% drop
   - INSILICO: 20.4% drop

## Detailed Performance Metrics
### COD5 Classification (5 causes)
| Model | CSMF Accuracy | COD Accuracy | CSMF Std | COD Std |
|-------|--------------|-------------|----------|----------|
| CATEGORICAL_NB | 0.534 | 0.405 | 0.233 | 0.188 |
| RANDOM_FOREST | 0.916 | 0.677 | 0.067 | 0.059 |
| XGBOOST | 0.968 | 0.696 | 0.016 | 0.061 |
| LOGISTIC_REGRESSION | 0.835 | 0.635 | 0.066 | 0.099 |
| INSILICO | 0.887 | 0.614 | 0.031 | 0.085 |

### VA34 Classification (34 causes)
| Model | CSMF Accuracy | COD Accuracy | CSMF Std | COD Std |
|-------|--------------|-------------|----------|----------|
| CATEGORICAL_NB | 0.384 | 0.194 | 0.227 | 0.133 |
| RANDOM_FOREST | 0.792 | 0.452 | 0.034 | 0.086 |
| XGBOOST | 0.884 | 0.467 | 0.028 | 0.091 |
| LOGISTIC_REGRESSION | 0.615 | 0.296 | 0.142 | 0.102 |
| INSILICO | 0.800 | 0.410 | 0.067 | 0.082 |

## Site-Specific Performance
### Best Performing Sites
| Classification | Model | Best Site | COD Accuracy |
|---------------|-------|-----------|-------------|
| COD5 | CATEGORICAL_NB | UP | 0.655 |
| VA34 | CATEGORICAL_NB | UP | 0.416 |
| COD5 | RANDOM_FOREST | Pemba | 0.733 |
| VA34 | RANDOM_FOREST | Pemba | 0.583 |
| COD5 | XGBOOST | Dar | 0.756 |
| VA34 | XGBOOST | Pemba | 0.617 |
| COD5 | LOGISTIC_REGRESSION | UP | 0.726 |
| VA34 | LOGISTIC_REGRESSION | UP | 0.448 |
| COD5 | INSILICO | Dar | 0.710 |
| VA34 | INSILICO | Pemba | 0.533 |

## Statistical Analysis
### Model Performance Comparison (One-way ANOVA)
#### COD5 Classification:
- F-statistic: 6.845
- p-value: 0.0007
- Significant difference between models: Yes (α=0.05)

#### VA34 Classification:
- F-statistic: 8.037
- p-value: 0.0003
- Significant difference between models: Yes (α=0.05)

## Recommendations
1. **Consider XGBOOST for COD5 classification** - best overall performance
2. **Avoid Categorical Naive Bayes** - consistently poor performance across tasks
3. **Site-specific calibration is critical** - performance varies significantly by location
4. **COD5 classification recommended over VA34** - better accuracy for practical deployment
5. **Ensemble methods (RF, XGBoost) generally outperform single models**
