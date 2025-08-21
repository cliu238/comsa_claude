# Out-Domain Model Performance Analysis Report
## Executive Summary
### Key Findings:
1. **Best out-domain models vary by classification complexity**
   - COD5: INSILICO achieves 50.1% COD accuracy
   - VA34: INSILICO achieves 21.5% COD accuracy

2. **High variance in out-domain performance**
   - Performance heavily depends on source-target site combination
   - Some transfers nearly random (e.g., Bohol→Pemba: 1.7% in VA34)

3. **Site-specific patterns emerge**
   - Pemba is consistently difficult both as source and target
   - Dar and AP tend to be good transfer targets

## Detailed Performance Metrics
### COD5 Out-Domain Performance
| Model | CSMF Mean±Std | COD Mean±Std | COD Range |
|-------|---------------|--------------|------------|
| CATEGORICAL_NB | 0.448±0.235 | 0.322±0.126 | 0.075-0.569 |
| RANDOM_FOREST | 0.671±0.192 | 0.454±0.112 | 0.210-0.665 |
| XGBOOST | 0.726±0.161 | 0.478±0.116 | 0.215-0.694 |
| LOGISTIC_REGRESSION | 0.494±0.259 | 0.342±0.129 | 0.099-0.505 |
| INSILICO | 0.730±0.117 | 0.501±0.079 | 0.301-0.641 |

### VA34 Out-Domain Performance
| Model | CSMF Mean±Std | COD Mean±Std | COD Range |
|-------|---------------|--------------|------------|
| CATEGORICAL_NB | 0.295±0.151 | 0.114±0.071 | 0.033-0.303 |
| RANDOM_FOREST | 0.365±0.180 | 0.162±0.101 | 0.000-0.338 |
| XGBOOST | 0.383±0.218 | 0.181±0.123 | 0.017-0.395 |
| LOGISTIC_REGRESSION | 0.371±0.173 | 0.129±0.101 | 0.000-0.400 |
| INSILICO | 0.461±0.116 | 0.215±0.064 | 0.088-0.417 |

## Transfer Pattern Analysis
### Best Cross-Site Transfers
#### COD5 Top 3:

#### VA34 Top 3:

### Worst Cross-Site Transfers
#### COD5 Bottom 3:
1. CATEGORICAL_NB: Bohol → Mexico: COD=0.075
2. CATEGORICAL_NB: Mexico → Dar: COD=0.136
3. CATEGORICAL_NB: Bohol → Dar: COD=0.157

#### VA34 Bottom 3:
1. CATEGORICAL_NB: Pemba → Mexico: COD=0.033
2. CATEGORICAL_NB: Dar → Mexico: COD=0.039
3. CATEGORICAL_NB: Dar → Bohol: COD=0.048

## Site-Specific Analysis
### Best Source Sites (for generalization)
- COD5: AP (avg COD accuracy: 0.466)
- VA34: UP (avg COD accuracy: 0.220)

### Most Difficult Target Sites
- COD5: Bohol (avg COD accuracy: 0.384)
- VA34: Mexico (avg COD accuracy: 0.122)

## Recommendations
1. **Consider InSilico for multi-site deployments** - better cross-site generalization
2. **Site-specific calibration is critical** - especially for Pemba site
3. **Pool training data from multiple sites** - to improve generalization
4. **Validate on target population before deployment** - transfer performance varies widely
5. **COD5 more robust for cross-site deployment** - maintains reasonable accuracy
