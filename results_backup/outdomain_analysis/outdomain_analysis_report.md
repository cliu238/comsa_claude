# Out-Domain Model Performance Analysis Report
## Executive Summary
### Key Findings:
1. **InSilico shows better generalization in out-domain scenarios**
   - COD5: InSilico (50.1%) vs XGBoost (47.8%) COD accuracy
   - VA34: InSilico (21.5%) vs XGBoost (18.1%) COD accuracy

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
| XGBOOST | 0.726±0.161 | 0.478±0.116 | 0.215-0.694 |
| INSILICO | 0.730±0.117 | 0.501±0.079 | 0.301-0.641 |

### VA34 Out-Domain Performance
| Model | CSMF Mean±Std | COD Mean±Std | COD Range |
|-------|---------------|--------------|------------|
| XGBOOST | 0.383±0.218 | 0.181±0.123 | 0.017-0.395 |
| INSILICO | 0.461±0.116 | 0.215±0.064 | 0.088-0.417 |

## Transfer Pattern Analysis
### Best Cross-Site Transfers
#### COD5 Top 3:
1. XGBOOST: Bohol → Dar: COD=0.694, CSMF=0.883
2. XGBOOST: AP → Dar: COD=0.648, CSMF=0.880
3. XGBOOST: Bohol → AP: COD=0.640, CSMF=0.933

#### VA34 Top 3:

### Worst Cross-Site Transfers
#### COD5 Bottom 3:
1. XGBOOST: Pemba → Bohol: COD=0.215
2. XGBOOST: Bohol → Pemba: COD=0.267
3. XGBOOST: Pemba → Dar: COD=0.269

#### VA34 Bottom 3:
1. XGBOOST: Bohol → Pemba: COD=0.017
2. XGBOOST: Dar → Pemba: COD=0.033
3. XGBOOST: UP → Bohol: COD=0.040

## Site-Specific Analysis
### Best Source Sites (for generalization)
- COD5: AP (avg COD accuracy: 0.549)
- VA34: Mexico (avg COD accuracy: 0.263)

### Most Difficult Target Sites
- COD5: Bohol (avg COD accuracy: 0.440)
- VA34: Mexico (avg COD accuracy: 0.155)

## Recommendations
1. **Consider InSilico for multi-site deployments** - better cross-site generalization
2. **Site-specific calibration is critical** - especially for Pemba site
3. **Pool training data from multiple sites** - to improve generalization
4. **Validate on target population before deployment** - transfer performance varies widely
5. **COD5 more robust for cross-site deployment** - maintains reasonable accuracy
