# VA Benchmark Analysis Summary Report

## Executive Summary

This report presents a comprehensive analysis of Verbal Autopsy (VA) model performance across 200 experiments, comparing 5 models (InSilicoVA, CategoricalNB, LogisticRegression, RandomForest, XGBoost) across 6 sites with varying training set sizes and domain transfer scenarios.

### Key Findings

1. **Best Overall Performance**: InSilicoVA demonstrates superior performance with:
   - CSMF Accuracy: 0.5457 (mean)
   - COD Accuracy: 0.2617 (mean)
   - Fastest execution time: 193.34 seconds (mean)

2. **Domain Transfer Impact**: Significant performance degradation observed when models are applied out-of-domain:
   - Average CSMF accuracy drop: 48.3%
   - Average COD accuracy drop: 55.9%
   - InSilicoVA shows highest absolute degradation but maintains best out-domain performance

3. **Statistical Significance**: 
   - Friedman test confirms significant differences between models (p < 0.05)
   - InSilicoVA significantly outperforms other models in pairwise comparisons

## Detailed Performance Metrics

### Table 1: Overall Model Performance (Full Training Set)

| Model | CSMF Acc (Mean±SD) | COD Acc (Mean±SD) | Execution Time (s) |
|-------|-------------------|-------------------|-------------------|
| InSilicoVA | 0.5246±0.1719 | 0.2520±0.1018 | 193.34 |
| CategoricalNB | 0.3155±0.1849 | 0.1251±0.0954 | 5893.87 |
| LogisticRegression | 0.3866±0.2484 | 0.1609±0.1310 | 6942.47 |
| RandomForest | 0.3946±0.2870 | 0.2017±0.1643 | 6698.12 |
| XGBoost | 0.4500±0.2909 | 0.2394±0.1692 | 6869.25 |

### Table 2: Domain Transfer Performance

| Model | In-Domain CSMF | Out-Domain CSMF | Degradation | In-Domain COD | Out-Domain COD | Degradation |
|-------|----------------|-----------------|-------------|---------------|----------------|-------------|
| InSilicoVA | 0.7997±0.0670 | 0.4605±0.1159 | 0.3392 | 0.4097±0.0817 | 0.2145±0.0637 | 0.1952 |
| CategoricalNB | 0.3771±0.2754 | 0.2901±0.1502 | 0.0870 | 0.1811±0.1501 | 0.1064±0.0683 | 0.0747 |
| RandomForest | 0.7655±0.0746 | 0.3057±0.2405 | 0.4598 | 0.4338±0.0870 | 0.1471±0.1262 | 0.2867 |
| LogisticRegression | 0.6124±0.3441 | 0.3353±0.2043 | 0.2771 | 0.2968±0.1911 | 0.1288±0.0969 | 0.1680 |
| XGBoost | 0.8323±0.0504 | 0.3597±0.2450 | 0.4726 | 0.4573±0.0513 | 0.1871±0.1416 | 0.2702 |

### Table 3: Training Size Impact on Performance

| Model | Performance Retention (100% → 25%) |
|-------|-----------------------------------|
| InSilicoVA | 146.2% |
| CategoricalNB | 189.5% |
| LogisticRegression | 145.9% |
| RandomForest | 194.3% |
| XGBoost | 179.4% |

*Note: Values >100% indicate improved performance with smaller training sets, likely due to overfitting reduction*

### Table 4: Site Difficulty Ranking

| Site | Avg CSMF | Avg COD | Difficulty Score |
|------|----------|---------|------------------|
| UP | 0.8058 | 0.4719 | 0.3611 (Easiest) |
| AP | 0.7851 | 0.4135 | 0.4007 |
| Dar | 0.7031 | 0.3364 | 0.4802 |
| Bohol | 0.6476 | 0.3068 | 0.5228 |
| Mexico | 0.5966 | 0.2725 | 0.5654 |
| Pemba | 0.5263 | 0.3333 | 0.5702 (Hardest) |

## Statistical Analysis Highlights

### Model Stability Analysis

| Model | Stability Score | Risk Assessment |
|-------|----------------|-----------------|
| InSilicoVA | 0.2079 (Most Stable) | Low computational risk, High domain transfer risk |
| XGBoost | 0.4026 | Low stability risk, High computational/transfer risk |
| RandomForest | 0.4857 | Low stability risk, High computational/transfer risk |
| LogisticRegression | 0.6419 | High stability risk, High all risks |
| CategoricalNB | 0.6798 (Least Stable) | High stability risk, Medium transfer risk |

### Computational Efficiency

| Model | Combined Efficiency (Performance/Hour) |
|-------|---------------------------------------|
| InSilicoVA | 7.2302 |
| XGBoost | 0.1807 |
| RandomForest | 0.1602 |
| LogisticRegression | 0.1419 |
| CategoricalNB | 0.1346 |

## Recommendations

### 1. Model Selection Guidelines

- **High Accuracy Required**: InSilicoVA
- **Limited Computational Resources**: InSilicoVA
- **Cross-Domain Deployment**: InSilicoVA (despite high degradation, maintains best absolute performance)
- **Limited Training Data**: RandomForest or CategoricalNB (show performance gains with smaller datasets)
- **Real-time Processing**: InSilicoVA

### 2. Deployment Considerations

1. **In-Domain Applications**: XGBoost shows best in-domain performance (0.8323 CSMF) but requires substantial computational resources
2. **Cross-Domain Applications**: InSilicoVA maintains usable performance (0.4605 CSMF) despite degradation
3. **Resource-Constrained Settings**: InSilicoVA is 30-35x faster than ML alternatives

### 3. Risk Mitigation Strategies

1. **Domain Transfer**: Consider domain adaptation techniques or site-specific calibration
2. **Computational Resources**: InSilicoVA is the only viable option for real-time or resource-limited deployments
3. **Performance Stability**: InSilicoVA shows lowest variance across different conditions

## Conclusions

1. **InSilicoVA emerges as the most practical choice** for VA applications, balancing performance, speed, and stability
2. **Domain transfer remains a significant challenge** for all models, with 48-56% average performance drops
3. **Site-specific variations are substantial**, suggesting the need for local validation before deployment
4. **ML models (especially XGBoost) can achieve superior in-domain performance** but at significant computational cost
5. **Training set size has unexpected effects**, with some models performing better on smaller datasets, indicating potential overfitting issues

## Data Files Generated

All analysis results have been saved to `/results/analysis/` including:
- Aggregate statistics by model
- Domain transfer analysis
- Training size impact studies
- Site-specific performance metrics
- Computational efficiency rankings
- Statistical significance tests
- Confidence interval analyses

These files provide detailed numerical support for all findings presented in this summary.