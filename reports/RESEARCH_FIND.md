# Verbal Autopsy Baseline Benchmark Research Findings

## Executive Summary

This document presents the comprehensive findings from the baseline benchmark analysis of verbal autopsy (VA) models conducted on July 29, 2025. The study evaluated five models (InSilicoVA, XGBoost, Random Forest, Logistic Regression, and Categorical Naive Bayes) across six geographic sites using both in-domain and out-of-domain experimental configurations. The analysis reveals significant performance variations between models and experimental conditions, with critical implications for VA model deployment in diverse settings.

## Study Overview

### Experimental Design
- **Total Experiments**: 200 comparative analyses
- **Models Evaluated**: 
  - InSilicoVA (n=40)
  - XGBoost (n=40)
  - Random Forest (n=40)
  - Logistic Regression (n=40)
  - Categorical Naive Bayes (n=40)
- **Sites**: Andhra Pradesh (AP), Bohol, Dar es Salaam (Dar), Mexico, Pemba, Uttar Pradesh (UP)
- **Experiment Types**:
  - In-domain: 30 experiments (6 sites × 5 models)
  - Out-of-domain: 150 experiments (30 site pairs × 5 models)
  - Training size: 20 experiments (4 subset sizes × 5 models)

### Sample Sizes by Site
- **Andhra Pradesh (AP)**: 1,186 training samples, 297 test samples
- **Bohol**: 1,001 training samples, 251 test samples
- **Dar es Salaam**: 1,293 training samples, 324 test samples
- **Mexico**: 1,222 training samples, 306 test samples
- **Pemba**: 237 training samples, 60 test samples (smallest dataset)
- **Uttar Pradesh (UP)**: 1,124 training samples, 281 test samples

## Key Findings

### 1. Overall Model Performance

#### In-Domain Performance (Same Site Training/Testing)
- **Average CSMF Accuracy**: 0.7904 ± 0.1259
- **Average COD Accuracy**: 0.4330 ± 0.1300

**Model-Specific In-Domain Results**:
- **InSilicoVA**: CSMF = 0.7997 ± 0.0537, COD = 0.4097 ± 0.0945
- **XGBoost**: CSMF = 0.8663 ± 0.0518, COD = 0.4863 ± 0.1079
- **Random Forest**: CSMF = 0.8447 ± 0.0496, COD = 0.5045 ± 0.0956
- **Logistic Regression**: CSMF = 0.7893 ± 0.0642, COD = 0.4370 ± 0.0890
- **Categorical Naive Bayes**: CSMF = 0.6520 ± 0.1368, COD = 0.3273 ± 0.1445

#### Out-of-Domain Performance (Cross-Site Transfer)
- **Average CSMF Accuracy**: 0.4056 ± 0.1871 (48.7% decrease from in-domain)
- **Average COD Accuracy**: 0.1866 ± 0.0960 (56.9% decrease from in-domain)

**Model-Specific Out-of-Domain Results**:
- **InSilicoVA**: CSMF = 0.4605 ± 0.1567, COD = 0.2145 ± 0.0945
- **XGBoost**: CSMF = 0.3999 ± 0.2078, COD = 0.1993 ± 0.1120
- **Random Forest**: CSMF = 0.4147 ± 0.1877, COD = 0.2095 ± 0.0943
- **Logistic Regression**: CSMF = 0.4154 ± 0.1681, COD = 0.1908 ± 0.0827
- **Categorical Naive Bayes**: CSMF = 0.3375 ± 0.1839, COD = 0.1189 ± 0.0653

### 2. Site-Specific Performance Analysis

#### Best Performing Site Pairs (Out-of-Domain)
1. **XGBoost Mexico→AP**: CSMF = 0.7744, COD = 0.3737
2. **XGBoost AP→UP**: CSMF = 0.7429, COD = 0.4342
3. **InSilicoVA Mexico→AP**: CSMF = 0.7137, COD = 0.3333
4. **InSilicoVA Mexico→UP**: CSMF = 0.7097, COD = 0.3629
5. **Random Forest Mexico→AP**: CSMF = 0.7086, COD = 0.3670

**Notable finding**: Mexico as source domain consistently produces best transfer results across all models

#### Worst Performing Site Pairs (Out-of-Domain)
1. **Categorical NB UP→Pemba**: CSMF = 0.0167, COD = 0.0167
2. **Categorical NB Bohol→Pemba**: CSMF = 0.0333, COD = 0.0167
3. **Categorical NB AP→Pemba**: CSMF = 0.0500, COD = 0.0167
4. **XGBoost Dar→Mexico**: CSMF = 0.0654, COD = 0.0327
5. **Categorical NB Dar→Pemba**: CSMF = 0.0667, COD = 0.0333

**Critical observation**: Pemba as target domain shows severe performance degradation across all models

### 3. Model Comparison

#### Overall Model Rankings
**In-Domain Performance**:
1. **XGBoost**: CSMF = 0.8663, COD = 0.4863 (best overall)
2. **Random Forest**: CSMF = 0.8447, COD = 0.5045 (best COD accuracy)
3. **InSilicoVA**: CSMF = 0.7997, COD = 0.4097
4. **Logistic Regression**: CSMF = 0.7893, COD = 0.4370
5. **Categorical NB**: CSMF = 0.6520, COD = 0.3273 (poorest performance)

**Out-of-Domain Performance**:
1. **InSilicoVA**: CSMF = 0.4605, COD = 0.2145 (best generalization)
2. **Logistic Regression**: CSMF = 0.4154, COD = 0.1908
3. **Random Forest**: CSMF = 0.4147, COD = 0.2095
4. **XGBoost**: CSMF = 0.3999, COD = 0.1993
5. **Categorical NB**: CSMF = 0.3375, COD = 0.1189

#### Performance Gap Analysis (In-Domain vs Out-of-Domain)
- **InSilicoVA**: 42.4% CSMF drop (most stable)
- **Logistic Regression**: 47.4% CSMF drop
- **Categorical NB**: 48.2% CSMF drop
- **Random Forest**: 50.9% CSMF drop
- **XGBoost**: 53.8% CSMF drop (largest degradation)

#### Site-Specific Model Robustness
- **Pemba** (smallest dataset): Shows catastrophic performance as target across all models
- **Mexico**: Strongest source domain for all models
- **AP and UP**: Demonstrate good bidirectional transfer for traditional ML models
- **Dar**: Shows model-dependent asymmetric performance

### 4. Statistical Significance

All experiments included 95% confidence intervals. Notable findings:
- **Widest CI ranges**: Observed in small sample size experiments (Pemba-related transfers)
- **Narrowest CI ranges**: Found in large sample in-domain experiments
- **CI Overlap**: Significant overlap between models in out-of-domain scenarios suggests comparable performance under domain shift

### 5. Training Size Experiments

The training size experiments evaluated model performance with varying dataset sizes (25%, 50%, 75%, and 100% of available training data) to understand data efficiency:

**Key Findings**:
- **InSilicoVA**: Shows robust performance even with 50% data, minimal improvement beyond 75%
- **XGBoost & Random Forest**: Demonstrate continuous improvement with more data, suggesting they benefit from larger datasets
- **Logistic Regression**: Plateaus at 75% data, indicating diminishing returns
- **Categorical NB**: Shows erratic performance across different data sizes, confirming instability

**Implications**: For resource-constrained settings, InSilicoVA can achieve acceptable performance with half the typical training data, while tree-based models require fuller datasets for optimal performance.

### 6. Execution Performance
- **InSilicoVA**: 238.3 ± 134.7 seconds (fastest)
- **Logistic Regression**: 5,754.2 ± 3,223.5 seconds
- **Categorical Naive Bayes**: 5,773.6 ± 3,240.0 seconds
- **Random Forest**: 6,652.8 ± 3,650.5 seconds
- **XGBoost**: 6,885.4 ± 3,796.2 seconds (slowest)
- **No failed experiments**: 0 errors across all 200 runs

**Critical finding**: InSilicoVA is 24-29x faster than ML models, making it suitable for real-time deployment

## Research Implications

### 1. Domain Adaptation Challenge
The substantial performance degradation (42-54%) in out-of-domain scenarios highlights the critical need for domain adaptation techniques in VA deployment. InSilicoVA shows the best resilience to domain shift (42.4% drop) while XGBoost shows the worst (53.8% drop), suggesting that model architecture significantly impacts transferability.

### 2. Sample Size Effects
Pemba's consistently poor performance (both as source and target) with only 237 training samples suggests a minimum sample size threshold for reliable VA model training. Sites with >1000 samples showed markedly better generalization.

### 3. Model Selection Considerations
- **For single-site deployment**: Random Forest offers best COD accuracy (0.5045), XGBoost best CSMF accuracy (0.8663)
- **For multi-site deployment**: InSilicoVA's superior generalization (0.4605 CSMF) makes it most suitable
- **For real-time applications**: InSilicoVA's 24-29x faster execution time (238s vs 5,700-6,800s) is crucial
- **For resource-constrained settings**: Avoid Categorical NB due to poor performance across all metrics

### 4. Geographic and Cultural Factors
Mexico emerges as the strongest source domain across all models, while Pemba consistently fails as target. Within-region transfers (e.g., AP↔UP in North India) show better performance than cross-continental transfers, suggesting that geographic/cultural proximity influences model transferability. However, the Mexico anomaly indicates that data quality and cause-of-death distribution may override geographic factors.

## Recommendations for Future Work

1. **Develop domain adaptation techniques** to bridge the 42-54% performance gap, prioritizing XGBoost and Random Forest which show largest drops
2. **Investigate minimum sample size requirements** - Pemba's catastrophic performance with 237 samples suggests >1000 samples critical threshold
3. **Explore ensemble methods** combining InSilicoVA's generalization with Random Forest's in-domain COD accuracy
4. **Analyze Mexico's exceptional transfer properties** to understand what makes it an ideal source domain
5. **Implement active learning strategies** specifically targeting Pemba and other small-sample sites
6. **Optimize ML model efficiency** - current 5,700-6,800s runtimes impractical for deployment

## Limitations

1. **Unbalanced sample sizes**: Pemba's small dataset (237 samples) severely impacts all model performances
2. **Computational constraints**: ML models' 1.5-2 hour runtimes limited extensive hyperparameter optimization
3. **Single metric focus**: Analysis primarily on CSMF and COD accuracy; other VA metrics not examined
4. **No temporal validation**: All experiments assume static populations; temporal drift not assessed
5. **Missing deep learning baselines**: Neural network approaches not included due to computational constraints

## Conclusion

This comprehensive baseline benchmark of five VA models establishes critical performance boundaries across diverse geographic settings. The findings demonstrate that while in-domain performance can reach excellent levels (86.6% CSMF accuracy for XGBoost), significant challenges remain for cross-site deployment (40.6% average CSMF accuracy). Model selection should be guided by deployment context: Random Forest for best in-domain COD accuracy (50.5%), InSilicoVA for multi-site robustness and real-time applications (24-29x faster), and avoidance of Categorical Naive Bayes due to consistently poor performance. The dramatic performance variations between sites (Mexico as ideal source, Pemba as problematic target) highlight the critical importance of data quality and sample size over geographic proximity. These results provide a quantitative foundation for developing targeted improvements in transfer learning and domain adaptation for global VA deployment.

---
*Document generated: July 29, 2025*  
*Data source: `/results/full_comparison_20250729_155434/va34_comparison_results.csv`*
*Analysis timestamp: 15:54:34*