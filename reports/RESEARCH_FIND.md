# Verbal Autopsy Baseline Benchmark Research Findings

## Executive Summary

This document presents the comprehensive findings from the baseline benchmark analysis of verbal autopsy (VA) models conducted on July 29, 2025. The study evaluated two primary models (InSilicoVA and XGBoost) across six geographic sites using both in-domain and out-of-domain experimental configurations. The analysis reveals significant performance variations between models and experimental conditions, with critical implications for VA model deployment in diverse settings.

## Study Overview

### Experimental Design
- **Total Experiments**: 74 comparative analyses
- **Models Evaluated**: InSilicoVA (n=37), XGBoost (n=37)
- **Sites**: Andhra Pradesh (AP), Bohol, Dar es Salaam (Dar), Mexico, Pemba, Uttar Pradesh (UP)
- **Experiment Types**:
  - In-domain: 12 experiments (training and testing on same site)
  - Out-of-domain: 60 experiments (training on one site, testing on another)
  - Training size: 2 experiments (full dataset evaluation)

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
- **Average CSMF Accuracy**: 0.8330 ± 0.0586
- **Average COD Accuracy**: 0.4480 ± 0.1026

**Model-Specific In-Domain Results**:
- **InSilicoVA**: CSMF = 0.7997, COD = 0.4097
- **XGBoost**: CSMF = 0.8663, COD = 0.4863

#### Out-of-Domain Performance (Cross-Site Transfer)
- **Average CSMF Accuracy**: 0.4302 ± 0.1722 (48.4% decrease from in-domain)
- **Average COD Accuracy**: 0.2069 ± 0.1008 (53.8% decrease from in-domain)

**Model-Specific Out-of-Domain Results**:
- **InSilicoVA**: CSMF = 0.4605, COD = 0.2145
- **XGBoost**: CSMF = 0.3999, COD = 0.1993

### 2. Site-Specific Performance Analysis

#### Best Performing Site Pairs (Out-of-Domain)
1. **XGBoost Mexico’AP**: CSMF = 0.7744, COD = 0.3737
2. **XGBoost AP’UP**: CSMF = 0.7429, COD = 0.4342
3. **XGBoost Bohol’Dar**: CSMF = 0.6914, COD = 0.3241
4. **XGBoost Mexico’UP**: CSMF = 0.6762, COD = 0.3843
5. **XGBoost UP’AP**: CSMF = 0.6689, COD = 0.3569

#### Worst Performing Site Pairs (Out-of-Domain)
1. **XGBoost Dar’Mexico**: CSMF = 0.0654, COD = 0.0327
2. **XGBoost Dar’Pemba**: CSMF = 0.1000, COD = 0.0667
3. **XGBoost Bohol’Mexico**: CSMF = 0.1013, COD = 0.0425
4. **XGBoost Dar’UP**: CSMF = 0.1357, COD = 0.0427
5. **XGBoost Bohol’Pemba**: CSMF = 0.1667, COD = 0.0167

### 3. Model Comparison

#### InSilicoVA vs XGBoost Performance
- **In-Domain**: XGBoost outperforms InSilicoVA by 8.4% in CSMF accuracy and 18.7% in COD accuracy
- **Out-of-Domain**: InSilicoVA shows better generalization with 15.2% higher CSMF accuracy than XGBoost
- **Stability**: InSilicoVA demonstrates more consistent performance across sites (lower standard deviation)

#### Site-Specific Model Robustness
- **Pemba** (smallest dataset): Shows poorest generalization as both source and target domain
- **AP and UP**: Demonstrate strongest bidirectional transfer learning capabilities
- **Dar**: Shows asymmetric performance - good as target but poor as source domain

### 4. Statistical Significance

All experiments included 95% confidence intervals. Notable findings:
- **Widest CI ranges**: Observed in small sample size experiments (Pemba-related transfers)
- **Narrowest CI ranges**: Found in large sample in-domain experiments
- **CI Overlap**: Significant overlap between models in out-of-domain scenarios suggests comparable performance under domain shift

### 5. Execution Performance
- **Average InSilicoVA runtime**: 52.3 seconds per experiment
- **Average XGBoost runtime**: 194.7 seconds per experiment
- **No failed experiments**: 0 errors across all 74 runs

## Research Implications

### 1. Domain Adaptation Challenge
The substantial performance degradation (48-54%) in out-of-domain scenarios highlights the critical need for domain adaptation techniques in VA deployment. This finding suggests that models trained in one geographic region may not be directly applicable to others without significant performance loss.

### 2. Sample Size Effects
Pemba's consistently poor performance (both as source and target) with only 237 training samples suggests a minimum sample size threshold for reliable VA model training. Sites with >1000 samples showed markedly better generalization.

### 3. Model Selection Considerations
- **For single-site deployment**: XGBoost offers superior performance
- **For multi-site deployment**: InSilicoVA's better generalization makes it more suitable
- **Computational constraints**: InSilicoVA's 3.7x faster execution time provides practical advantages

### 4. Geographic and Cultural Factors
The strong performance in AP”UP transfers (both in North India) versus poor Dar’Mexico transfers suggests that geographic/cultural proximity may influence model transferability, warranting further investigation into demographic and epidemiological similarities.

## Recommendations for Future Work

1. **Develop domain adaptation techniques** to bridge the 48-54% performance gap in cross-site deployments
2. **Investigate minimum sample size requirements** - recommend collecting >1000 samples per site
3. **Explore ensemble methods** combining InSilicoVA's generalization with XGBoost's in-domain accuracy
4. **Conduct demographic analysis** to understand which population characteristics drive transfer learning success
5. **Implement active learning strategies** to efficiently improve performance on underrepresented sites

## Limitations

1. **Unbalanced sample sizes**: Pemba's small dataset (237 samples) limits conclusions about its true performance potential
2. **Limited model diversity**: Only two model types evaluated; neural network approaches not included
3. **Single metric focus**: Analysis primarily on CSMF and COD accuracy; other VA metrics not examined
4. **No temporal validation**: All experiments assume static populations; temporal drift not assessed

## Conclusion

This baseline benchmark establishes critical performance boundaries for VA models across diverse geographic settings. The findings demonstrate that while in-domain performance is strong (83.3% CSMF accuracy), significant challenges remain for cross-site deployment (43.0% CSMF accuracy). The differential performance characteristics of InSilicoVA and XGBoost suggest that model selection should be guided by deployment context: XGBoost for single-site accuracy, InSilicoVA for multi-site robustness. These results provide a quantitative foundation for developing targeted improvements in transfer learning and domain adaptation for global VA deployment.

---
*Document generated: July 29, 2025*  
*Data source: `/results/full_comparison_20250729_143342/va34_comparison_results.csv`*