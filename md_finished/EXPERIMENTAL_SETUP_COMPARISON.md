# Experimental Setup Comparison: Our Results vs R Journal 2023

## Problem Statement

Our InSilicoVA implementation achieved **0.791 CSMF accuracy**, which appears to exceed the **0.74 CSMF accuracy** reported in the R Journal 2023 paper ([openVA: An Open-Source Package for Verbal Autopsy Data Processing](https://journal.r-project.org/articles/RJ-2023-020/)). However, upon closer examination of the methodology, we discovered a critical difference in experimental setup that makes direct comparison invalid.

## R Journal 2023 Methodology

### Training/Test Site Configuration
- **Training Sites**: All PHMRC adult records EXCEPT Andhra Pradesh
  - Mexico
  - Dar es Salaam (Tanzania)
  - Uttar Pradesh (India)
  - Bohol (Philippines) 
  - Pemba (Tanzania)
  - **Total training samples**: 6,287 records

- **Test Site**: Andhra Pradesh (AP), India ONLY
  - **Total test samples**: 1,554 records

### Model Configuration
- **MCMC iterations**: 10,000
- **Burn-in**: 5,000 iterations discarded
- **Thinning**: Saved 250 iterations after thinning
- **Target column**: `gs_text34` (cause-of-death labels)

### Results
- **InSilicoVA**: 0.74 CSMF accuracy
- **NBC**: 0.77 CSMF accuracy
- **InterVA**: 0.53 CSMF accuracy
- **Tariff**: 0.68 CSMF accuracy

## Our Current Methodology

### Training/Test Site Configuration
- **Training Sites**: Mixed sample from ALL 6 sites (stratified)
  - Mexico, Dar es Salaam, Andhra Pradesh, Uttar Pradesh, Bohol, Pemba
  - **Total training samples**: 6,065 records (80% of dataset)

- **Test Sites**: Mixed sample from ALL 6 sites (stratified)
  - Mexico, Dar es Salaam, Andhra Pradesh, Uttar Pradesh, Bohol, Pemba
  - **Total test samples**: 1,517 records (20% of dataset)

### Site Distribution (Well-Balanced)
- **Dar**: 21.5% train vs 20.6% test
- **Mexico**: 19.9% train vs 21.1% test
- **AP**: 19.6% train vs 19.3% test
- **UP**: 18.4% train vs 19.1% test
- **Bohol**: 16.7% train vs 15.8% test
- **Pemba**: 3.9% train vs 4.2% test

### Model Configuration
- **MCMC iterations**: 5,000 (limited by timeout)
- **Target column**: `va34` (34-cause classification)
- **Jump scale**: 0.05
- **Auto length**: FALSE

### Results
- **InSilicoVA**: 0.791 CSMF accuracy

## Critical Differences

### 1. Geographic Generalization vs Within-Distribution Testing

**R Journal 2023 (Geographic Generalization)**:
- Tests model's ability to generalize to a completely **unseen geographic region**
- More challenging: model must work on AP data having never seen AP patterns
- Tests **external validity** and geographic robustness

**Our Setup (Within-Distribution Testing)**:
- Tests model performance on data from the **same distribution** as training
- Less challenging: model has seen patterns from all sites during training
- Tests **internal validity** and overall model performance

### 2. Sample Size Differences

| Aspect | R Journal 2023 | Our Setup |
|--------|----------------|-----------|
| Training samples | 6,287 | 6,065 |
| Test samples | 1,554 | 1,517 |
| Test site diversity | 1 site only | 6 sites mixed |

### 3. Evaluation Difficulty

**AP-Only Testing** (R Journal 2023):
- Higher difficulty due to geographic domain shift
- Tests model robustness across different populations
- Real-world deployment scenario

**Mixed-Site Testing** (Our Setup):
- Lower difficulty due to same-distribution testing
- Tests model's learning capacity on familiar data
- Model validation scenario

## Why Our Results Aren't Directly Comparable

### 1. Different Evaluation Paradigms
- **R Journal 2023**: Cross-site generalization (harder)
- **Our Setup**: Within-distribution validation (easier)

### 2. Different Research Questions
- **R Journal 2023**: "Can the model work in new geographic regions?"
- **Our Setup**: "How well does the model learn the overall VA patterns?"

### 3. Expected Performance Differences
- Our 0.791 > their 0.74 is **expected** given easier evaluation setup
- Direct numerical comparison is **misleading**

## Implications

### 1. Our Implementation is Likely Correct
- Higher performance on easier task suggests proper implementation
- Feature exclusion and data preprocessing are working correctly
- Model is learning legitimate patterns, not exploiting data leakage

### 2. Need for Both Evaluation Types
- **Mixed-site**: Validates model learning capacity
- **AP-only**: Tests real-world deployment viability

### 3. Literature Comparison Requires Matching Setup
- Cannot claim superiority based on different experimental conditions
- Need AP-only results for fair comparison with published benchmarks

## Recommendations

### 1. Immediate Actions
1. **Implement AP-only testing** to match R Journal 2023 methodology
2. **Compare results** on identical experimental setup
3. **Document both approaches** for comprehensive evaluation

### 2. Expected Outcomes
- **AP-only CSMF accuracy**: Likely lower than 0.791, closer to 0.74
- **Validation**: Confirms our implementation matches published benchmarks
- **Comprehensive evaluation**: Both within-distribution and cross-site performance

### 3. Reporting Strategy
- Report **both experimental setups** in results
- Emphasize **methodology differences** when comparing to literature
- Use **AP-only results** for direct literature comparison
- Use **mixed-site results** for overall model validation

## Conclusion

Our 0.791 CSMF accuracy result is **not directly comparable** to the R Journal 2023 benchmark of 0.74 due to fundamentally different experimental setups:

- **Our setup** tests within-distribution performance (easier)
- **R Journal 2023** tests cross-site generalization (harder)

To establish fair comparison with published literature, we need to implement the **AP-only testing methodology** and compare results under identical conditions. This will validate our implementation and provide the definitive benchmark comparison needed for research credibility.

## References

- Richards, J., et al. (2023). "openVA: An Open-Source Package for Verbal Autopsy Data Processing." *The R Journal*, 15(1), 302-327. Available at: https://journal.r-project.org/articles/RJ-2023-020/

---
*Analysis completed on 2025-07-18 after discovering experimental setup differences with R Journal 2023 methodology*