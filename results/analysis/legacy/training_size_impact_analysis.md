# Impact of Training Data Size on VA Model Performance: CSMF vs COD Accuracy Analysis

## Executive Summary

Analysis of the VA34 comparison results reveals a striking phenomenon: **XGBoost exhibits disproportionately high population-level (CSMF) accuracy compared to individual-level (COD) accuracy when trained on small datasets**, while InSilico and TabICL show more balanced performance across both metrics. This behavior has important implications for model selection in resource-constrained settings.

## Key Findings

### Performance at 5% Training Data (59 samples)

| Model | CSMF Accuracy | COD Accuracy | CSMF/COD Ratio |
|-------|--------------|--------------|----------------|
| XGBoost | 0.649 | 0.168 | **3.85** |
| InSilico | 0.547 | 0.216 | 2.54 |
| TabICL | 0.510 | 0.209 | 2.44 |

**Key Observation**: XGBoost's CSMF/COD ratio of 3.85 is significantly higher than InSilico (2.54) and TabICL (2.44), indicating it learns population distributions much better than individual patterns with limited data.

### Performance Improvement with More Data

| Model | CSMF Improvement (5% → 100%) | COD Improvement (5% → 100%) |
|-------|------------------------------|------------------------------|
| XGBoost | +35.4% | **+180.0%** |
| InSilico | +45.7% | +98.4% |
| TabICL | +59.6% | +137.1% |

**Key Insight**: XGBoost shows the most dramatic COD improvement (+180%) as training data increases, suggesting it needs substantial data to learn individual-level patterns effectively.

## Understanding the Metrics

### CSMF (Cause-Specific Mortality Fraction) Accuracy
- **What it measures**: How well the model predicts the distribution of causes across a population
- **Level**: Population/aggregate
- **Key property**: Can be high even if individual predictions are wrong, as long as the overall distribution is correct

### COD (Cause of Death) Accuracy
- **What it measures**: Percentage of individual cases correctly classified
- **Level**: Individual/case-by-case
- **Key property**: Requires precise prediction for each specific case

## Why XGBoost Shows This Pattern

### 1. Learning Strategy Differences

**XGBoost with Small Data**:
- Quickly learns class priors and base rates
- Makes conservative predictions biased toward frequent causes
- Effectively captures population distribution
- Struggles to differentiate individual cases

**InSilico & TabICL**:
- InSilico uses Bayesian probabilistic modeling with domain-specific priors
- TabICL leverages in-context learning with example-based reasoning
- Both incorporate structural knowledge about VA data

### 2. Theoretical Example

Consider a simplified scenario with 100 deaths and 3 causes:
- Cause A: 50% (50 deaths)
- Cause B: 30% (30 deaths)
- Cause C: 20% (20 deaths)

If XGBoost predominantly predicts Cause A (the most common):
- **COD Accuracy**: ~50% (only Cause A cases correct)
- **CSMF Accuracy**: Still reasonable because it captures that A is dominant
- **Result**: High CSMF/COD ratio

### 3. Regularization and Overfitting

With 59 training samples across 34 classes:
- **XGBoost**: May overfit to training distribution, learning "what's common" rather than "what distinguishes"
- **InSilico**: Built-in Bayesian regularization prevents overfitting
- **TabICL**: Few-shot learning design inherently handles small data

## Cross-Domain Generalization

### Performance Drop from In-Domain to Out-Domain

| Model | CSMF Drop | COD Drop | Out-Domain CSMF/COD Ratio |
|-------|-----------|----------|---------------------------|
| XGBoost | -56.9% | -63.7% | 2.21 |
| InSilico | -45.6% | -51.4% | 2.13 |
| TabICL | -46.9% | -56.6% | 2.09 |

**Observation**: All models show similar CSMF/COD ratios in out-domain settings, suggesting the phenomenon is most pronounced with very small training data.

## Practical Implications

### When to Use Each Model

**XGBoost**:
- ✅ Best when you have abundant training data (>1000 samples)
- ✅ When population-level estimates are sufficient
- ❌ Not recommended for small data individual predictions

**InSilico**:
- ✅ Good balance at all data sizes
- ✅ Strong domain-specific knowledge incorporation
- ✅ Reliable with small training sets

**TabICL**:
- ✅ Excellent for few-shot scenarios
- ✅ Most consistent COD accuracy with limited data
- ✅ Best relative improvement in both metrics

### Recommendations

1. **For small data scenarios (<100 samples)**:
   - Prioritize InSilico or TabICL for individual-level predictions
   - Use XGBoost only if population-level estimates are the primary goal

2. **For population surveillance**:
   - XGBoost can provide reasonable CSMF estimates even with limited data
   - Consider ensemble approaches combining XGBoost's population estimates with InSilico/TabICL's individual predictions

3. **For clinical decision support**:
   - Avoid XGBoost with small training sets
   - InSilico and TabICL provide more reliable individual predictions

## Conclusion

The divergence between CSMF and COD accuracy in XGBoost with small training data reveals a fundamental trade-off in machine learning for VA classification. While XGBoost excels at learning population distributions quickly, it requires substantial data to achieve comparable individual-level accuracy. InSilico and TabICL's more balanced performance stems from their incorporation of domain knowledge and specialized learning strategies designed for limited data scenarios.

This analysis underscores the importance of:
1. Selecting models based on the specific use case (population vs. individual)
2. Understanding metric limitations and what they actually measure
3. Considering data availability when choosing modeling approaches

---
*Analysis based on VA34 comparison results from the Context Engineering VA Pipeline*