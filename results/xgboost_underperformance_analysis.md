# XGBoost Underperformance Root Cause Analysis

## 1. Problem Summary

XGBoost is underperforming compared to InSilicoVA in VA (Verbal Autopsy) model comparison:
- **Overall CSMF Accuracy**: XGBoost (0.4800) vs InSilicoVA (0.5457)
- **Overall COD Accuracy**: XGBoost (0.2532) vs InSilicoVA (0.2617)
- **Head-to-head**: InSilicoVA wins 55.6% on CSMF and 52.8% on COD accuracy
- **Key Issue**: XGBoost shows severe overfitting with massive performance degradation in out-of-domain scenarios

## 2. Initial Hypotheses (Ranked by Likelihood)

1. **Severe Overfitting**: XGBoost memorizes training site patterns instead of learning generalizable cause-of-death features
2. **Suboptimal Hyperparameter Search Space**: Current tuning may not explore regularization adequately
3. **Feature Representation Issues**: Binary encoding may not capture VA data nuances as well as InSilicoVA's probabilistic approach
4. **Insufficient Regularization**: Default parameters favor complex models that don't generalize
5. **Class Imbalance Handling**: Sample weighting strategy may be suboptimal
6. **Missing Domain Adaptation**: No transfer learning mechanisms for cross-site generalization

## 3. Diagnostic Analysis

### 3.1 Overfitting Evidence

**In-Domain vs Out-Domain Performance Gap:**
```
XGBoost:
- In-Domain CSMF: 0.8323 (±0.0504)
- Out-Domain CSMF: 0.3597 (±0.2450)
- Performance Drop: 56.8%

InSilicoVA:
- In-Domain CSMF: 0.7997 (±0.0670)
- Out-Domain CSMF: 0.4605 (±0.1159)
- Performance Drop: 42.4%
```

**Worst Transfer Learning Scenarios (XGBoost):**
- Dar → UP: CSMF = 0.029 (97% drop!)
- AP → Bohol: CSMF = 0.048
- Multiple scenarios with <15% CSMF accuracy

### 3.2 Model Stability Analysis

**Standard Deviation Comparison:**
- XGBoost CSMF std: 0.2993 (64% higher variance than InSilicoVA)
- InSilicoVA CSMF std: 0.1814
- XGBoost shows high instability across different site combinations

### 3.3 Hyperparameter Configuration Issues

**Current XGBoost Search Space:**
```python
{
    'max_depth': [3, 5, 7, 10],  # Allows deep trees
    'learning_rate': [0.01, 0.3],  # Wide range
    'n_estimators': [100, 200, 500],  # Many trees
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'reg_alpha': [1e-4, 1.0],  # L1 regularization too weak
    'reg_lambda': [1.0, 10.0],  # L2 regularization range limited
}
```

**Issues Identified:**
1. Regularization ranges are insufficient for high-dimensional VA data
2. No explicit min_child_weight parameter for leaf node regularization
3. Missing gamma parameter for pruning control
4. Default early stopping may prevent proper regularization

## 4. Root Causes

### Primary Root Cause: Insufficient Regularization for High-Dimensional Sparse Data

VA data characteristics that exacerbate overfitting:
- High dimensionality (200+ binary features)
- Sparse features (many symptoms absent)
- Site-specific symptom reporting patterns
- Small effective sample sizes per cause

### Secondary Root Causes:

1. **Feature Engineering Gap**: Binary "Y"/"." encoding loses information compared to InSilicoVA's probabilistic symptom modeling

2. **No Domain Adaptation**: XGBoost treats each site independently without leveraging cross-site knowledge

3. **Hyperparameter Tuning Bias**: Optimizing on in-domain CV doesn't capture generalization needs

## 5. Recommendations (Prioritized by Impact)

### 5.1 Immediate Fixes (High Impact, Low Effort)

**1. Expand Regularization Search Space:**
```python
def get_xgboost_search_space() -> Dict[str, Any]:
    return {
        # Tree-specific parameters
        'config__max_depth': tune.choice([2, 3, 4, 5, 6]),  # Reduced depth
        'config__learning_rate': tune.loguniform(0.01, 0.1),  # Lower rates
        'config__n_estimators': tune.choice([200, 500, 1000]),  # More trees with lower LR
        
        # Sampling parameters
        'config__subsample': tune.uniform(0.5, 0.8),  # More aggressive subsampling
        'config__colsample_bytree': tune.uniform(0.3, 0.7),  # Fewer features per tree
        'config__colsample_bylevel': tune.uniform(0.3, 0.7),  # Add level-wise sampling
        
        # Enhanced regularization
        'config__reg_alpha': tune.loguniform(0.1, 100.0),  # Much stronger L1
        'config__reg_lambda': tune.loguniform(1.0, 100.0),  # Stronger L2
        'config__gamma': tune.loguniform(0.1, 10.0),  # Add minimum loss reduction
        'config__min_child_weight': tune.choice([5, 10, 20, 50]),  # Prevent small leaves
    }
```

**2. Add Cross-Domain Validation to Tuning:**
```python
# In ray_tasks.py tune_and_train_model function
# Instead of just in-domain CV, use leave-one-site-out CV
def create_cross_domain_cv(X, y, sites, n_folds=5):
    """Create CV splits that test generalization across sites."""
    # Implementation to ensure validation sets come from different sites
```

### 5.2 Medium-Term Improvements (High Impact, Medium Effort)

**3. Implement Multi-Objective Tuning:**
```python
# Optimize for both in-domain and out-domain performance
tuning_metric = "weighted_score"  # 0.7 * in_domain + 0.3 * out_domain
```

**4. Add Feature Engineering for VA Data:**
```python
def engineer_va_features(X):
    """Create domain-specific features."""
    # Symptom co-occurrence patterns
    # Symptom cluster indicators
    # Missing data patterns as features
    return X_engineered
```

**5. Implement Ensemble with Domain Adaptation:**
```python
class DomainAdaptiveXGBoost:
    """XGBoost with site-specific adaptations."""
    def __init__(self):
        self.site_models = {}
        self.global_model = None
    
    def fit(self, X, y, sites):
        # Train global model on all data
        # Train site-specific models with transfer learning
        pass
```

### 5.3 Advanced Solutions (High Impact, High Effort)

**6. Implement Gradient Reversal for Domain Adaptation:**
- Add adversarial training to make features site-invariant
- Force model to learn cause patterns that generalize across sites

**7. Hierarchical Bayesian Approach:**
- Model site-specific variations explicitly
- Share information across sites through hierarchical priors

## 6. Experimental Validation Plan

### Experiment 1: Regularization Impact
```bash
# Test with enhanced regularization
poetry run python model_comparison/scripts/run_distributed_comparison.py \
    --models xgboost \
    --enable-tuning \
    --tuning-trials 200 \
    --tuning-metric csmf_accuracy \
    --custom-search-space enhanced_regularization
```

### Experiment 2: Cross-Domain Validation
```bash
# Compare in-domain vs cross-domain tuning
poetry run python test_cross_domain_tuning.py \
    --validation-strategy cross_domain \
    --n-trials 100
```

### Experiment 3: Feature Engineering Impact
```bash
# Test with engineered features
poetry run python test_feature_engineering.py \
    --feature-set engineered \
    --models xgboost insilico
```

## 7. Success Metrics

1. **Primary**: Reduce out-domain CSMF accuracy gap to <30% (from current 56.8%)
2. **Secondary**: Achieve >50% out-domain CSMF accuracy average
3. **Stability**: Reduce CSMF std deviation to <0.20
4. **Competitiveness**: Win >45% of head-to-head comparisons with InSilicoVA

## 8. Implementation Priority

1. **Week 1**: Implement enhanced regularization search space and test
2. **Week 2**: Add cross-domain validation to tuning pipeline
3. **Week 3**: Develop and test VA-specific feature engineering
4. **Week 4**: Implement ensemble approach with site adaptation
5. **Week 5**: Evaluate all improvements and create final recommendation

## Key Insight

XGBoost's flexibility becomes a liability with high-dimensional, sparse, multi-site VA data. While it achieves excellent in-domain performance, it catastrophically fails to generalize. The solution requires aggressive regularization, domain-aware tuning, and potentially architectural changes to handle the unique characteristics of verbal autopsy data where symptom patterns vary significantly across geographic and cultural contexts.