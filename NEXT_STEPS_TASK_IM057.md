# NEXT STEPS for Task IM-057: Incorporate InSilicoVA Medical Priors into XGBoost

## Task Overview
**Task ID**: IM-057  
**Title**: Incorporate InSilicoVA medical priors into XGBoost for improved generalization  
**Type**: Implementation  
**Priority**: High  
**Dependencies**: RD-018 (completed), IM-035 (completed), IM-045 (completed)

## Current Context
Based on the RD-018 research report, we discovered that InSilicoVA's superior cross-site generalization (33.9% drop vs XGBoost's 37.7% drop) is largely due to its incorporation of medical expert knowledge through Bayesian priors. This task aims to bridge that gap by integrating similar medical constraints into XGBoost.

## Key Research Findings to Apply
1. **InSilicoVA's Advantage**: Uses conditional probability tables encoding medical knowledge
2. **XGBoost's Weakness**: Learns spurious site-specific patterns without medical constraints
3. **Expected Impact**: 15-25% reduction in generalization gap based on analysis

## Implementation Steps

### 1. Prior Data Extraction Phase
- Access InSilicoVA prior tables from https://github.com/verbal-autopsy-software/InSilicoVA/tree/master/InSilicoVA/data
- Parse and understand the structure of:
  - `probbase.csv`: Base probability tables
  - `condprob.csv`: Conditional probabilities for symptom-cause relationships
  - Population-level cause distributions
- Convert R data formats to Python-compatible structures
- Create a prior knowledge module to encapsulate this data

### 2. XGBoost Integration Approaches

#### Approach 1: Custom Objective Function (Primary)
- Modify XGBoost training to include prior knowledge term
- Implement custom objective: `L_total = L_data + λ × L_prior`
- Where L_prior penalizes predictions that deviate from medical priors
- Create hyperparameter λ to balance data vs prior influence

#### Approach 2: Feature Engineering (Secondary)
- Create synthetic features based on symptom-cause associations
- Calculate prior probability features for each sample
- Include medical plausibility scores as features
- Design ratio features that encode known medical relationships

#### Approach 3: Sample Weighting (Tertiary)
- Weight training samples based on prior likelihood
- Up-weight medically plausible patterns
- Down-weight unlikely symptom-cause combinations
- Implement dynamic weighting during training

### 3. Implementation Architecture
```
baseline/models/
├── xgboost_model.py (existing)
├── xgboost_prior_enhanced.py (new)
├── medical_priors/
│   ├── __init__.py
│   ├── prior_loader.py (load InSilicoVA data)
│   ├── prior_calculator.py (compute prior probabilities)
│   └── prior_constraints.py (enforce medical rules)
└── tests/
    └── test_xgboost_prior_enhanced.py
```

### 4. Validation Strategy
- Rerun IM-035 experiments with prior-enhanced XGBoost
- Compare against vanilla XGBoost and InSilicoVA
- Measure both in-domain and cross-site performance
- Track computational overhead
- Ensure interpretability is maintained

### 5. Success Metrics
- **Primary**: Reduce cross-site generalization gap by 15-25%
- **Secondary**: Maintain or improve in-domain performance
- **Tertiary**: Keep computational overhead under 20%
- **Qualitative**: Ensure predictions are medically plausible

## Technical Considerations

### Data Format Compatibility
- InSilicoVA uses "Y"/"." format for symptoms
- XGBoost uses numeric encoding (0/1)
- Need mapping between formats when loading priors

### Prior Knowledge Structure
```python
# Expected structure of medical priors
priors = {
    'conditional_probs': {
        ('symptom_1', 'cause_1'): 0.85,
        ('symptom_1', 'cause_2'): 0.12,
        # ... more symptom-cause pairs
    },
    'cause_priors': {
        'cause_1': 0.15,
        'cause_2': 0.08,
        # ... population-level distributions
    },
    'implausible_patterns': [
        # List of medically impossible combinations
    ]
}
```

### Integration Points
1. Custom objective function hooks in XGBoost
2. Feature preprocessing pipeline modifications
3. Model evaluation extensions for medical plausibility

## Risk Mitigation
1. **Prior Misspecification**: Validate priors with medical experts
2. **Performance Degradation**: Implement λ scheduling to find optimal balance
3. **Complexity**: Start with simple approach, incrementally add sophistication
4. **Backward Compatibility**: Ensure vanilla XGBoost mode still works

## Dependencies to Review
- `baseline/models/xgboost_model.py`: Current implementation
- `baseline/models/insilico_model.py`: How InSilicoVA handles priors
- `results/full_va34_comparison_complete/`: Baseline performance metrics
- `reports/xgboost_insilico_analysis.md`: Detailed algorithmic analysis

## Estimated Effort
- Prior data extraction and parsing: 1 day
- Custom objective implementation: 2 days
- Feature engineering approach: 1 day
- Testing and validation: 1 day
- Documentation and cleanup: 0.5 days
- Total: 5.5 days

## Next Implementation Tasks (Future)
Based on this implementation, potential follow-ups:
- [IM-058] Create ensemble combining prior-enhanced XGBoost with InSilicoVA
- [IM-059] Extend prior integration to other ML models (RF, SVM)
- [IM-060] Develop automated prior extraction from medical literature

## Acceptance Criteria
- [ ] Medical priors successfully extracted from InSilicoVA repository
- [ ] At least one integration approach fully implemented
- [ ] Cross-site experiments show 15%+ improvement in generalization
- [ ] All tests pass with >90% coverage
- [ ] Documentation includes usage examples and parameter tuning guide
- [ ] Results comparison table showing vanilla vs enhanced performance