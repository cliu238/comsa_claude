# Product Requirements Prompt (PRP) for Task IM-057

## Task Identification
**Task ID**: IM-057  
**Title**: Incorporate InSilicoVA Medical Priors into XGBoost for Improved Generalization  
**Type**: Implementation  
**Priority**: High  
**Issue**: #16  
**Deliverable**: Prior-enhanced XGBoost model with 15-25% improved cross-site generalization

## Context and Background

### Problem Statement
Based on the RD-018 research findings, XGBoost suffers from a 37.7% performance drop in out-of-domain scenarios compared to InSilicoVA's 33.9% drop. The key difference is InSilicoVA's incorporation of medical expert knowledge through Bayesian priors, which constrains the model to learn medically plausible patterns. This task aims to bridge that gap by integrating similar medical constraints into XGBoost.

### Business Impact
- **Improved Reliability**: Better cross-site generalization means VA models can be deployed across diverse populations
- **Cost Efficiency**: Reduces need for site-specific model training
- **Medical Validity**: Ensures predictions align with medical knowledge
- **Scalability**: Enables broader deployment of fast XGBoost models with improved robustness

### Technical Context
Based on codebase analysis:
- **XGBoost Implementation**: Uses sklearn interface with configurable parameters but no custom objective support
- **InSilicoVA Integration**: Docker-based execution with prior type configuration
- **Data Pipeline**: Supports both numeric (XGBoost) and Y/. format (InSilicoVA) encoding
- **Model Comparison**: Framework exists for evaluating cross-site performance

## Requirements

### Functional Requirements

#### 1. Prior Data Extraction Module
- **R1.1**: Create `baseline/models/medical_priors/` module structure
- **R1.2**: Implement prior data loader that can parse InSilicoVA's conditional probability tables
- **R1.3**: Convert R data formats to Python-compatible structures (numpy arrays, pandas DataFrames)
- **R1.4**: Create mapping between symptom names and indices for both encoding formats
- **R1.5**: Implement validation to ensure prior data integrity

#### 2. Custom XGBoost Objective Function
- **R2.1**: Implement custom objective function that incorporates prior probabilities
- **R2.2**: Design loss function: `L_total = L_data + λ × L_prior`
- **R2.3**: Create gradient and hessian calculations for the custom objective
- **R2.4**: Implement hyperparameter λ (lambda) to balance data vs prior influence
- **R2.5**: Ensure backward compatibility with vanilla XGBoost mode

#### 3. Feature Engineering Approach
- **R3.1**: Create prior probability features for each sample
- **R3.2**: Calculate symptom-cause association scores based on conditional probabilities
- **R3.3**: Design ratio features encoding medical relationships
- **R3.4**: Implement feature augmentation without replacing original features
- **R3.5**: Create feature importance analysis for prior-based features

#### 4. Model Implementation
- **R4.1**: Create `XGBoostPriorEnhanced` class extending current XGBoost model
- **R4.2**: Maintain sklearn-compatible interface (fit, predict, predict_proba)
- **R4.3**: Support both custom objective and feature engineering modes
- **R4.4**: Implement configuration for prior integration method selection
- **R4.5**: Add methods for prior influence visualization

#### 5. Validation and Testing
- **R5.1**: Implement unit tests with >90% coverage
- **R5.2**: Create integration tests with model comparison framework
- **R5.3**: Add tests for medical plausibility of predictions
- **R5.4**: Implement performance benchmarks vs vanilla XGBoost
- **R5.5**: Create cross-site validation experiments

### Non-Functional Requirements

#### Performance
- **NFR1**: Computational overhead must be under 20% vs vanilla XGBoost
- **NFR2**: Memory usage should not exceed 1.5x vanilla XGBoost
- **NFR3**: Model training time should remain under 5 minutes for standard datasets

#### Maintainability
- **NFR4**: Code must follow existing project patterns and style
- **NFR5**: All methods must have comprehensive docstrings
- **NFR6**: Configuration must use Pydantic models like existing code

#### Compatibility
- **NFR7**: Must work with existing data preprocessing pipeline
- **NFR8**: Must integrate with model comparison framework
- **NFR9**: Must support both single-site and cross-site experiments

## Implementation Guide

### 1. Module Structure
```
baseline/models/
├── xgboost_model.py (existing)
├── xgboost_prior_enhanced.py (new)
├── medical_priors/
│   ├── __init__.py
│   ├── prior_loader.py
│   ├── prior_calculator.py
│   ├── prior_constraints.py
│   └── data/
│       └── (extracted prior files)
└── tests/
    ├── test_xgboost_prior_enhanced.py
    └── test_medical_priors/
        ├── test_prior_loader.py
        └── test_prior_calculator.py
```

### 2. Prior Data Structure
```python
@dataclass
class MedicalPriors:
    """Container for medical prior probabilities"""
    conditional_probs: Dict[Tuple[str, str], float]  # (symptom, cause) -> probability
    cause_priors: Dict[str, float]  # cause -> population probability
    symptom_names: List[str]  # ordered list of symptoms
    cause_names: List[str]  # ordered list of causes
    implausible_patterns: List[Tuple[str, str]]  # medically impossible combinations
```

### 3. Custom Objective Implementation
```python
def prior_informed_objective(y_true, y_pred):
    """Custom objective incorporating medical priors
    
    Args:
        y_true: True labels (encoded as integers)
        y_pred: Current predictions (raw scores)
    
    Returns:
        grad: Gradient vector
        hess: Hessian vector
    """
    # Convert predictions to probabilities
    probs = softmax(y_pred)
    
    # Calculate data likelihood gradient
    grad_data = probs - one_hot(y_true)
    
    # Calculate prior gradient
    grad_prior = calculate_prior_gradient(probs, prior_probs)
    
    # Combine with lambda weighting
    grad = grad_data + lambda_param * grad_prior
    
    # Calculate hessian
    hess = probs * (1 - probs) + lambda_param * prior_hessian
    
    return grad, hess
```

### 4. Configuration Extension
```python
class XGBoostPriorConfig(XGBoostConfig):
    """Configuration for prior-enhanced XGBoost"""
    use_medical_priors: bool = True
    prior_method: Literal["custom_objective", "feature_engineering", "both"] = "custom_objective"
    lambda_prior: float = 0.1  # Weight for prior term
    prior_data_path: Optional[str] = None
    feature_prior_weight: float = 1.0  # Weight for prior features
```

### 5. Testing Strategy
- **Unit Tests**: Test each component in isolation
- **Integration Tests**: Test full pipeline with sample data
- **Performance Tests**: Compare speed and memory vs vanilla
- **Validation Tests**: Run cross-site experiments from IM-035
- **Medical Plausibility**: Verify predictions align with priors

## Implementation Steps

### Phase 1: Prior Data Infrastructure (Day 1)
1. Create medical_priors module structure
2. Research and document InSilicoVA prior data format
3. Implement prior data loader with validation
4. Create prior calculator for computing probabilities
5. Write comprehensive unit tests

### Phase 2: Custom Objective (Day 2-3)
1. Study XGBoost custom objective API
2. Implement gradient and hessian calculations
3. Create prior-informed objective function
4. Integrate with XGBoostPriorEnhanced class
5. Test convergence and stability

### Phase 3: Feature Engineering (Day 4)
1. Implement prior probability feature extraction
2. Create symptom-cause association features
3. Design medical ratio features
4. Integrate with preprocessing pipeline
5. Test feature importance

### Phase 4: Integration and Validation (Day 5)
1. Complete XGBoostPriorEnhanced implementation
2. Run cross-site validation experiments
3. Compare results with baseline
4. Document performance improvements
5. Create usage examples

### Phase 5: Documentation and Cleanup (Day 5.5)
1. Write comprehensive documentation
2. Create parameter tuning guide
3. Add examples to README
4. Clean up code and ensure style compliance
5. Prepare PR with results

## Validation Criteria

### Code Validation
```bash
# Run unit tests
poetry run pytest baseline/models/tests/test_xgboost_prior_enhanced.py -v

# Check coverage
poetry run pytest --cov=baseline.models.xgboost_prior_enhanced --cov-report=html

# Run linting
poetry run ruff check baseline/models/xgboost_prior_enhanced.py

# Type checking
poetry run mypy baseline/models/xgboost_prior_enhanced.py
```

### Performance Validation
```python
# Run cross-site experiments
poetry run python model_comparison/scripts/run_va34_comparison.py \
    --models xgboost xgboost_prior_enhanced insilico \
    --sites AP Bohol Dar Mexico Pemba UP \
    --output-dir results/prior_enhanced_comparison
```

### Expected Results
- Cross-site generalization gap reduced by 15-25%
- In-domain performance maintained within 2%
- Computational overhead under 20%
- All tests passing with >90% coverage

## Technical Specifications

### API Design
```python
class XGBoostPriorEnhanced(XGBoostModel):
    """XGBoost with medical prior integration"""
    
    def __init__(self, config: XGBoostPriorConfig):
        super().__init__(config)
        self.config = config
        self.priors = None
        
    def fit(self, X, y, sample_weight=None):
        """Fit model with prior integration"""
        # Load priors if not already loaded
        if self.priors is None:
            self.priors = self._load_medical_priors()
        
        # Apply feature engineering if enabled
        if self.config.prior_method in ["feature_engineering", "both"]:
            X = self._augment_with_prior_features(X)
        
        # Use custom objective if enabled
        if self.config.prior_method in ["custom_objective", "both"]:
            self._setup_custom_objective()
        
        # Train model
        return super().fit(X, y, sample_weight)
```

### Data Flow
```
Input Features (X) → 
    ↓
Prior Feature Augmentation (optional) →
    ↓
XGBoost Training →
    ↓ (with custom objective)
Prior-Constrained Model →
    ↓
Predictions aligned with medical knowledge
```

### Error Handling
- Graceful fallback if prior data unavailable
- Validation of prior data format
- Warning for implausible predictions
- Clear error messages for configuration issues

## Dependencies and Constraints

### External Dependencies
- InSilicoVA prior data (from GitHub repository)
- XGBoost >= 1.6.0 (for custom objective support)
- NumPy, Pandas for data manipulation
- SciPy for probability calculations

### Constraints
- Must maintain backward compatibility
- Cannot modify existing XGBoost model
- Prior data format is fixed by InSilicoVA
- Performance overhead must be minimal

### Assumptions
- Prior probabilities from InSilicoVA are medically valid
- Custom objective will converge properly
- Feature engineering won't cause overfitting
- Cross-site improvement will be measurable

## Risk Mitigation

### Technical Risks
1. **Custom objective instability**
   - Mitigation: Extensive testing, gradual lambda scheduling
   
2. **Prior data incompatibility**
   - Mitigation: Robust parsing, format validation
   
3. **Performance degradation**
   - Mitigation: Profiling, optimization, caching

### Implementation Risks
1. **Complexity creep**
   - Mitigation: Start simple, iterative enhancement
   
2. **Testing coverage**
   - Mitigation: Test-driven development
   
3. **Integration issues**
   - Mitigation: Early integration testing

## Success Metrics

### Primary Metrics
- Cross-site generalization gap reduction: 15-25%
- In-domain performance maintained: ±2%
- Computational overhead: <20%

### Secondary Metrics
- Code coverage: >90%
- Medical plausibility score: >0.8
- Feature importance of prior features: >0.1

### Qualitative Metrics
- Code clarity and maintainability
- Documentation completeness
- Ease of parameter tuning

## Example Usage

```python
from baseline.models.xgboost_prior_enhanced import XGBoostPriorEnhanced
from baseline.models.configs import XGBoostPriorConfig

# Configure prior-enhanced model
config = XGBoostPriorConfig(
    n_estimators=100,
    max_depth=6,
    use_medical_priors=True,
    prior_method="both",  # Use both objective and features
    lambda_prior=0.15,
    random_state=42
)

# Initialize and train
model = XGBoostPriorEnhanced(config)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Analyze prior influence
prior_influence = model.get_prior_influence_report()
print(f"Average prior contribution: {prior_influence['avg_contribution']:.2%}")
```

## Appendix

### A. InSilicoVA Prior Structure Reference
Based on InSilicoVA documentation, the prior data includes:
- Conditional probability tables (P(symptom|cause))
- Population-level cause distributions
- Expert-informed constraints
- Hierarchical regularization parameters

### B. XGBoost Custom Objective API
XGBoost supports custom objectives through:
- Gradient and Hessian calculation functions
- Integration with tree building algorithm
- Support for multi-class objectives
- Compatibility with existing features

### C. Related Work
- InSilicoVA paper (McCormick et al., JASA)
- XGBoost custom objective documentation
- Domain adaptation literature
- Medical AI constraint learning

This PRP provides comprehensive guidance for implementing prior-enhanced XGBoost that incorporates InSilicoVA's medical knowledge to improve cross-site generalization while maintaining computational efficiency.