# Baseline Module: InSilicoVA Model Implementation

## FEATURE:

**Part of baseline module** - InSilicoVA model implementation for VA (Verbal Autopsy) cause-of-death prediction using existing baseline data pipeline components.

- Use baseline data loader, preprocessor, and splitter modules to load and prepare data
- Docker-based InSilicoVA implementation using openVA package with old JDK requirements
- Calculate predictions using InSilicoVA algorithm through Docker container
- CSMF (Cause-Specific Mortality Fraction) accuracy evaluation matching Table 3 results from "Probabilistic Cause-of-death Assignment using Verbal Autopsies" paper
- Modular design within baseline module for future ML model comparisons

## EXAMPLES:

In the `examples/` folder, there are reference implementations for InSilicoVA:

- `examples/insilico_model.py` - Docker-based InSilicoVA implementation (reference only)
- `examples/insilico_va_model.py` - Advanced InSilicoVA model with BaseModel inheritance (reference only)

**Note:** The code in `examples/` is for reference only and might not be correct for the current implementation.

## TECHNICAL REQUIREMENTS:

### Docker Integration:
- **Primary Docker Image**: `sha256:61df64731dec9b9e188b2b5a78000dd81e63caf78957257872b9ac1ba947efa4` (insilicova-arm64:latest)
- **Fallback**: Use `@Dockerfile` if the primary image doesn't work
- **Reason**: InSilicoVA requires very old JDK versions that are difficult to install directly

### Data Pipeline Integration:
- Use existing `baseline/config/data_config.py` for configuration
- Leverage `baseline/data/data_loader.py` for data loading
- Utilize `baseline/data/data_preprocessor.py` for preprocessing
- Apply `baseline/data/data_splitter.py` for train/test splitting

### Model Requirements:
- **Single Label Column**: Ensure only one label column is used for training
- **Column Dropping**: Identify and drop non-feature columns before training (site, ID, etc.)
- **OpenVA Compatibility**: Data must be formatted for openVA/InSilicoVA requirements

### Expected Results:
- **Target Performance**: Results should be close to Table 3 from "Probabilistic Cause-of-death Assignment using Verbal Autopsies" paper
- **CSMF Accuracy**: Match the InSilicoVA performance reported in the paper's Table 3
- **Reference Implementation**: The paper's R code for CSMF accuracy calculation:
  ```r
  csmf_true <- table(c(test$gs_text34, unique(PHMRC_adult$gs_text34))) - 1
  csmf_true <- csmf_true / sum(csmf_true)
  c(getCSMF_accuracy(csmf_inter, csmf_true, undet = "Undetermined"), 
    getCSMF_accuracy(csmf_ins[, "Mean"], csmf_true), 
    getCSMF_accuracy(csmf_nbc, csmf_true), 
    getCSMF_accuracy(csmf_tariff, csmf_true))
  [1] 0.53 0.74 0.77 0.68
  ```
- **Target Metric**: InSilicoVA should achieve approximately 0.74 CSMF accuracy as shown in the results

## DOCUMENTATION:

### Research References:
- **Primary Reference**: "Probabilistic Cause-of-death Assignment using Verbal Autopsies" paper (located in `@doc/`)
  - Table 3 contains the target CSMF accuracy benchmarks
  - InSilicoVA target performance: 0.74 CSMF accuracy
- **Official InSilicoVA Repository**: https://github.com/verbal-autopsy-software/InSilicoVA
- **InSilicoVA Usage Examples**: https://github.com/cliu238/InSilicoVA-sim (for implementation guidance)
- **OpenVA Documentation**: https://github.com/verbal-autopsy-software/openVA
- **Academic Paper**: https://journal.r-project.org/articles/RJ-2023-020/

### Implementation Guidelines:
- Follow KISS (Keep It Simple, Stupid) principle
- Avoid over-engineering the solution
- Focus on minimal viable implementation that meets requirements
- Use existing baseline module patterns and conventions

## ARCHITECTURE CONSIDERATIONS:

### Model Module Structure:
1. **InSilicoVA Model Class**: Standalone model following sklearn-like interface
   - `fit(X, y)`: Train model using Docker container
   - `predict(X)`: Generate cause-of-death predictions
   - `predict_proba(X)`: Get probability distributions
2. **Docker Integration**: Encapsulated Docker execution logic
3. **Model Comparison Ready**: Common interface for future ML model comparisons

### Data Flow:
1. **Baseline Data Pipeline**: Use existing baseline modules (loader, preprocessor, splitter)
2. **Model Training**: InSilicoVA model executes via Docker container
3. **Prediction**: Model generates predictions and probabilities
4. **Evaluation**: Calculate CSMF accuracy and comparison metrics

### Configuration:
- Model-specific parameters (nsim, jump_scale, etc.)
- Docker configuration options (image, platform, etc.)
- Separate from data pipeline configuration

### Error Handling:
- Robust Docker execution with proper error handling
- Fallback mechanisms for Docker failures
- Comprehensive logging for debugging

## IMPLEMENTATION CONSTRAINTS:

**IMPORTANT: Keep logic simple and straightforward. Avoid over-engineering the solution.**

**CRITICAL: Prioritize simplicity over complexity - implement the minimum viable solution that meets requirements.**

### Key Implementation Notes:
1. **Sklearn-like Interface**: Model should follow scikit-learn conventions for future comparisons
2. **Label Column**: Must identify and use only one label column for training
3. **Feature Selection**: Automatically drop non-feature columns (site, ID, metadata)
4. **Docker Isolation**: All InSilicoVA execution must happen in Docker containers
5. **Memory Management**: Handle large datasets efficiently
6. **Reproducibility**: Ensure consistent results with proper random seeding

### Module Location:
- Create `baseline/models/` directory for model implementations
- Main module: `baseline/models/insilico_model.py`
- Support files: Docker scripts, utilities as needed

### Testing Requirements:
- Unit tests for model class methods (fit, predict, predict_proba)
- Docker container execution tests
- CSMF accuracy validation tests
- Model interface compliance tests

**Note**: This feature creates a standalone model module that can be used with the existing baseline data pipeline and will serve as a foundation for future ML model comparisons.