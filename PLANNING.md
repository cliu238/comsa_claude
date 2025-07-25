# Project Planning & Architecture

## Overview

This project demonstrates Context Engineering - a systematic approach to providing AI coding assistants with comprehensive context for successful implementation. It combines structured requirements (PRPs), validation loops, and example-driven development to achieve high-quality, one-pass implementations. At its core, this framework is designed to systematically analyze Verbal Autopsy data using both modern AI/ML approaches and classical VA algorithms.

## Project Purpose

This project aims to systematically analyze Verbal Autopsy (VA) data by combining modern machine learning and AI methods with established classical VA algorithms. Our goal is to:

- **Provide a unified framework** for processing VA data through multiple analytical approaches
- **Compare performance** between traditional algorithms (InSilicoVA, InterVA, openVA) and modern ML/AI techniques
- **Enable researchers** to leverage both proven statistical methods and cutting-edge AI for cause-of-death determination
- **Create reproducible pipelines** that can handle diverse VA datasets with consistent methodology
- **Advance VA methodology** by facilitating systematic comparison and evaluation of different approaches

The framework serves as both a practical tool for VA analysis and a research platform for developing and testing new algorithms in the field of verbal autopsy.

## Core Architecture

### 1. Context Engineering Framework

```
┌─────────────────────────────────────────────────────────────┐
│                     CONTEXT ENGINEERING                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INITIAL.md  ──┐                                             │
│                ├──> /generate-prp ──> PRP Document           │
│  Examples/     ┘                           │                 │
│                                           ▼                  │
│                                    /execute-prp              │
│                                           │                  │
│                                           ▼                  │
│                                  ┌─────────────────┐         │
│                                  │ Implementation  │         │
│                                  │   with Tests    │         │
│                                  └─────────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Component Architecture

#### Claude Commands (.claude/commands/)
- **generate-prp.md**: Researches codebase, gathers context, creates comprehensive PRPs
- **execute-prp.md**: Reads PRPs, implements features with validation loops

#### PRP System (PRPs/)
- **templates/prp_base.md**: Base template ensuring all context is captured
- **Generated PRPs**: Complete implementation blueprints with context, gotchas, and validation

#### Examples (examples/)
- Reference implementations that AI can learn from
- Critical for pattern matching and consistency
- Should cover common patterns in your domain

#### Project Rules (CLAUDE.md)
- Global rules that apply to every AI interaction
- Coding standards, conventions, and constraints
- Updated as project patterns emerge

### 3. Implementation Modules

#### Baseline Module (baseline/)
First implementation demonstrating the framework:
```
baseline/
├── config/          # Pydantic-based configuration
├── data/            # Core processing logic
├── data_splitter.py # Site-based and train/test splitting
├── models/          # ML model implementations
│   └── insilico_model.py # InSilicoVA model module
├── utils/           # Validation and utility functions
├── example_usage.py # Demonstration script
└── tests/           # Comprehensive unit tests
```

#### Future Modules (Planned)
- **transfer_learning/**: Cross-dataset model adaptation
- **active_learning/**: Intelligent sample selection
- **model_comparison/**: Systematic algorithm evaluation with InSilicoVA baseline

#### Current Development
- **Task IM-052**: Fix Bootstrap Confidence Intervals
  - Fix bootstrap CI calculation in model comparison framework
  - Update metrics to return [lower, upper] format
  - Ensure proper statistical validation with 100-1000 iterations
  - Critical for model comparison statistical significance

- **Task IM-053**: Implement Hyperparameter Tuning for All ML Models
  - Priority: High (10-30% performance improvement expected)
  - Dependencies: IM-045 (XGBoost ✅), IM-046 (Random Forest ✅), IM-047 (Logistic Regression ✅)
  - Integrate with existing Ray infrastructure for distributed tuning
  - Support multiple optimization methods (Grid, Random, Bayesian, Optuna)
  - Create model_comparison/hyperparameter_tuning/ module
  - Extend ExperimentConfig with search space definitions

## Design Principles

### 1. Context is King
- More context = better implementations
- Include documentation, examples, gotchas
- Explicit > Implicit

### 2. Validation-Driven Development
- Every PRP includes executable validation
- AI iterates until all checks pass
- Tests are part of the implementation

### 3. Example-Based Learning
- AI performs best with concrete examples
- Show patterns to follow, not just requirements
- Include both good and bad examples

### 4. Progressive Enhancement
- Start simple, validate, then enhance
- Each module builds on previous patterns
- Complexity emerges from simple rules

## Data Flow Architecture

### VA Processing Pipeline Example

This pipeline demonstrates our dual approach of combining classical VA algorithms with modern ML methods:

```
Raw PHMRC CSV (Verbal Autopsy Data)
     │
     ▼
PHMRCData Validation (va-data submodule)
     │
     ▼
OpenVA Transformation
     │
     ├─────────────┬─────────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼             ▼
Numeric Encoding  OpenVA Format  Site          Transfer       Active
(for ML/AI)      (for Classic)  Stratification Learning      Learning
     │             │             │             │             │
     ▼             ▼             ▼             ▼             ▼
Modern ML/AI   Classical VA   Site-Specific  Domain        Sample
- Random Forest - InSilicoVA   Analysis       Adaptation    Selection
- XGBoost      - InterVA                     - Source/Target - Uncertainty
- SVM          - openVA                       - Fine-tuning  - Query Strategy
- Logistic Reg.
     │             │             │             │             │
     └─────────────┴─────────────┴─────────────┴─────────────┘
                                │
                                ▼
                    Model Comparison Framework
                    - CSMF Accuracy Metrics
                    - COD Agreement Analysis
                    - Performance Benchmarking
                    - Statistical Significance Testing
```

## Integration Strategy

### 1. Submodule Integration
- va-data integrated as git submodule
- Maintains separate versioning
- Clear dependency boundaries

### 2. Configuration Management
- Pydantic models for type safety
- Environment-specific settings
- Validation at boundaries

### 3. Output Management
- Timestamped outputs prevent overwrites
- Metadata tracking for reproducibility
- Separate directories by processing type

## Future Architecture Enhancements

### 1. Pipeline Orchestration
- Integrate with Prefect/Airflow
- Enable distributed processing
- Add checkpoint/resume capabilities

### 2. Model Registry
- Track trained models
- Version model artifacts
- Enable A/B testing

### 3. Data Versioning
- DVC integration for large files
- Track data lineage
- Enable reproducible experiments

### 4. API Layer
- REST/GraphQL endpoints
- Real-time predictions
- Batch processing queues

## Security & Best Practices

### 1. Code Security
- No hardcoded credentials
- Environment variable usage
- Secure data handling

### 2. Data Privacy
- PII handling protocols
- Data anonymization
- Access control patterns

### 3. Performance
- Lazy loading for large datasets
- Efficient memory usage
- Progress tracking for long operations

## Success Metrics

### Framework Success
- One-pass implementation rate
- Test coverage achieved
- Time to working implementation

### Module Success
- Processing accuracy
- Performance benchmarks
- Code maintainability scores

## Lessons Learned

### What Works Well
1. Comprehensive PRPs reduce implementation errors
2. Validation loops catch issues early
3. Examples significantly improve output quality

### Areas for Improvement
1. Need better handling of external API documentation
2. Site stratification requires custom xform handling
3. Long-running processes need checkpoint mechanisms
4. Data splitting for ML pipelines needs simple, robust implementation

## Contributing Guidelines

### Adding New Modules
1. Start with INITIAL.md describing the feature
2. Generate PRP using /generate-prp
3. Review and enhance PRP if needed
4. Execute with /execute-prp
5. Document in README.md

### Improving Framework
1. Enhance PRP templates based on patterns
2. Add more comprehensive examples
3. Update CLAUDE.md with discovered conventions
4. Share successful PRPs as examples