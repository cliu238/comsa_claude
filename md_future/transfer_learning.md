# Transfer Learning

## Overview
This module implements domain adaptation techniques to transfer knowledge from PHMRC-trained models to new target domains (WHO-2016 VA files, MITS, COMSA). It addresses the challenge of applying models trained on one VA standard to different VA instruments and populations.

## Source and Target Domains
### Source Domain
- **Dataset**: PHMRC gold standard (same as baseline)
- **VA Standard**: PHMRC 2013
- **Pre-trained models**: Best performing models from baseline benchmark

### Target Domains
1. **COMSA Data**
   - Files: `data/raw/COMSA/all_WHO_wgt.csv`, `data/raw/COMSA/all_WHO_with_age.csv`
   - VA Standard: WHO-2016
   
2. **MITS Data**
   - Files: Multiple VA files in `data/raw/MITS/`
   - Mixed standards requiring mapping

3. **WHO Standard Data**
   - Files: `data/raw/who/all_WHO.csv` and variants
   - WHO-2016 standard

## Transfer Learning Techniques

### 1. ADAPT Library Methods
**Library**: https://github.com/adapt-python/adapt

#### Instance-based Methods
- **TrAdaBoost**: Boosting-based transfer learning
- **KLIEP**: Kullback-Leibler Importance Estimation Procedure
- **KMM**: Kernel Mean Matching

#### Feature-based Methods
- **FA (Feature Augmentation)**: Augments features for domain adaptation
- **CORAL**: Correlation Alignment
- **SubspaceLearning**: Finds common subspace between domains

#### Parameter-based Methods
- **RegularTransferLR**: Regularized transfer for logistic regression
- **FineTuning**: Fine-tune pre-trained models on target data

### 2. Tabular Representation Learning
**Library**: TransTab (https://github.com/RyanWangZf/transtab)
- Pre-trained tabular transformers
- Feature embeddings for heterogeneous data
- Zero-shot and few-shot learning capabilities

## Implementation Strategy

### Phase 1: Data Alignment
1. **Feature Mapping**:
   - Map PHMRC features to WHO-2016 standard
   - Handle missing/new features
   - Create alignment dictionaries
   
2. **Cause Mapping**:
   - Map PHMRC causes to WHO/ICD-10 causes
   - Handle cause aggregation/disaggregation
   - Document unmappable causes

### Phase 2: Domain Adaptation
1. **Unsupervised Methods** (no target labels):
   - CORAL for distribution alignment
   - Subspace methods for feature transformation
   - Instance reweighting (KMM, KLIEP)
   
2. **Semi-supervised Methods** (few target labels):
   - TrAdaBoost with small labeled target set
   - Self-training with confidence thresholds
   - Co-training with multiple views

3. **Supervised Methods** (labeled target data):
   - Fine-tuning baseline models
   - Regular transfer learning
   - Ensemble methods

### Phase 3: Representation Learning
1. **TransTab Implementation**:
   - Pre-train on combined VA datasets
   - Extract learned representations
   - Use for downstream tasks

## Evaluation Protocol
1. **Target Domain Splits**:
   - Reserve 20% of target data for testing
   - Use varying amounts (10%, 25%, 50%) for adaptation
   
2. **Metrics**:
   - Same as baseline (CSMF, Top-1/3, COD5)
   - Transfer improvement ratio
   - Domain discrepancy measures
   
3. **Ablation Studies**:
   - Compare zero-shot vs few-shot performance
   - Impact of feature alignment strategies
   - Effect of source model selection

## Code Structure
```
transfer/
├── __init__.py
├── data_alignment.py        # Feature and cause mapping
├── domain_adaptation.py     # ADAPT method implementations
├── representation.py        # TransTab integration
├── evaluation.py           # Transfer-specific metrics
├── experiments.py          # Experiment orchestration
└── config.py              # Transfer learning settings
```

## Implementation Steps
1. Load pre-trained baseline models
2. Prepare target domain data with alignment
3. Apply unsupervised adaptation methods
4. If labels available, apply supervised methods
5. Implement TransTab representations
6. Evaluate on held-out target test sets
7. Compare adaptation methods
8. Export results to `results/transfer/transfer_results.csv`

## Expected Deliverables
- `transfer_results.csv` with columns:
  - Source model
  - Target dataset
  - Adaptation method
  - Label percentage used
  - CSMF accuracy
  - Top-1/3 accuracy
  - COD5 accuracy
  - Improvement over baseline
  
- `feature_mappings.json`: Documented feature alignments
- `cause_mappings.json`: Cause correspondence tables

## Key Considerations
- **Distribution Shift**: Account for demographic differences
- **Label Noise**: Target labels may be less reliable
- **Feature Mismatch**: Not all features map 1-to-1
- **Computational Cost**: Some methods are expensive
- **Sample Size**: Target domains may have limited data

## Dependencies
- adapt-python
- transtab
- scikit-learn
- torch (for TransTab)
- pandas
- numpy