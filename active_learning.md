# Active Learning

## Overview
This module implements active learning strategies to efficiently improve model performance by intelligently selecting the most informative samples for expert annotation. It's designed to maximize model improvement while minimizing annotation costs for verbal autopsy cause assignment.

## Active Learning Framework

### Core Components
1. **Unlabeled Pool**: Large set of VA records without gold-standard labels
2. **Labeled Set**: Initially small set of labeled examples
3. **Query Strategy**: Algorithm to select most informative samples
4. **Oracle**: Expert annotator (physician/VA specialist)
5. **Model Update**: Retrain with newly labeled data

### Query Strategies

#### 1. Uncertainty Sampling
- **Least Confidence**: Select samples with lowest prediction confidence
- **Margin Sampling**: Select samples with smallest margin between top two predictions
- **Entropy Sampling**: Select samples with highest prediction entropy
```python
entropy = -sum(p * log(p) for p in probabilities)
```

#### 2. Diversity-based Sampling
- **Cluster-based**: Ensure selected samples cover different regions
- **Representative Sampling**: Select samples that represent unlabeled pool
- **Outlier Detection**: Identify unusual cases for annotation

#### 3. Committee-based Methods
- **Query-by-Committee**: Use ensemble disagreement
- **Vote Entropy**: Measure disagreement across multiple models
- **Bayesian Active Learning**: Use posterior uncertainty

#### 4. Expected Model Change
- **Expected Gradient Length (EGL)**: Select samples that would most change the model
- **Expected Error Reduction**: Estimate impact on overall error
- **Information Gain**: Maximize mutual information

## Implementation Architecture

### Batch Active Learning
Instead of selecting one sample at a time:
1. Select batches of K samples per iteration
2. Balance informativeness and diversity
3. Reduce retraining overhead

### Multi-class Considerations
- Handle class imbalance in cause-of-death distribution
- Ensure rare causes are adequately sampled
- Use class-weighted uncertainty measures

## Experimental Protocol

### 1. Initial Setup
```python
# Start with small labeled set (e.g., 100 samples)
initial_size = 100
batch_size = 20
max_iterations = 50
```

### 2. Active Learning Loop
```python
for iteration in range(max_iterations):
    # Train model on current labeled set
    model.fit(X_labeled, y_labeled)
    
    # Select batch of samples using query strategy
    query_indices = query_strategy(model, X_unlabeled, batch_size)
    
    # Simulate oracle annotation
    new_labels = oracle.annotate(X_unlabeled[query_indices])
    
    # Update labeled/unlabeled sets
    update_sets(query_indices, new_labels)
    
    # Evaluate performance
    metrics = evaluate(model, X_test, y_test)
```

### 3. Baseline Comparisons
- **Random Sampling**: Select samples uniformly at random
- **Stratified Sampling**: Maintain class distribution
- **Full Supervision**: Upper bound with all labels

## Evaluation Metrics

### 1. Learning Curves
- Plot performance vs number of labeled samples
- Compare active learning to random baseline
- Calculate area under learning curve (ALC)

### 2. Annotation Efficiency
- **Label Efficiency**: Performance achieved with X% of labels
- **Annotation Budget**: Labels needed for target performance
- **Cost-Performance Trade-off**: ROI analysis

### 3. Class-specific Metrics
- Per-cause accuracy improvements
- Impact on rare cause detection
- CSMF accuracy progression

## Practical Considerations

### 1. Cold Start Problem
- Initial model may be poor with few labels
- Use transfer learning from PHMRC as warm start
- Consider hybrid strategies early in process

### 2. Annotation Cost Model
- **Uniform Cost**: All samples equally expensive
- **Variable Cost**: Some cases harder to diagnose
- **Batch Discounts**: Annotating similar cases together

### 3. Stopping Criteria
- Performance plateau detection
- Annotation budget exhausted
- Target performance achieved

## Code Structure
```
active/
├── __init__.py
├── query_strategies.py      # Uncertainty, diversity, committee methods
├── active_learner.py        # Main AL loop implementation
├── oracle_simulator.py      # Simulate expert annotation
├── evaluation.py            # AL-specific metrics
├── visualization.py         # Learning curves, sample analysis
└── config.py               # Active learning settings
```

## Implementation Steps
1. Set up initial labeled/unlabeled splits
2. Implement multiple query strategies
3. Create active learning experiment loop
4. Run experiments with different:
   - Query strategies
   - Batch sizes
   - Initial set sizes
   - Model architectures
5. Generate learning curves
6. Analyze selected sample characteristics
7. Export results to `results/active/active_learning_results.csv`

## Expected Deliverables
- `active_learning_results.csv` with columns:
  - Query strategy
  - Iteration number
  - Samples labeled
  - CSMF accuracy
  - Top-1/3 accuracy
  - COD5 accuracy
  - Time elapsed
  
- `learning_curves.png`: Visualization of AL performance
- `selected_samples_analysis.json`: Characteristics of queried samples
- `strategy_comparison.csv`: Head-to-head strategy comparison

## Advanced Extensions

### 1. Multi-modal Active Learning
- Combine VA questionnaire with other data sources
- Joint selection across modalities

### 2. Active Transfer Learning
- Combine with transfer learning for new domains
- Select samples that improve adaptation

### 3. Human-in-the-loop Considerations
- Interface design for efficient annotation
- Annotator fatigue modeling
- Quality control mechanisms

## Dependencies
- scikit-learn
- modAL (active learning library)
- numpy
- pandas
- matplotlib/seaborn (visualization)
- scipy (statistical analysis)