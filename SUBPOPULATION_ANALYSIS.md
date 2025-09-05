# InSilicoVA Subpopulation Experiment Analysis

## Assessment of Issues #34 and #35

Based on thorough analysis of the codebase and Issue #32 (the epic), Issues #34 and #35 make partial sense but need refinement.

## Current Infrastructure Analysis

### ✅ What Already Exists (Can Be Reused)

#### 1. **Data Splitting Infrastructure (Issue #34)**
- **File:** `baseline/data/data_splitter.py`
- **Existing Features:**
  - `VADataSplitter` class with multiple splitting strategies
  - `_cross_site_split()` method for site-based splits
  - `_stratified_site_split()` for stratified sampling within sites
  - Site validation and metadata tracking

**What's Missing:**
- The specific 100% source + 50% target splitting logic needed for subpopulation experiments
- Subpopulation label generation (site indicators for InSilicoVA)

#### 2. **InSilicoVA Docker Wrapper (Issue #35)**
- **File:** `baseline/models/insilico_model.py`
- **Existing Features:**
  - Complete Docker wrapper implementation
  - R script generation with `codeVA` function
  - Data format conversion (Python ↔ R)
  - Docker lifecycle management
  - Error handling and validation

**What's Missing:**
- **Subpopulation parameter support** - The current implementation doesn't pass subpopulation labels to InSilicoVA
- The R script needs modification to accept `sub.pop` parameter

#### 3. **Experiment Orchestration**
- **File:** `model_comparison/scripts/run_distributed_comparison.py`
- **Existing Features:**
  - Parallel execution with Ray
  - Site-based experiments (in_domain, out_domain)
  - Checkpoint management
  - Result aggregation
  - Multiple experiment types

## Required Modifications

### For Issue #34 (Data Preparation)

**Minimal Changes Needed:**
```python
# Add to data_splitter.py
def _subpopulation_split(self, data: pd.DataFrame) -> SplitResult:
    """Special split for subpopulation experiments.
    100% source + 50% target for training, 50% target for testing.
    """
    source_site = self.config.source_site
    target_site = self.config.target_site
    
    # Get all source data
    source_data = data[data[self.config.site_column] == source_site]
    
    # Split target data 50/50
    target_data = data[data[self.config.site_column] == target_site]
    target_train, target_test = train_test_split(
        target_data, test_size=0.5, random_state=self.config.random_state
    )
    
    # Combine for training
    train_data = pd.concat([source_data, target_train])
    
    # Add subpopulation labels
    train_data['subpop'] = train_data[self.config.site_column]
    
    return SplitResult(train=train_data, test=target_test, metadata={...})
```

### For Issue #35 (InSilicoVA Wrapper)

**Minimal Changes Needed:**
```python
# Modify insilico_model.py
def fit(self, X: pd.DataFrame, y: pd.Series, subpop_labels: Optional[pd.Series] = None):
    """Extended fit method with subpopulation support."""
    # Store subpopulation labels if provided
    self.subpop_labels = subpop_labels
    # ... existing code ...

def _generate_r_script(self) -> str:
    """Generate R script with subpopulation support."""
    # ... existing code ...
    
    # Add subpopulation parameter if available
    if self.subpop_labels is not None:
        # Write subpop labels to CSV
        # Modify codeVA call to include sub.pop parameter
        r_script += """
        subpop_data <- read.csv("/data/subpop_labels.csv")
        results <- codeVA(
            ...,
            sub.pop = subpop_data$subpop,
            ...
        )
        """
```

## Recommendations

### 1. **Merge Issues #34 and #35**
These tasks are tightly coupled - the data preparation needs to know about subpopulation requirements, and the wrapper needs the prepared data format. Consider combining them into a single task.

### 2. **Leverage Existing Infrastructure**
- **Don't rewrite** the Docker wrapper - just extend it
- **Don't duplicate** splitting logic - add a new strategy to existing splitter
- **Use** the existing experiment orchestration from `run_distributed_comparison.py`

### 3. **Create a New Experiment Type**
Add `"subpopulation"` as a new experiment type alongside `"in_domain"`, `"out_domain"`, and `"training_size"`.

### 4. **Minimal Implementation Path**
```
1. Extend DataConfig to include subpopulation settings
2. Add _subpopulation_split() method to VADataSplitter
3. Modify InSilicoVAModel to accept optional subpop_labels
4. Update R script generation to include sub.pop parameter
5. Add "subpopulation" experiment type to run_distributed_comparison.py
```

## What Can Be Directly Reused

### From `run_distributed_comparison.py`:
- Ray parallel execution infrastructure
- Experiment configuration management
- Result aggregation and visualization
- Checkpoint/resume functionality
- Progress tracking

### From `baseline/models/insilico_model.py`:
- Docker container management (100% reusable)
- Data format conversion (100% reusable)
- Error handling and validation (100% reusable)
- R script template (95% reusable - just add sub.pop)

### From `baseline/data/data_splitter.py`:
- Base splitting infrastructure (100% reusable)
- Site validation logic (100% reusable)
- Split result management (100% reusable)
- Metadata tracking (100% reusable)

## Estimated Effort

Given the existing infrastructure:
- **Issue #34:** 2-3 hours (mostly adding new split strategy)
- **Issue #35:** 1-2 hours (just adding subpop parameter)
- **Integration:** 2-3 hours (connecting pieces and testing)

**Total:** ~6-8 hours instead of building from scratch

## Conclusion

Issues #34 and #35 make sense conceptually but significantly underestimate what's already available. The codebase already has 80% of what's needed - the tasks should focus on **extending** existing code rather than creating new implementations.