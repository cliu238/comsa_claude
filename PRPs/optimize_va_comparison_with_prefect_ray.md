name: "Optimize VA Comparison Scripts with Prefect and Ray"
description: |

## Purpose
Template for implementing parallel execution of VA model comparison experiments using Prefect workflows and Ray distributed computing to significantly reduce execution time and enable better scalability.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Transform the sequential VA34 model comparison script into a highly performant, distributed system using Prefect for workflow orchestration and Ray for parallel computation. The implementation should reduce execution time by at least 50% while maintaining reproducibility and adding comprehensive monitoring capabilities.

## Why
- **Performance**: Current sequential execution is too slow for large-scale experiments
- **Scalability**: Enable running larger experiments with more sites and models
- **Monitoring**: Real-time visibility into experiment progress and performance
- **Reliability**: Checkpoint/resume capability for long-running experiments
- **Resource Efficiency**: Better utilization of available compute resources

## What
The system will provide:
- Parallel execution of model training across sites and configurations
- Workflow orchestration with dependency management and error handling
- Real-time progress monitoring and performance metrics
- Checkpoint/resume functionality for interrupted experiments
- Distributed computing support for multi-core and multi-machine setups

### Success Criteria
- [ ] 50%+ reduction in execution time for VA34 comparison experiments
- [ ] Support for distributed execution across 4+ workers
- [ ] Checkpoint/resume functionality working reliably
- [ ] Real-time progress monitoring with tqdm integration
- [ ] All existing tests pass with new implementation
- [ ] Documentation updated with usage examples and configuration options

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://docs.prefect.io/latest/
  why: Prefect 2.0 workflow orchestration patterns, task dependencies, error handling
  
- url: https://docs.ray.io/en/latest/ray-core/walkthrough.html
  why: Ray core concepts, remote functions, actors, object store usage
  
- file: baseline/utils/logging_config.py
  why: Existing logging patterns to integrate with Prefect/Ray logging
  
- file: model_comparison/experiments/site_comparison.py
  why: Current implementation to parallelize, understand data flow and dependencies

- doc: https://docs.prefect.io/latest/concepts/flows/
  section: Flow run retries and error handling
  critical: Configure proper retry logic for transient failures

- doc: https://docs.ray.io/en/latest/ray-core/patterns/index.html
  section: Design patterns and anti-patterns
  critical: Avoid common pitfalls with distributed computing

```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
model_comparison/
├── __init__.py
├── experiments/
│   ├── __init__.py
│   ├── experiment_config.py
│   └── site_comparison.py
├── metrics/
│   ├── __init__.py
│   └── comparison_metrics.py
├── results/
├── scripts/
│   ├── __init__.py
│   └── run_va34_comparison.py
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   └── test_site_comparison.py
└── visualization/
    ├── __init__.py
    └── comparison_plots.py
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
model_comparison/
├── __init__.py
├── experiments/
│   ├── __init__.py
│   ├── experiment_config.py
│   ├── site_comparison.py  # Modified to support parallel execution
│   └── parallel_experiment.py  # New: Ray-based parallel experiment runner
├── metrics/
│   ├── __init__.py
│   └── comparison_metrics.py
├── orchestration/  # New directory
│   ├── __init__.py
│   ├── prefect_flows.py  # Prefect workflow definitions
│   ├── ray_tasks.py  # Ray remote functions and actors
│   └── checkpoint_manager.py  # Checkpoint save/restore logic
├── monitoring/  # New directory
│   ├── __init__.py
│   ├── progress_tracker.py  # Real-time progress monitoring
│   └── performance_monitor.py  # Performance metrics collection
├── results/
├── scripts/
│   ├── __init__.py
│   ├── run_va34_comparison.py  # Modified to use Prefect/Ray
│   └── run_distributed_comparison.py  # New: Distributed execution script
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_site_comparison.py
│   ├── test_parallel_experiment.py  # New tests
│   └── test_orchestration.py  # New tests
└── visualization/
    ├── __init__.py
    └── comparison_plots.py
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Ray requires serializable objects
# InSilicoVAModel uses Docker which may not be directly serializable
# Solution: Use Ray actors or custom serialization

# CRITICAL: Prefect 2.0 uses async by default
# Our models are sync - need proper async/sync bridging

# CRITICAL: VA data processing memory usage
# Large datasets may exceed Ray object store limits
# Solution: Batch processing and proper memory management

# CRITICAL: Model state management
# XGBoost and InSilico models have different state requirements
# Ensure proper initialization in distributed context

# CRITICAL: Logging in distributed environment
# Centralized logging required to avoid lost logs from workers
# Use Ray's logging integration with our existing setup
```

## Implementation Blueprint

### Data models and structure

Create the core data models and configuration for parallel execution.
```python
# orchestration/config.py
from pydantic import BaseModel, Field
from typing import Optional, List

class ParallelConfig(BaseModel):
    """Configuration for parallel execution."""
    n_workers: int = Field(default=-1, description="Number of Ray workers (-1 for auto)")
    memory_per_worker: Optional[str] = Field(default="4GB", description="Memory per worker")
    checkpoint_interval: int = Field(default=10, description="Checkpoint every N experiments")
    prefect_dashboard: bool = Field(default=True, description="Enable Prefect dashboard")
    ray_dashboard: bool = Field(default=True, description="Enable Ray dashboard")
    retry_attempts: int = Field(default=3, description="Retry failed tasks N times")
    
class CheckpointState(BaseModel):
    """State for checkpoint/resume functionality."""
    completed_experiments: List[str] = Field(default_factory=list)
    partial_results: Optional[dict] = None
    timestamp: str
    config_hash: str  # To ensure config compatibility
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1:
CREATE model_comparison/orchestration/__init__.py:
  - Empty init file for package

CREATE model_comparison/orchestration/config.py:
  - Define ParallelConfig and CheckpointState models
  - Add validation for resource limits
  - Include serialization methods for Ray

Task 2:
CREATE model_comparison/orchestration/ray_tasks.py:
  - MIRROR pattern from: baseline/models/xgboost_model.py for model wrapping
  - CREATE Ray remote functions for model training
  - IMPLEMENT memory-efficient data passing
  - ADD proper error handling and retries

Task 3:
CREATE model_comparison/orchestration/checkpoint_manager.py:
  - IMPLEMENT checkpoint saving with atomic writes
  - ADD checkpoint loading and validation
  - CREATE resume logic to skip completed experiments
  - PRESERVE existing result aggregation patterns

Task 4:
CREATE model_comparison/monitoring/progress_tracker.py:
  - INTEGRATE tqdm with Ray progress reporting
  - ADD real-time metrics collection
  - IMPLEMENT progress persistence for resume
  - MIRROR logging patterns from baseline/utils/logging_config.py

Task 5:
CREATE model_comparison/orchestration/prefect_flows.py:
  - DEFINE main experiment flow with task dependencies
  - ADD sub-flows for in-domain, out-domain, and size experiments
  - IMPLEMENT error handling and retry logic
  - INTEGRATE with Ray for parallel execution

Task 6:
CREATE model_comparison/experiments/parallel_experiment.py:
  - EXTEND SiteComparisonExperiment for parallel execution
  - MODIFY evaluation methods to be Ray-compatible
  - ADD batching logic for memory efficiency
  - PRESERVE existing metric calculation logic

Task 7:
MODIFY model_comparison/experiments/site_comparison.py:
  - ADD hooks for progress tracking
  - MODIFY to support checkpoint/resume
  - ENSURE backward compatibility
  - ADD parallel execution flag

Task 8:
CREATE model_comparison/scripts/run_distributed_comparison.py:
  - CREATE CLI interface for distributed execution
  - ADD Ray cluster initialization
  - IMPLEMENT Prefect flow execution
  - INCLUDE monitoring dashboard URLs

Task 9:
MODIFY model_comparison/scripts/run_va34_comparison.py:
  - ADD --parallel flag for opt-in parallel execution
  - PRESERVE existing sequential behavior by default
  - ADD parallel configuration options
  - MAINTAIN backward compatibility

Task 10:
CREATE tests for all new components:
  - Unit tests for Ray tasks
  - Integration tests for Prefect flows
  - Checkpoint/resume functionality tests
  - Performance benchmarks

```


### Per task pseudocode as needed added to each task
```python

# Task 2 - Ray Tasks Implementation
# Pseudocode for ray_tasks.py
import ray
from typing import Tuple, Dict
import pandas as pd

@ray.remote
def train_and_evaluate_model(
    model_name: str,
    train_data: Tuple[pd.DataFrame, pd.Series],
    test_data: Tuple[pd.DataFrame, pd.Series],
    experiment_metadata: dict
) -> Dict:
    # PATTERN: Import inside remote function for serialization
    from baseline.models.insilico_model import InSilicoVAModel
    from baseline.models.xgboost_model import XGBoostModel
    from ..metrics.comparison_metrics import calculate_metrics
    
    # GOTCHA: Initialize model inside remote function
    if model_name == "insilico":
        model = InSilicoVAModel()
    else:
        model = XGBoostModel()
    
    # PATTERN: Unpack data (passed as serialized objects)
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # CRITICAL: Handle potential OOM by monitoring memory
    # Ray will retry if worker crashes
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        n_bootstrap=experiment_metadata.get("n_bootstrap", 100)
    )
    
    # Return serializable result
    return {
        **experiment_metadata,
        **metrics
    }

# Task 5 - Prefect Flows
# Pseudocode for prefect_flows.py
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import ray

@task(retries=3, retry_delay_seconds=60)
async def run_single_experiment(experiment_config: dict) -> dict:
    # PATTERN: Bridge async Prefect with sync Ray
    # GOTCHA: Use ray.get() properly to avoid blocking
    
    # Submit work to Ray
    result_ref = train_and_evaluate_model.remote(**experiment_config)
    
    # Wait for result with timeout
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(ray.get, result_ref),
            timeout=300  # 5 minute timeout per experiment
        )
    except asyncio.TimeoutError:
        # Handle timeout - maybe retry with smaller bootstrap
        raise
    
    return result

@flow(
    name="VA34 Comparison Experiment",
    task_runner=ConcurrentTaskRunner(max_workers=10),
    persist_result=True
)
async def va34_comparison_flow(config: ExperimentConfig) -> pd.DataFrame:
    # PATTERN: Load data once, share via Ray object store
    data_ref = ray.put(load_and_prepare_data(config))
    
    # Generate all experiment configurations
    experiments = generate_experiment_configs(config, data_ref)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(config)
    if checkpoint:
        experiments = filter_completed_experiments(experiments, checkpoint)
    
    # Run experiments in parallel
    results = []
    with tqdm(total=len(experiments)) as pbar:
        # Batch experiments to avoid overwhelming Ray
        for batch in batch_experiments(experiments, batch_size=50):
            batch_results = await asyncio.gather(*[
                run_single_experiment(exp) for exp in batch
            ])
            results.extend(batch_results)
            
            # Update progress
            pbar.update(len(batch))
            
            # Save checkpoint periodically
            if len(results) % config.checkpoint_interval == 0:
                save_checkpoint(results, config)
    
    # Aggregate and save final results
    final_results = pd.DataFrame(results)
    save_results(final_results, config)
    
    return final_results

# Task 6 - Parallel Experiment Runner
# Key modifications to support parallelization
class ParallelSiteComparisonExperiment(SiteComparisonExperiment):
    def __init__(self, config: ExperimentConfig, parallel_config: ParallelConfig):
        super().__init__(config)
        self.parallel_config = parallel_config
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.parallel_config.n_workers,
                dashboard_host="0.0.0.0" if self.parallel_config.ray_dashboard else None
            )
    
    def _run_in_domain_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        # PATTERN: Generate configs, then execute in parallel
        experiment_configs = []
        
        for site in self.config.sites:
            site_data = data[data["site"] == site]
            if len(site_data) < 50:
                continue
                
            # Prepare data splits
            X_train, X_test, y_train, y_test = self._split_data(site_data)
            
            # Put data in Ray object store for efficient sharing
            train_ref = ray.put((X_train, y_train))
            test_ref = ray.put((X_test, y_test))
            
            for model_name in self.config.models:
                experiment_configs.append({
                    "model_name": model_name,
                    "train_data": train_ref,
                    "test_data": test_ref,
                    "experiment_metadata": {
                        "experiment_type": "in_domain",
                        "train_site": site,
                        "test_site": site,
                        "n_bootstrap": self.config.n_bootstrap
                    }
                })
        
        # Execute all experiments in parallel
        result_refs = [
            train_and_evaluate_model.remote(**config) 
            for config in experiment_configs
        ]
        
        # Gather results with progress bar
        results = []
        with tqdm(total=len(result_refs), desc="In-domain experiments") as pbar:
            while result_refs:
                # Wait for any task to complete
                ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
                results.extend(ray.get(ready_refs))
                pbar.update(1)
        
        return pd.DataFrame(results)
```

### Integration Points
```yaml
DATABASE:
  - No database changes required
  
CONFIG:
  - add to: pyproject.toml
  - dependencies: "prefect>=2.0", "ray[default]>=2.0", "tqdm>=4.65"
  
ENVIRONMENT:
  - add to: .env.example
  - variables: "RAY_WORKERS=4", "PREFECT_API_URL=http://localhost:4200"
  
LOGGING:
  - integrate with: baseline/utils/logging_config.py
  - pattern: Use existing get_logger() with component="orchestration"
  
CLI:
  - modify: model_comparison/scripts/run_va34_comparison.py
  - add flags: --parallel, --n-workers, --checkpoint-dir
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check model_comparison/orchestration/ --fix
ruff check model_comparison/monitoring/ --fix
mypy model_comparison/orchestration/
mypy model_comparison/monitoring/

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE test_ray_tasks.py with these test cases:
def test_train_and_evaluate_model():
    """Test Ray remote function works correctly"""
    ray.init(local_mode=True)  # Local mode for testing
    
    # Create small test data
    X_train = pd.DataFrame(np.random.rand(100, 10))
    y_train = pd.Series(np.random.choice(['A', 'B', 'C'], 100))
    X_test = pd.DataFrame(np.random.rand(20, 10))
    y_test = pd.Series(np.random.choice(['A', 'B', 'C'], 20))
    
    # Test remote execution
    result_ref = train_and_evaluate_model.remote(
        model_name="xgboost",
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        experiment_metadata={"experiment_type": "test"}
    )
    
    result = ray.get(result_ref)
    assert "csmf_accuracy" in result
    assert "cod_accuracy" in result
    ray.shutdown()

def test_checkpoint_save_and_load():
    """Test checkpoint functionality"""
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")
    
    # Save checkpoint
    test_results = [{"experiment": 1, "accuracy": 0.8}]
    manager.save_checkpoint(test_results, config_hash="test123")
    
    # Load checkpoint
    loaded = manager.load_checkpoint(config_hash="test123")
    assert loaded is not None
    assert len(loaded.completed_experiments) == 1

def test_prefect_flow_with_ray():
    """Test Prefect flow integrates with Ray"""
    # Use Prefect testing utilities
    from prefect.testing.utilities import prefect_test_harness
    
    with prefect_test_harness():
        # Run a minimal flow
        result = va34_comparison_flow(
            config=ExperimentConfig(
                data_path="test_data.csv",
                sites=["site1"],
                models=["xgboost"],
                training_sizes=[1.0],
                n_bootstrap=10
            )
        )
        assert isinstance(result, pd.DataFrame)
```

```bash
# Run and iterate until passing:
poetry run pytest model_comparison/tests/test_ray_tasks.py -v
poetry run pytest model_comparison/tests/test_orchestration.py -v

# Performance benchmark test
poetry run python -m pytest model_comparison/tests/test_parallel_performance.py -v --benchmark
```

### Level 3: Integration Test
```bash
# Start Ray cluster (local)
ray start --head --dashboard-host 0.0.0.0

# Start Prefect server
prefect server start

# Test distributed execution with small dataset
poetry run python model_comparison/scripts/run_distributed_comparison.py \
  --data-path data/test_subset.csv \
  --sites site_1 site_2 \
  --models xgboost \
  --n-workers 2 \
  --training-sizes 0.5 1.0 \
  --n-bootstrap 10

# Expected: 
# - Ray dashboard accessible at http://localhost:8265
# - Prefect dashboard at http://localhost:4200
# - Execution completes in < 50% of sequential time
# - Results saved with checkpoint files

# Cleanup
ray stop
```

## Final validation Checklist
- [ ] All tests pass: `poetry run pytest model_comparison/tests/ -v`
- [ ] No linting errors: `ruff check model_comparison/`
- [ ] No type errors: `mypy model_comparison/`
- [ ] Sequential mode still works: `python run_va34_comparison.py --data-path test.csv`
- [ ] Parallel mode reduces execution time by 50%+
- [ ] Checkpoint/resume works after interruption
- [ ] Ray and Prefect dashboards accessible
- [ ] Memory usage stays within limits
- [ ] Documentation updated with examples

---

## Anti-Patterns to Avoid
- ❌ Don't pass large data through Prefect - use Ray object store
- ❌ Don't create Ray actors for stateless operations - use remote functions
- ❌ Don't ignore Ray's memory limits - batch operations appropriately
- ❌ Don't mix async/sync carelessly - use proper bridges
- ❌ Don't hardcode cluster configuration - make it configurable
- ❌ Don't skip checkpoint validation - ensure resume compatibility