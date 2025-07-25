"""Prefect flows for orchestrating VA model comparison experiments."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import ray
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from baseline.utils import get_logger
from model_comparison.experiments.experiment_config import ExperimentConfig
from model_comparison.monitoring.progress_tracker import (
    PerformanceMonitor,
    RayProgressTracker,
)
from model_comparison.orchestration.checkpoint_manager import CheckpointManager
from model_comparison.orchestration.config import (
    ExperimentResult,
    ParallelConfig,
)
from model_comparison.orchestration.ray_tasks import (
    ProgressReporter,
    prepare_data_for_site,
    train_and_evaluate_model,
)

logger = get_logger(__name__, component="orchestration")


@task(retries=3, retry_delay_seconds=60)
async def run_single_experiment(experiment_config: Dict) -> ExperimentResult:
    """Run a single experiment with retry logic.

    Args:
        experiment_config: Configuration for the experiment

    Returns:
        ExperimentResult from the experiment
    """
    # Submit work to Ray
    result_ref = train_and_evaluate_model.remote(**experiment_config)

    # Wait for result with timeout
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(ray.get, result_ref),
            timeout=experiment_config.get("timeout", 300),
        )
        return result
    except asyncio.TimeoutError:
        logger.error(
            f"Experiment timed out: {experiment_config['experiment_metadata']['experiment_id']}"
        )
        raise


@task
def generate_experiment_configs(
    config: ExperimentConfig,
    data_ref: ray.ObjectRef,
    checkpoint_manager: CheckpointManager,
) -> List[Dict]:
    """Generate all experiment configurations.

    Args:
        config: Main experiment configuration
        data_ref: Ray object reference to data
        checkpoint_manager: Checkpoint manager for experiment IDs

    Returns:
        List of experiment configurations
    """
    experiments = []

    # In-domain experiments
    for site in config.sites:
        for model_name in config.models:
            experiment_id = checkpoint_manager.create_experiment_id(
                model_name=model_name,
                experiment_type="in_domain",
                train_site=site,
                test_site=site,
            )

            experiments.append(
                {
                    "model_name": model_name,
                    "site": site,
                    "experiment_type": "in_domain",
                    "data_ref": data_ref,
                    "experiment_metadata": {
                        "experiment_id": experiment_id,
                        "experiment_type": "in_domain",
                        "train_site": site,
                        "test_site": site,
                        "n_bootstrap": config.n_bootstrap,
                    },
                }
            )

    # Out-domain experiments
    for train_site in config.sites:
        for test_site in config.sites:
            if train_site == test_site:
                continue

            for model_name in config.models:
                experiment_id = checkpoint_manager.create_experiment_id(
                    model_name=model_name,
                    experiment_type="out_domain",
                    train_site=train_site,
                    test_site=test_site,
                )

                experiments.append(
                    {
                        "model_name": model_name,
                        "train_site": train_site,
                        "test_site": test_site,
                        "experiment_type": "out_domain",
                        "data_ref": data_ref,
                        "experiment_metadata": {
                            "experiment_id": experiment_id,
                            "experiment_type": "out_domain",
                            "train_site": train_site,
                            "test_site": test_site,
                            "n_bootstrap": config.n_bootstrap,
                        },
                    }
                )

    # Training size experiments
    primary_site = config.sites[0] if config.sites else None
    if primary_site:
        for training_size in config.training_sizes:
            for model_name in config.models:
                experiment_id = checkpoint_manager.create_experiment_id(
                    model_name=model_name,
                    experiment_type="training_size",
                    train_site=primary_site,
                    test_site=primary_site,
                    training_size=training_size,
                )

                experiments.append(
                    {
                        "model_name": model_name,
                        "site": primary_site,
                        "training_size": training_size,
                        "experiment_type": "training_size",
                        "data_ref": data_ref,
                        "experiment_metadata": {
                            "experiment_id": experiment_id,
                            "experiment_type": "training_size",
                            "train_site": primary_site,
                            "test_site": primary_site,
                            "training_size": training_size,
                            "n_bootstrap": config.n_bootstrap,
                        },
                    }
                )

    logger.info(f"Generated {len(experiments)} experiment configurations")
    return experiments


@task
def prepare_experiment_data(
    experiments: List[Dict], data: pd.DataFrame, config: ExperimentConfig, data_openva_ref=None
) -> List[Dict]:
    """Prepare data for experiments using Ray.

    Args:
        experiments: List of experiment configurations
        data: Full dataset (numeric encoding for ML models)
        config: Experiment configuration
        data_openva_ref: Ray object reference to OpenVA encoded data (for InSilico)

    Returns:
        Updated experiment configurations with data references
    """
    # Get OpenVA data from Ray if provided
    data_openva = ray.get(data_openva_ref) if data_openva_ref else None
    
    # Prepare data splits in parallel
    site_data_refs = {}
    site_data_refs_openva = {}
    sites_to_prepare = list(set(exp.get("site", exp.get("train_site")) for exp in experiments))

    # Submit data preparation tasks for both data formats
    prep_tasks = {}
    prep_tasks_openva = {}
    for site in sites_to_prepare:
        if site and site not in site_data_refs:
            prep_tasks[site] = prepare_data_for_site.remote(
                data, site, random_seed=config.random_seed
            )
            if data_openva is not None:
                prep_tasks_openva[site] = prepare_data_for_site.remote(
                    data_openva, site, random_seed=config.random_seed
                )

    # Collect results
    for site, task_ref in prep_tasks.items():
        result = ray.get(task_ref)
        if result is not None:
            X_train, X_test, y_train, y_test = result
            site_data_refs[site] = {
                "train_data": ray.put((X_train, y_train)),
                "test_data": ray.put((X_test, y_test)),
            }
    
    # Collect OpenVA results if available
    for site, task_ref in prep_tasks_openva.items():
        result = ray.get(task_ref)
        if result is not None:
            X_train, X_test, y_train, y_test = result
            site_data_refs_openva[site] = {
                "train_data": ray.put((X_train, y_train)),
                "test_data": ray.put((X_test, y_test)),
            }

    # Update experiment configs with data references
    updated_experiments = []
    for exp in experiments:
        exp_copy = exp.copy()
        
        # Choose appropriate data format based on model
        model_name = exp.get("model_name", "")
        if model_name == "insilico" and site_data_refs_openva:
            data_refs = site_data_refs_openva
        else:
            data_refs = site_data_refs

        if exp["experiment_type"] == "in_domain":
            site = exp["site"]
            if site in data_refs:
                exp_copy["train_data"] = data_refs[site]["train_data"]
                exp_copy["test_data"] = data_refs[site]["test_data"]
                updated_experiments.append(exp_copy)

        elif exp["experiment_type"] == "out_domain":
            train_site = exp["train_site"]
            test_site = exp["test_site"]
            if train_site in data_refs and test_site in data_refs:
                exp_copy["train_data"] = data_refs[train_site]["train_data"]
                exp_copy["test_data"] = data_refs[test_site]["test_data"]
                updated_experiments.append(exp_copy)

        elif exp["experiment_type"] == "training_size":
            site = exp["site"]
            if site in data_refs:
                # For training size experiments, subsample the training data
                exp_copy["train_data"] = data_refs[site]["train_data"]
                exp_copy["test_data"] = data_refs[site]["test_data"]
                exp_copy["training_fraction"] = exp["training_size"]
                updated_experiments.append(exp_copy)

    logger.info(
        f"Prepared data for {len(updated_experiments)}/{len(experiments)} experiments"
    )
    return updated_experiments


def batch_experiments(experiments: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Batch experiments for controlled parallel execution.

    Args:
        experiments: List of experiments
        batch_size: Size of each batch

    Returns:
        List of experiment batches
    """
    batches = []
    for i in range(0, len(experiments), batch_size):
        batches.append(experiments[i : i + batch_size])
    return batches


@flow(
    name="VA34 Comparison Experiment",
    task_runner=ConcurrentTaskRunner(max_workers=10),
    persist_result=True,
)
async def va34_comparison_flow(
    config: ExperimentConfig, parallel_config: ParallelConfig
) -> pd.DataFrame:
    """Main Prefect flow for VA34 comparison experiments.

    Args:
        config: Experiment configuration
        parallel_config: Parallel execution configuration

    Returns:
        DataFrame with all experiment results
    """
    start_time = time.time()
    logger.info("Starting VA34 comparison flow")

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(**parallel_config.to_ray_init_kwargs())
        logger.info(
            f"Ray initialized with dashboard at "
            f"http://localhost:{parallel_config.ray_dashboard_port}"
        )

    # Load and prepare data
    import sys
    from pathlib import Path as PathLib
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(Path(config.output_dir) / "checkpoints")
    )
    
    # Add va-data to path if it exists
    va_data_path = PathLib(__file__).parent.parent.parent / "va-data"
    if va_data_path.exists() and str(va_data_path) not in sys.path:
        sys.path.insert(0, str(va_data_path))
    
    from baseline.data.data_loader_preprocessor import VADataProcessor
    from baseline.config.data_config import DataConfig
    
    # Create data config for loading
    data_config = DataConfig(
        data_path=config.data_path,
        output_dir=config.output_dir,
        openva_encoding=False,  # Load numeric first for ML models
        stratify_by_site=False,
        label_column="va34"
    )
    
    processor = VADataProcessor(data_config)
    data = processor.load_and_process()
    
    # Also load OpenVA encoded data for InSilico
    data_config_openva = DataConfig(
        data_path=config.data_path,
        output_dir=config.output_dir,
        openva_encoding=True,  # OpenVA format for InSilico
        stratify_by_site=False,
        label_column="va34"
    )
    processor_openva = VADataProcessor(data_config_openva)
    data_openva = processor_openva.load_and_process()

    # Filter to specified sites
    if config.sites:
        data = data[data["site"].isin(config.sites)]
        data_openva = data_openva[data_openva["site"].isin(config.sites)]

    # Handle va34 column - ensure all models use "cause" as the label column
    # This standardization is critical for consistent model comparison
    if "va34" in data.columns and "cause" not in data.columns:
        data["cause"] = data["va34"].astype(str)
    if "va34" in data_openva.columns and "cause" not in data_openva.columns:
        data_openva["cause"] = data_openva["va34"].astype(str)

    logger.info(f"Loaded ML data with shape: {data.shape}")
    logger.info(f"Loaded OpenVA data with shape: {data_openva.shape}")

    # Put both datasets in Ray object store
    data_ref = ray.put(data)
    data_openva_ref = ray.put(data_openva)
    
    # Generate experiment configurations
    experiments = generate_experiment_configs(config, data_ref, checkpoint_manager)

    # Check for existing checkpoint
    checkpoint = checkpoint_manager.load_checkpoint(config.model_dump())
    existing_results = []
    if checkpoint:
        experiments = checkpoint_manager.filter_completed_experiments(
            experiments, checkpoint
        )
        existing_results = checkpoint_manager.load_partial_results(checkpoint)
        start_time -= checkpoint.elapsed_seconds  # Adjust for previous run time

    # Prepare data for experiments
    experiments = prepare_experiment_data(experiments, data, config, data_openva_ref)

    # Initialize progress tracking
    progress_actor = ProgressReporter.remote(len(experiments))
    progress_tracker = RayProgressTracker(
        total_experiments=len(experiments) + len(existing_results),
        description="VA34 Experiments",
        progress_actor=progress_actor,
    )

    # Add existing results to progress
    for result in existing_results:
        progress_tracker.update(result)

    performance_monitor = PerformanceMonitor()

    # Run experiments in batches
    all_results = existing_results.copy()
    experiment_batches = batch_experiments(experiments, parallel_config.batch_size)

    for batch_idx, batch in enumerate(experiment_batches):
        logger.info(f"Processing batch {batch_idx + 1}/{len(experiment_batches)}")

        # Submit batch to Ray
        result_refs = []
        for exp in batch:
            # Extract required fields for the remote function
            model_name = exp["model_name"]
            train_data = exp["train_data"]
            test_data = exp["test_data"]
            experiment_metadata = exp["experiment_metadata"]
            
            result_ref = train_and_evaluate_model.remote(
                model_name=model_name,
                train_data=train_data,
                test_data=test_data,
                experiment_metadata=experiment_metadata,
                n_bootstrap=config.n_bootstrap
            )
            result_refs.append(result_ref)

        # Wait for results with progress updates
        while result_refs:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1, timeout=1.0)

            if ready_refs:
                results = ray.get(ready_refs)
                for result in results:
                    all_results.append(result)
                    progress_tracker.update(result)
                    performance_monitor.record_experiment(result)

        # Save checkpoint after each batch
        if (batch_idx + 1) % parallel_config.checkpoint_interval == 0:
            elapsed = time.time() - start_time
            checkpoint_manager.save_checkpoint(
                all_results, config.model_dump(), len(experiments) + len(existing_results), elapsed
            )

    # Close progress tracker
    progress_tracker.close()

    # Save final results
    final_results = pd.DataFrame([r.to_dict() for r in all_results])
    output_path = Path(config.output_dir) / "va34_comparison_results.csv"
    final_results.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Log performance summary
    perf_summary = performance_monitor.get_performance_summary()
    logger.info(f"Performance summary: {perf_summary}")

    # Generate visualizations if requested
    if config.generate_plots:
        logger.info("Generating visualizations...")
        from model_comparison.visualization.comparison_plots import (
            plot_model_comparison,
            plot_model_performance,
            plot_generalization_gap,
        )

        plot_dir = Path(config.output_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Generate different plot types
        plot_model_comparison(final_results, str(plot_dir / "model_comparison.png"))
        plot_model_performance(final_results, str(plot_dir / "model_performance.png"))
        plot_generalization_gap(final_results, str(plot_dir / "generalization_gap.png"))

    total_time = time.time() - start_time
    logger.info(
        f"Experiment completed in {total_time:.1f}s "
        f"({len(all_results)} experiments, "
        f"{len(all_results)/total_time:.2f} exp/s)"
    )

    return final_results


@flow(name="Cleanup Ray Resources")
def cleanup_ray_resources():
    """Clean up Ray resources after experiment completion."""
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")