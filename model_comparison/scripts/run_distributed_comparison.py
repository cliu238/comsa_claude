#!/usr/bin/env python
"""Distributed VA34 model comparison using Prefect and Ray.

This script provides a CLI interface for running VA model comparison experiments
in a distributed manner using Ray for parallel computation and Prefect for
workflow orchestration.

Usage:
    python run_distributed_comparison.py \
        --data-path data/va34_data.csv \
        --sites site_1 site_2 site_3 \
        --models xgboost insilico \
        --n-workers 4 \
        --output-dir results/distributed

Expected runtime: Depends on data size and number of workers
Progress will be displayed in real-time with checkpoint saves
"""

import argparse
import asyncio
import sys
from pathlib import Path

import ray
from prefect import serve

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from baseline.utils import get_logger
from model_comparison.experiments.experiment_config import ExperimentConfig, TuningConfig
from model_comparison.orchestration.config import ParallelConfig
from model_comparison.orchestration.prefect_flows import (
    cleanup_ray_resources,
    va34_comparison_flow,
)

logger = get_logger(__name__, component="orchestration")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run distributed VA34 model comparison experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-path", required=True, help="Path to VA data CSV file"
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        required=True,
        help="List of sites to include in comparison",
    )

    # Model arguments
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "insilico"],
        choices=["xgboost", "insilico", "random_forest", "logistic_regression", "categorical_nb"],
        help="Models to compare",
    )

    # Training configuration
    parser.add_argument(
        "--training-sizes",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Training data fractions for size experiments",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations for metrics",
    )

    # Parallel execution arguments
    parser.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of Ray workers (-1 for auto)",
    )
    parser.add_argument(
        "--memory-per-worker",
        default="4GB",
        help="Memory allocation per worker (e.g., 4GB, 512MB)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of experiments to run in parallel",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N batches",
    )

    # Dashboard options
    parser.add_argument(
        "--no-ray-dashboard",
        action="store_true",
        help="Disable Ray dashboard",
    )
    parser.add_argument(
        "--ray-dashboard-port",
        type=int,
        default=8265,
        help="Port for Ray dashboard",
    )
    parser.add_argument(
        "--no-prefect-dashboard",
        action="store_true",
        help="Disable Prefect dashboard",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="results/distributed",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )

    # Hyperparameter tuning arguments
    parser.add_argument(
        "--enable-tuning",
        action="store_true",
        help="Enable hyperparameter tuning for ML models",
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=100,
        help="Number of tuning trials per model",
    )
    parser.add_argument(
        "--tuning-algorithm",
        choices=["grid", "random", "bayesian"],
        default="bayesian",
        help="Hyperparameter search algorithm",
    )
    parser.add_argument(
        "--tuning-metric",
        choices=["csmf_accuracy", "cod_accuracy"],
        default="csmf_accuracy",
        help="Metric to optimize during tuning",
    )
    parser.add_argument(
        "--tuning-cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for tuning",
    )
    parser.add_argument(
        "--tuning-cpus-per-trial",
        type=float,
        default=1.0,
        help="CPUs allocated per tuning trial",
    )
    
    # Other arguments
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear existing checkpoints before starting",
    )

    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_arguments()

    # Create tuning configuration
    tuning_config = TuningConfig(
        enabled=args.enable_tuning,
        n_trials=args.tuning_trials,
        search_algorithm=args.tuning_algorithm,
        tuning_metric=args.tuning_metric,
        cv_folds=args.tuning_cv_folds,
        n_cpus_per_trial=args.tuning_cpus_per_trial,
    )
    
    # Create configurations
    experiment_config = ExperimentConfig(
        data_path=args.data_path,
        sites=args.sites,
        models=args.models,
        training_sizes=args.training_sizes,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        tuning=tuning_config,
    )

    parallel_config = ParallelConfig(
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        ray_dashboard=not args.no_ray_dashboard,
        ray_dashboard_port=args.ray_dashboard_port,
        prefect_dashboard=not args.no_prefect_dashboard,
    )

    # Log configuration
    logger.info("Starting distributed VA34 comparison")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Sites: {args.sites}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Workers: {args.n_workers}")
    logger.info(f"Output directory: {args.output_dir}")

    # Clear checkpoints if requested
    if args.clear_checkpoints:
        from model_comparison.orchestration.checkpoint_manager import CheckpointManager

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(Path(args.output_dir) / "checkpoints")
        )
        checkpoint_manager.clear_checkpoints()
        logger.info("Cleared existing checkpoints")

    # Display dashboard URLs
    if not args.no_ray_dashboard:
        logger.info(f"Ray dashboard will be available at: http://localhost:{args.ray_dashboard_port}")
    if not args.no_prefect_dashboard:
        logger.info("Prefect dashboard available at: http://localhost:4200")
        logger.info("Start Prefect server with: prefect server start")

    try:
        # Run the flow
        logger.info("Starting Prefect flow execution")
        results = await va34_comparison_flow(experiment_config, parallel_config)

        # Summary statistics
        logger.info("\n" + "=" * 50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful experiments: {len(results[results['error'].isna()])}")
        logger.info(f"Failed experiments: {len(results[results['error'].notna()])}")

        # Model performance summary
        for model in args.models:
            model_results = results[results["model_name"] == model]
            if len(model_results) > 0:
                avg_csmf = model_results["csmf_accuracy"].mean()
                avg_cod = model_results["cod_accuracy"].mean()
                logger.info(
                    f"\n{model.upper()} Performance:"
                    f"\n  Avg CSMF Accuracy: {avg_csmf:.3f}"
                    f"\n  Avg COD Accuracy: {avg_cod:.3f}"
                )

        logger.info(f"\nResults saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        # Clean up Ray resources
        cleanup_ray_resources()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())