#!/usr/bin/env python
"""Run VA34 site-based model comparison experiment."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.utils import setup_logging
from model_comparison.experiments.experiment_config import ExperimentConfig
from model_comparison.experiments.site_comparison import SiteComparisonExperiment
from model_comparison.experiments.parallel_experiment import ParallelSiteComparisonExperiment
from model_comparison.orchestration.config import ParallelConfig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run VA34 site-based model comparison experiment"
    )
    
    # Required arguments
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to VA data CSV file",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        default="model_comparison/results/va34_comparison",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Specific sites to include (default: all sites in data)",
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["insilico", "xgboost"],
        default=["insilico", "xgboost"],
        help="Models to compare",
    )
    
    parser.add_argument(
        "--training-sizes",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Training data fractions to test",
    )
    
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations for confidence intervals",
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores)",
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    # Parallel execution arguments
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution using Ray",
    )
    
    parser.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of Ray workers for parallel execution (-1 for auto)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for parallel execution",
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints (defaults to output_dir/checkpoints)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    
    args = parser.parse_args()
    
    # Setup centralized logging
    logger = setup_logging(
        __name__,
        level="DEBUG" if args.debug else "INFO",
        component="va34_comparison"
    )
    
    try:
        # Validate data path
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Create configuration
        config = ExperimentConfig(
            data_path=str(data_path),
            sites=args.sites if args.sites else ["site_1", "site_2", "site_3", "site_4"],
            models=args.models,
            training_sizes=args.training_sizes,
            n_bootstrap=args.n_bootstrap,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            generate_plots=not args.no_plots,
        )
        
        logger.info("Starting VA34 site comparison experiment")
        logger.info(f"Configuration: {config.model_dump()}")
        logger.info(f"Parallel execution: {args.parallel}")
        
        # Run experiment
        if args.parallel:
            # Create parallel configuration
            parallel_config = ParallelConfig(
                n_workers=args.n_workers,
                batch_size=args.batch_size,
                checkpoint_interval=10,
                ray_dashboard=True,
                prefect_dashboard=False,  # Don't use Prefect in CLI mode
            )
            
            logger.info("Using parallel execution with Ray")
            logger.info(f"Workers: {args.n_workers} (-1 for auto)")
            logger.info(f"Batch size: {args.batch_size}")
            
            # Use parallel experiment runner
            experiment = ParallelSiteComparisonExperiment(config, parallel_config)
            
            if not args.resume and args.checkpoint_dir:
                # Clear checkpoints if not resuming
                experiment.checkpoint_manager.clear_checkpoints()
                
            results = experiment.run_experiment()
            
            # Clean up Ray
            experiment.cleanup()
        else:
            # Use sequential experiment runner
            experiment = SiteComparisonExperiment(config)
            results = experiment.run_experiment()
        
        # Print summary
        print("\n" + "=" * 60)
        print("VA34 SITE COMPARISON EXPERIMENT SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal experiments run: {len(results)}")
        print(f"Output directory: {config.output_dir}")
        
        # Model performance summary
        print("\n--- Average CSMF Accuracy by Model ---")
        model_summary = results.groupby("model")["csmf_accuracy"].agg(["mean", "std"])
        for model, row in model_summary.iterrows():
            print(f"{model}: {row['mean']:.3f} (±{row['std']:.3f})")
        
        # Generalization gap
        print("\n--- Generalization Gap (In-domain - Out-domain) ---")
        in_domain = results[results["experiment_type"] == "in_domain"]
        out_domain = results[results["experiment_type"] == "out_domain"]
        
        if not in_domain.empty and not out_domain.empty:
            for model in config.models:
                in_perf = in_domain[in_domain["model"] == model]["csmf_accuracy"].mean()
                out_perf = out_domain[out_domain["model"] == model]["csmf_accuracy"].mean()
                gap = in_perf - out_perf
                print(f"{model}: {gap:.3f} ({in_perf:.3f} → {out_perf:.3f})")
        
        # Training size impact
        print("\n--- Training Size Impact ---")
        size_results = results[results["experiment_type"] == "training_size"]
        if not size_results.empty:
            for model in config.models:
                model_size = size_results[size_results["model"] == model]
                if not model_size.empty:
                    print(f"\n{model}:")
                    for _, row in model_size.iterrows():
                        print(
                            f"  {row['training_size']:.0%}: "
                            f"CSMF={row['csmf_accuracy']:.3f}"
                        )
        
        print("\n" + "=" * 60)
        print("Experiment completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()