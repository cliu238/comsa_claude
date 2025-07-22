"""Checkpoint management for resumable experiment execution."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from baseline.utils import get_logger
from model_comparison.orchestration.config import CheckpointState, ExperimentResult

logger = get_logger(__name__, component="orchestration")


class CheckpointManager:
    """Manages checkpoints for experiment state persistence and recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint_file = self.checkpoint_dir / "current_checkpoint.json"
        self.results_dir = self.checkpoint_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        results: List[ExperimentResult],
        config: Dict,
        total_experiments: int,
        elapsed_seconds: float,
    ) -> Path:
        """Save checkpoint with atomic write to prevent corruption.

        Args:
            results: List of completed experiment results
            config: Experiment configuration
            total_experiments: Total number of experiments
            elapsed_seconds: Elapsed time in seconds

        Returns:
            Path to saved checkpoint file
        """
        # Create checkpoint state
        checkpoint = CheckpointState.from_config(
            config=config,
            completed_experiments=[r.experiment_id for r in results],
            total_experiments=total_experiments,
            elapsed_seconds=elapsed_seconds,
        )

        # Save results DataFrame
        if results:
            results_df = pd.DataFrame([r.to_dict() for r in results])
            results_file = self.results_dir / f"results_{checkpoint.timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            checkpoint.partial_results = {"file": str(results_file)}

        # Atomic write: write to temp file first, then move
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.checkpoint_dir, delete=False
        ) as tmp_file:
            json.dump(checkpoint.model_dump(), tmp_file, indent=2)
            tmp_path = tmp_file.name

        # Move temp file to final location (atomic on POSIX systems)
        shutil.move(tmp_path, self.current_checkpoint_file)

        logger.info(
            f"Saved checkpoint: {len(results)} experiments completed "
            f"({checkpoint.get_completion_percentage():.1f}%)"
        )

        # Clean up old checkpoints (keep last 5)
        self._cleanup_old_checkpoints()

        return self.current_checkpoint_file

    def load_checkpoint(self, config: Dict) -> Optional[CheckpointState]:
        """Load checkpoint if it exists and is compatible with config.

        Args:
            config: Current experiment configuration

        Returns:
            CheckpointState if valid checkpoint exists, None otherwise
        """
        if not self.current_checkpoint_file.exists():
            logger.info("No checkpoint found")
            return None

        try:
            with open(self.current_checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            checkpoint = CheckpointState(**checkpoint_data)

            # Verify compatibility
            if not checkpoint.is_compatible(config):
                logger.warning(
                    "Checkpoint config mismatch - starting fresh. "
                    "Delete checkpoint directory to force resume with different config."
                )
                return None

            logger.info(
                f"Loaded checkpoint: {len(checkpoint.completed_experiments)} "
                f"experiments completed ({checkpoint.get_completion_percentage():.1f}%)"
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def load_partial_results(self, checkpoint: CheckpointState) -> List[ExperimentResult]:
        """Load partial results from checkpoint.

        Args:
            checkpoint: Checkpoint state with results file info

        Returns:
            List of experiment results
        """
        if not checkpoint.partial_results or "file" not in checkpoint.partial_results:
            return []

        results_file = Path(checkpoint.partial_results["file"])
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return []

        try:
            results_df = pd.read_csv(results_file)
            results = []

            for _, row in results_df.iterrows():
                # Convert row to dict and create ExperimentResult
                result_dict = row.to_dict()

                # Handle potential None values for lists
                if pd.isna(result_dict.get("csmf_accuracy_ci")):
                    result_dict["csmf_accuracy_ci"] = None
                if pd.isna(result_dict.get("cod_accuracy_ci")):
                    result_dict["cod_accuracy_ci"] = None
                    
                # Handle NaN values for optional string fields
                if pd.isna(result_dict.get("worker_id")):
                    result_dict["worker_id"] = None
                if pd.isna(result_dict.get("error")):
                    result_dict["error"] = None

                results.append(ExperimentResult(**result_dict))

            logger.info(f"Loaded {len(results)} results from checkpoint")
            return results

        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return []

    def create_experiment_id(
        self,
        model_name: str,
        experiment_type: str,
        train_site: str,
        test_site: str,
        training_size: float = 1.0,
    ) -> str:
        """Create unique experiment ID.

        Args:
            model_name: Model name
            experiment_type: Type of experiment
            train_site: Training site
            test_site: Test site
            training_size: Training data fraction

        Returns:
            Unique experiment identifier
        """
        components = [
            model_name,
            experiment_type,
            train_site,
            test_site,
            f"size{training_size}",
        ]
        return "_".join(components)

    def filter_completed_experiments(
        self, experiments: List[Dict], checkpoint: CheckpointState
    ) -> List[Dict]:
        """Filter out already completed experiments.

        Args:
            experiments: List of experiment configurations
            checkpoint: Checkpoint with completed experiment IDs

        Returns:
            List of experiments that haven't been completed
        """
        completed_ids = set(checkpoint.completed_experiments)
        remaining = []

        for exp in experiments:
            # Get experiment ID from metadata if available, otherwise create it
            if "experiment_id" in exp.get("experiment_metadata", {}):
                exp_id = exp["experiment_metadata"]["experiment_id"]
            else:
                exp_id = self.create_experiment_id(
                    model_name=exp["model_name"],
                    experiment_type=exp["experiment_metadata"]["experiment_type"],
                    train_site=exp["experiment_metadata"]["train_site"],
                    test_site=exp["experiment_metadata"]["test_site"],
                    training_size=exp["experiment_metadata"].get("training_size", 1.0),
                )

            if exp_id not in completed_ids:
                remaining.append(exp)

        logger.info(
            f"Filtered experiments: {len(experiments)} total, "
            f"{len(completed_ids)} completed, {len(remaining)} remaining"
        )

        return remaining

    def _cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Clean up old checkpoint files, keeping the most recent ones.

        Args:
            keep_last: Number of recent checkpoints to keep
        """
        # Find all result files
        result_files = sorted(
            self.results_dir.glob("results_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove old files
        for file in result_files[keep_last:]:
            try:
                file.unlink()
                logger.debug(f"Removed old checkpoint file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints (useful for starting fresh)."""
        try:
            shutil.rmtree(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(exist_ok=True)
            logger.info("Cleared all checkpoints")
        except Exception as e:
            logger.error(f"Failed to clear checkpoints: {e}")