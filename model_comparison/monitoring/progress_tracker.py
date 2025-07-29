"""Progress tracking with tqdm integration for distributed experiments."""

import time
from datetime import timedelta
from typing import Dict, List, Optional

import ray
from tqdm import tqdm

from baseline.utils import get_logger
from model_comparison.orchestration.config import ExperimentResult

logger = get_logger(__name__, component="orchestration")


class ProgressTracker:
    """Tracks progress of distributed experiments with tqdm integration."""

    def __init__(
        self,
        total_experiments: int,
        description: str = "Experiments",
        show_progress_bar: bool = True,
    ):
        """Initialize progress tracker.

        Args:
            total_experiments: Total number of experiments to track
            description: Description for progress bar
            show_progress_bar: Whether to show tqdm progress bar
        """
        self.total_experiments = total_experiments
        self.completed_experiments = 0
        self.failed_experiments = 0
        self.start_time = time.time()
        self.results: List[ExperimentResult] = []
        self.show_progress_bar = show_progress_bar

        # Initialize tqdm progress bar
        if self.show_progress_bar:
            self.pbar = tqdm(
                total=total_experiments,
                desc=description,
                unit="exp",
                unit_scale=True,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            self._update_postfix()

    def update(self, result: ExperimentResult) -> None:
        """Update progress with a completed experiment result.

        Args:
            result: Completed experiment result
        """
        self.completed_experiments += 1
        self.results.append(result)

        if result.error:
            self.failed_experiments += 1

        if self.show_progress_bar:
            self.pbar.update(1)
            self._update_postfix()

        # Log progress periodically
        if self.completed_experiments % 10 == 0:
            self._log_progress()
        
        # Log errors immediately
        if result.error:
            logger.warning(
                f"Experiment failed: {result.experiment_id} - "
                f"Model: {result.model_name} - Error: {result.error}"
            )

    def _update_postfix(self) -> None:
        """Update tqdm postfix with current statistics."""
        if not self.show_progress_bar:
            return

        success_rate = (
            (self.completed_experiments - self.failed_experiments)
            / max(self.completed_experiments, 1)
            * 100
        )

        postfix = {
            "Success": f"{success_rate:.1f}%",
            "Failed": self.failed_experiments,
        }

        # Add average metrics if available
        if self.results:
            recent_results = self.results[-20:]  # Last 20 results
            avg_csmf = sum(r.csmf_accuracy for r in recent_results) / len(
                recent_results
            )
            avg_cod = sum(r.cod_accuracy for r in recent_results) / len(recent_results)
            postfix["CSMF"] = f"{avg_csmf:.3f}"
            postfix["COD"] = f"{avg_cod:.3f}"

        self.pbar.set_postfix(postfix)

    def _log_progress(self) -> None:
        """Log detailed progress information."""
        elapsed = time.time() - self.start_time
        rate = self.completed_experiments / elapsed if elapsed > 0 else 0

        remaining_experiments = self.total_experiments - self.completed_experiments
        eta_seconds = remaining_experiments / rate if rate > 0 else 0
        eta = timedelta(seconds=int(eta_seconds))

        # Get current best model performance
        best_info = ""
        if self.results:
            successful_results = [r for r in self.results if not r.error]
            if successful_results:
                best_result = max(successful_results, key=lambda r: r.csmf_accuracy)
                best_info = f" - Best: {best_result.model_name} (CSMF: {best_result.csmf_accuracy:.3f})"

        logger.info(
            f"\n{'='*60}\n"
            f"PROGRESS UPDATE: {self.completed_experiments}/{self.total_experiments} experiments "
            f"({self.get_completion_percentage():.1f}%)\n"
            f"  • Success Rate: {((self.completed_experiments - self.failed_experiments) / max(self.completed_experiments, 1) * 100):.1f}%\n"
            f"  • Failed: {self.failed_experiments}\n"
            f"  • Runtime: {timedelta(seconds=int(elapsed))}\n"
            f"  • Speed: {rate:.2f} exp/s\n"
            f"  • ETA: {eta}{best_info}\n"
            f"{'='*60}"
        )

    def get_completion_percentage(self) -> float:
        """Get percentage of experiments completed."""
        if self.total_experiments == 0:
            return 0.0
        return (self.completed_experiments / self.total_experiments) * 100

    def get_statistics(self) -> Dict:
        """Get detailed progress statistics."""
        elapsed = time.time() - self.start_time
        rate = self.completed_experiments / elapsed if elapsed > 0 else 0

        stats = {
            "total_experiments": self.total_experiments,
            "completed_experiments": self.completed_experiments,
            "failed_experiments": self.failed_experiments,
            "success_rate": (
                (self.completed_experiments - self.failed_experiments)
                / max(self.completed_experiments, 1)
            ),
            "completion_percentage": self.get_completion_percentage(),
            "elapsed_seconds": elapsed,
            "experiments_per_second": rate,
            "estimated_remaining_seconds": (
                (self.total_experiments - self.completed_experiments) / rate
                if rate > 0
                else 0
            ),
        }

        # Add metric statistics if available
        if self.results:
            csmf_scores = [r.csmf_accuracy for r in self.results if not r.error]
            cod_scores = [r.cod_accuracy for r in self.results if not r.error]

            if csmf_scores:
                stats["avg_csmf_accuracy"] = sum(csmf_scores) / len(csmf_scores)
                stats["min_csmf_accuracy"] = min(csmf_scores)
                stats["max_csmf_accuracy"] = max(csmf_scores)

            if cod_scores:
                stats["avg_cod_accuracy"] = sum(cod_scores) / len(cod_scores)
                stats["min_cod_accuracy"] = min(cod_scores)
                stats["max_cod_accuracy"] = max(cod_scores)

        return stats

    def close(self) -> None:
        """Close the progress bar."""
        if self.show_progress_bar and hasattr(self, "pbar"):
            self.pbar.close()

        # Log final statistics
        stats = self.get_statistics()
        logger.info(
            f"Completed {self.completed_experiments}/{self.total_experiments} experiments "
            f"in {timedelta(seconds=int(stats['elapsed_seconds']))} "
            f"(Success rate: {stats['success_rate']*100:.1f}%)"
        )


class RayProgressTracker(ProgressTracker):
    """Progress tracker that integrates with Ray's distributed execution."""

    def __init__(
        self,
        total_experiments: int,
        progress_actor: Optional["ray.actor.ActorHandle"] = None,
        **kwargs,
    ):
        """Initialize Ray-aware progress tracker.

        Args:
            total_experiments: Total number of experiments
            progress_actor: Optional Ray actor for centralized progress
            **kwargs: Additional arguments for ProgressTracker
        """
        super().__init__(total_experiments, **kwargs)
        self.progress_actor = progress_actor

    def update_from_ray_results(
        self, ready_refs: List[ray.ObjectRef], results: List[ExperimentResult]
    ) -> None:
        """Update progress from Ray results.

        Args:
            ready_refs: Ray object references that completed
            results: Corresponding experiment results
        """
        for result in results:
            self.update(result)

            # Report to Ray actor if available
            if self.progress_actor:
                ray.get(self.progress_actor.report_completion.remote(result))

    def get_ray_progress(self) -> Optional[Dict]:
        """Get progress from Ray actor if available."""
        if self.progress_actor:
            return ray.get(self.progress_actor.get_progress.remote())
        return None


class PerformanceMonitor:
    """Monitors performance metrics during distributed execution."""

    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = time.time()
        self.experiment_times: List[float] = []
        self.worker_stats: Dict[str, Dict] = {}

    def record_experiment(self, result: ExperimentResult) -> None:
        """Record experiment execution metrics.

        Args:
            result: Completed experiment result
        """
        self.experiment_times.append(result.execution_time_seconds)

        # Track per-worker statistics
        if result.worker_id:
            if result.worker_id not in self.worker_stats:
                self.worker_stats[result.worker_id] = {
                    "experiments": 0,
                    "total_time": 0.0,
                    "failures": 0,
                }

            stats = self.worker_stats[result.worker_id]
            stats["experiments"] += 1
            stats["total_time"] += result.execution_time_seconds
            if result.error:
                stats["failures"] += 1

    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.experiment_times:
            return {}

        total_time = time.time() - self.start_time
        avg_time = sum(self.experiment_times) / len(self.experiment_times)

        summary = {
            "total_elapsed_seconds": total_time,
            "total_experiments": len(self.experiment_times),
            "avg_experiment_seconds": avg_time,
            "min_experiment_seconds": min(self.experiment_times),
            "max_experiment_seconds": max(self.experiment_times),
            "throughput_per_second": len(self.experiment_times) / total_time,
            "worker_count": len(self.worker_stats),
        }

        # Add worker load balancing metrics
        if self.worker_stats:
            worker_loads = [
                stats["experiments"] for stats in self.worker_stats.values()
            ]
            avg_experiments = sum(worker_loads) / len(worker_loads)
            summary["worker_load_balance"] = {
                "min_experiments": min(worker_loads),
                "max_experiments": max(worker_loads),
                "avg_experiments": avg_experiments,
                "std_experiments": (
                    sum((x - avg_experiments) ** 2 for x in worker_loads)
                    / len(worker_loads)
                )
                ** 0.5,
            }

        return summary