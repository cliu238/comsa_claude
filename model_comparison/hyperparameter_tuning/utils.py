"""Utility functions for hyperparameter tuning."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from baseline.utils import get_logger

logger = get_logger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory for tuned parameters.
    
    Returns:
        Path to cache directory
    """
    cache_dir = Path("cache/tuned_params")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(cache_key: str) -> Path:
    """Get the cache file path for a specific key.
    
    Args:
        cache_key: Unique identifier for the cached parameters
        
    Returns:
        Path to cache file
    """
    cache_dir = get_cache_dir()
    # Sanitize cache key for filesystem
    safe_key = cache_key.replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe_key}.json"


def save_cached_params(
    cache_key: str,
    params: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save tuned parameters to cache.
    
    Args:
        cache_key: Unique identifier for the cached parameters
        params: Parameters to cache
        metadata: Optional metadata (e.g., score, timestamp)
    """
    cache_path = get_cache_path(cache_key)
    
    cache_data = {
        "params": params,
        "timestamp": datetime.now().isoformat(),
        "cache_key": cache_key,
    }
    
    if metadata:
        cache_data["metadata"] = metadata
    
    try:
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Cached parameters saved to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save cached parameters: {e}")


def load_cached_params(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached parameters if they exist.
    
    Args:
        cache_key: Unique identifier for the cached parameters
        
    Returns:
        Cached parameters or None if not found
    """
    cache_path = get_cache_path(cache_key)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        
        logger.info(
            f"Loaded cached parameters from {cache_path} "
            f"(cached at {cache_data.get('timestamp', 'unknown')})"
        )
        return cache_data["params"]
    except Exception as e:
        logger.error(f"Failed to load cached parameters: {e}")
        return None


def clear_cache(pattern: Optional[str] = None) -> int:
    """Clear cached parameters.
    
    Args:
        pattern: Optional pattern to match cache keys (None clears all)
        
    Returns:
        Number of files cleared
    """
    cache_dir = get_cache_dir()
    cleared = 0
    
    for cache_file in cache_dir.glob("*.json"):
        if pattern is None or pattern in cache_file.stem:
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.error(f"Failed to remove {cache_file}: {e}")
    
    logger.info(f"Cleared {cleared} cached parameter files")
    return cleared


def create_tuning_report(
    model_name: str,
    tuning_result: Any,  # TuningResult from tuner.py
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Create a tuning report with best parameters and performance.
    
    Args:
        model_name: Name of the model
        tuning_result: TuningResult object
        output_path: Optional path to save the report
        
    Returns:
        DataFrame with tuning results
    """
    report_data = {
        "model": model_name,
        "best_score": tuning_result.best_score,
        "n_trials": tuning_result.n_trials_completed,
        "duration_seconds": tuning_result.duration_seconds,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add best parameters as separate columns
    for param, value in tuning_result.best_params.items():
        report_data[f"param_{param}"] = value
    
    # Create DataFrame
    report_df = pd.DataFrame([report_data])
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)
        logger.info(f"Tuning report saved to {output_path}")
    
    return report_df


def merge_tuning_reports(report_paths: list) -> pd.DataFrame:
    """Merge multiple tuning reports into a single DataFrame.
    
    Args:
        report_paths: List of paths to tuning report CSV files
        
    Returns:
        Combined DataFrame with all tuning results
    """
    reports = []
    
    for path in report_paths:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                reports.append(df)
            except Exception as e:
                logger.error(f"Failed to read report {path}: {e}")
    
    if reports:
        combined = pd.concat(reports, ignore_index=True)
        combined = combined.sort_values("best_score", ascending=False)
        return combined
    else:
        return pd.DataFrame()


class TuningProgressTracker:
    """Track and report progress during hyperparameter tuning."""
    
    def __init__(self, total_experiments: int):
        """Initialize progress tracker.
        
        Args:
            total_experiments: Total number of experiments to track
        """
        self.total_experiments = total_experiments
        self.completed_experiments = 0
        self.start_time = datetime.now()
        self.experiment_times = []
    
    def update(self, experiment_id: str, duration: float) -> None:
        """Update progress for a completed experiment.
        
        Args:
            experiment_id: ID of the completed experiment
            duration: Time taken for the experiment
        """
        self.completed_experiments += 1
        self.experiment_times.append(duration)
        
        # Calculate statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_time = sum(self.experiment_times) / len(self.experiment_times)
        remaining = (self.total_experiments - self.completed_experiments) * avg_time
        
        logger.info(
            f"Progress: {self.completed_experiments}/{self.total_experiments} "
            f"({self.completed_experiments/self.total_experiments*100:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s | Est. remaining: {remaining:.1f}s | "
            f"Experiment: {experiment_id}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dictionary with progress statistics
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_experiments": self.total_experiments,
            "completed_experiments": self.completed_experiments,
            "completion_rate": self.completed_experiments / self.total_experiments,
            "elapsed_seconds": elapsed,
            "average_experiment_seconds": (
                sum(self.experiment_times) / len(self.experiment_times)
                if self.experiment_times else 0
            ),
        }