"""Configuration for parallel execution with Prefect and Ray."""

import hashlib
import json
import platform
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ParallelConfig(BaseModel):
    """Configuration for parallel execution."""

    # Ray configuration
    n_workers: int = Field(
        default=-1, description="Number of Ray workers (-1 for auto)"
    )
    memory_per_worker: Optional[str] = Field(
        default="4GB", description="Memory per worker"
    )
    object_store_memory: Optional[str] = Field(
        default="8GB", description="Ray object store memory"
    )

    # Execution configuration
    checkpoint_interval: int = Field(
        default=10, description="Checkpoint every N experiments"
    )
    batch_size: int = Field(
        default=50, description="Number of experiments to run in parallel"
    )
    retry_attempts: int = Field(default=3, description="Retry failed tasks N times")
    timeout_seconds: int = Field(
        default=300, description="Timeout per experiment in seconds"
    )

    # Dashboard configuration
    prefect_dashboard: bool = Field(
        default=True, description="Enable Prefect dashboard"
    )
    ray_dashboard: bool = Field(default=True, description="Enable Ray dashboard")
    ray_dashboard_port: int = Field(default=8265, description="Ray dashboard port")

    # Resource limits
    max_concurrent_experiments: int = Field(
        default=100, description="Maximum concurrent experiments"
    )

    @field_validator("memory_per_worker", "object_store_memory")
    def validate_memory_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate memory format (e.g., '4GB', '512MB')."""
        if v is None:
            return v

        # Check format
        if not v.endswith(("GB", "MB", "KB")):
            raise ValueError(f"Memory must end with GB, MB, or KB: {v}")

        # Extract numeric part
        try:
            numeric_part = v[:-2]
            float(numeric_part)
        except ValueError:
            raise ValueError(f"Invalid memory format: {v}")

        return v

    def to_ray_init_kwargs(self) -> Dict:
        """Convert to Ray initialization kwargs."""
        kwargs = {
            "dashboard_host": "0.0.0.0" if self.ray_dashboard else None,
            "dashboard_port": self.ray_dashboard_port if self.ray_dashboard else None,
        }

        if self.n_workers > 0:
            kwargs["num_cpus"] = self.n_workers

        if self.object_store_memory:
            # Convert to bytes for Ray
            memory_bytes = self._memory_to_bytes(self.object_store_memory)
            
            # Check if running on macOS and limit memory to 2GB
            if platform.system() == "Darwin":
                max_macos_memory = self._memory_to_bytes("2GB")
                if memory_bytes > max_macos_memory:
                    memory_bytes = max_macos_memory
            
            kwargs["object_store_memory"] = memory_bytes

        return kwargs

    def _memory_to_bytes(self, memory_str: str) -> int:
        """Convert memory string to bytes."""
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return int(float(memory_str[:-2]) * multiplier)

        raise ValueError(f"Unknown memory unit in: {memory_str}")

    model_config = {"validate_assignment": True}


class CheckpointState(BaseModel):
    """State for checkpoint/resume functionality."""

    completed_experiments: List[str] = Field(
        default_factory=list, description="IDs of completed experiments"
    )
    partial_results: Optional[Dict] = Field(
        default=None, description="Partial results from interrupted run"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Checkpoint timestamp",
    )
    config_hash: str = Field(..., description="Hash to ensure config compatibility")
    total_experiments: int = Field(0, description="Total number of experiments")
    elapsed_seconds: float = Field(0.0, description="Elapsed time in seconds")

    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "CheckpointState":
        """Create checkpoint state from configuration."""
        # Create hash from config for compatibility checking
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        return cls(config_hash=config_hash, **kwargs)

    def is_compatible(self, config: Dict) -> bool:
        """Check if checkpoint is compatible with current config."""
        config_str = json.dumps(config, sort_keys=True)
        current_hash = hashlib.md5(config_str.encode()).hexdigest()
        return current_hash == self.config_hash

    def add_completed_experiment(self, experiment_id: str) -> None:
        """Add experiment to completed list."""
        if experiment_id not in self.completed_experiments:
            self.completed_experiments.append(experiment_id)

    def get_completion_percentage(self) -> float:
        """Get percentage of experiments completed."""
        if self.total_experiments == 0:
            return 0.0
        return (len(self.completed_experiments) / self.total_experiments) * 100

    model_config = {"validate_assignment": True}


class ExperimentResult(BaseModel):
    """Result from a single experiment execution."""

    # Experiment metadata
    experiment_id: str = Field(..., description="Unique experiment identifier")
    model_name: str = Field(..., description="Model name")
    experiment_type: str = Field(..., description="Type of experiment")
    train_site: str = Field(..., description="Training site")
    test_site: str = Field(..., description="Test site")
    training_size: Optional[float] = Field(
        default=1.0, description="Fraction of training data used"
    )

    # Metrics
    csmf_accuracy: float = Field(..., description="CSMF accuracy")
    cod_accuracy: float = Field(..., description="COD accuracy")
    csmf_accuracy_ci: Optional[List[float]] = Field(
        default=None, description="CSMF accuracy confidence interval"
    )
    cod_accuracy_ci: Optional[List[float]] = Field(
        default=None, description="COD accuracy confidence interval"
    )
    
    # Dataset sizes
    n_train: int = Field(default=0, description="Number of training samples")
    n_test: int = Field(default=0, description="Number of test samples"
    )

    # Execution metadata
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    worker_id: Optional[str] = Field(default=None, description="Ray worker ID")
    retry_count: int = Field(default=0, description="Number of retries")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        data = self.model_dump()
        
        # Transform field names to match visualization expectations
        data["model"] = data.pop("model_name")
        data["training_fraction"] = data.pop("training_size")
        
        # Convert CI lists to separate lower/upper fields
        csmf_ci = data.get("csmf_accuracy_ci")
        if csmf_ci and isinstance(csmf_ci, list) and len(csmf_ci) >= 2:
            data["csmf_accuracy_ci_lower"] = csmf_ci[0]
            data["csmf_accuracy_ci_upper"] = csmf_ci[1]
            # Keep the original CI list for debugging
            data["csmf_accuracy_ci"] = csmf_ci
        else:
            # If no CI provided, don't set CI bounds (let visualization handle it)
            data.pop("csmf_accuracy_ci", None)
            data["csmf_accuracy_ci_lower"] = None
            data["csmf_accuracy_ci_upper"] = None
            
        cod_ci = data.get("cod_accuracy_ci")
        if cod_ci and isinstance(cod_ci, list) and len(cod_ci) >= 2:
            data["cod_accuracy_ci_lower"] = cod_ci[0]
            data["cod_accuracy_ci_upper"] = cod_ci[1]
            # Keep the original CI list for debugging
            data["cod_accuracy_ci"] = cod_ci
        else:
            # If no CI provided, don't set CI bounds (let visualization handle it)
            data.pop("cod_accuracy_ci", None)
            data["cod_accuracy_ci_lower"] = None
            data["cod_accuracy_ci_upper"] = None
        
        return data

    model_config = {"validate_assignment": True}