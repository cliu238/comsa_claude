"""Tests for orchestration components."""

import tempfile

import pytest

from model_comparison.orchestration.checkpoint_manager import CheckpointManager
from model_comparison.orchestration.config import (
    CheckpointState,
    ExperimentResult,
    ParallelConfig,
)
from model_comparison.monitoring.progress_tracker import (
    PerformanceMonitor,
    ProgressTracker,
)


class TestParallelConfig:
    """Test ParallelConfig validation and methods."""

    def test_valid_config(self):
        """Test creating valid configuration."""
        config = ParallelConfig(
            n_workers=4,
            memory_per_worker="2GB",
            object_store_memory="8GB",
            batch_size=100,
        )

        assert config.n_workers == 4
        assert config.memory_per_worker == "2GB"
        assert config.batch_size == 100

    def test_memory_validation(self):
        """Test memory format validation."""
        # Valid formats
        ParallelConfig(memory_per_worker="512MB")
        ParallelConfig(memory_per_worker="4GB")
        ParallelConfig(memory_per_worker="1024KB")

        # Invalid formats
        with pytest.raises(ValueError):
            ParallelConfig(memory_per_worker="4G")  # Missing B

        with pytest.raises(ValueError):
            ParallelConfig(memory_per_worker="invalid")

    def test_ray_init_kwargs(self):
        """Test conversion to Ray initialization kwargs."""
        config = ParallelConfig(
            n_workers=8,
            object_store_memory="16GB",
            ray_dashboard=True,
            ray_dashboard_port=8266,
        )

        kwargs = config.to_ray_init_kwargs()

        assert kwargs["num_cpus"] == 8
        assert kwargs["object_store_memory"] == 16 * 1024**3  # 16GB in bytes
        assert kwargs["dashboard_host"] == "0.0.0.0"
        assert kwargs["dashboard_port"] == 8266

    def test_memory_conversion(self):
        """Test memory string to bytes conversion."""
        config = ParallelConfig()

        assert config._memory_to_bytes("1KB") == 1024
        assert config._memory_to_bytes("1MB") == 1024**2
        assert config._memory_to_bytes("1GB") == 1024**3
        assert config._memory_to_bytes("1.5GB") == int(1.5 * 1024**3)


class TestCheckpointManager:
    """Test checkpoint management functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_results(self):
        """Create sample experiment results."""
        results = []
        for i in range(5):
            results.append(
                ExperimentResult(
                    experiment_id=f"exp_{i}",
                    model_name="xgboost",
                    experiment_type="in_domain",
                    train_site="site_1",
                    test_site="site_1",
                    csmf_accuracy=0.8 + i * 0.01,
                    cod_accuracy=0.7 + i * 0.01,
                    execution_time_seconds=1.5,
                )
            )
        return results

    def test_save_and_load_checkpoint(self, temp_checkpoint_dir, sample_results):
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        config = {"test": "config", "version": 1}

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            results=sample_results,
            config=config,
            total_experiments=10,
            elapsed_seconds=100.0,
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint(config)
        assert loaded_checkpoint is not None
        assert len(loaded_checkpoint.completed_experiments) == 5
        assert loaded_checkpoint.total_experiments == 10
        assert loaded_checkpoint.elapsed_seconds == 100.0

    def test_checkpoint_compatibility(self, temp_checkpoint_dir):
        """Test checkpoint config compatibility checking."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        config1 = {"test": "config", "version": 1}
        config2 = {"test": "config", "version": 2}  # Different config

        # Save with config1
        manager.save_checkpoint([], config1, 0, 0.0)

        # Load with same config - should work
        checkpoint = manager.load_checkpoint(config1)
        assert checkpoint is not None

        # Load with different config - should return None
        checkpoint = manager.load_checkpoint(config2)
        assert checkpoint is None

    def test_load_partial_results(self, temp_checkpoint_dir, sample_results):
        """Test loading partial results from checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        config = {"test": "config"}

        # Save checkpoint with results
        manager.save_checkpoint(sample_results, config, 10, 100.0)

        # Load checkpoint and results
        checkpoint = manager.load_checkpoint(config)
        loaded_results = manager.load_partial_results(checkpoint)

        assert len(loaded_results) == 5
        assert all(isinstance(r, ExperimentResult) for r in loaded_results)
        assert loaded_results[0].experiment_id == "exp_0"

    def test_create_experiment_id(self, temp_checkpoint_dir):
        """Test experiment ID creation."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        exp_id = manager.create_experiment_id(
            model_name="xgboost",
            experiment_type="in_domain",
            train_site="site_1",
            test_site="site_1",
            training_size=0.5,
        )

        assert exp_id == "xgboost_in_domain_site_1_site_1_size0.5"

    def test_filter_completed_experiments(self, temp_checkpoint_dir):
        """Test filtering completed experiments."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Create checkpoint with some completed experiments
        checkpoint = CheckpointState.from_config(
            config={"test": "config"},
            completed_experiments=["exp_1", "exp_3"],
            total_experiments=5,
        )

        # Create experiment list
        experiments = [
            {
                "model_name": "xgboost",
                "experiment_metadata": {
                    "experiment_type": "test",
                    "train_site": "site_1",
                    "test_site": "site_1",
                },
            }
            for _ in range(5)
        ]

        # Add IDs that match completed ones
        for i, exp in enumerate(experiments):
            exp["experiment_metadata"]["experiment_id"] = f"exp_{i}"

        # Filter
        remaining = manager.filter_completed_experiments(experiments, checkpoint)

        # Should have 3 remaining (0, 2, 4)
        assert len(remaining) == 3


class TestProgressTracker:
    """Test progress tracking functionality."""

    def test_basic_progress_tracking(self):
        """Test basic progress tracking operations."""
        tracker = ProgressTracker(total_experiments=10, show_progress_bar=False)

        # Update with some results
        for i in range(5):
            result = ExperimentResult(
                experiment_id=f"exp_{i}",
                model_name="xgboost",
                experiment_type="test",
                train_site="site_1",
                test_site="site_1",
                csmf_accuracy=0.8,
                cod_accuracy=0.7,
                execution_time_seconds=1.0,
            )
            tracker.update(result)

        assert tracker.completed_experiments == 5
        assert tracker.get_completion_percentage() == 50.0

        # Get statistics
        stats = tracker.get_statistics()
        assert stats["completed_experiments"] == 5
        assert stats["failed_experiments"] == 0
        assert stats["success_rate"] == 1.0

    def test_progress_with_failures(self):
        """Test progress tracking with failed experiments."""
        tracker = ProgressTracker(total_experiments=10, show_progress_bar=False)

        # Add successful result
        success_result = ExperimentResult(
            experiment_id="success",
            model_name="xgboost",
            experiment_type="test",
            train_site="site_1",
            test_site="site_1",
            csmf_accuracy=0.8,
            cod_accuracy=0.7,
            execution_time_seconds=1.0,
        )
        tracker.update(success_result)

        # Add failed result
        failed_result = ExperimentResult(
            experiment_id="failed",
            model_name="xgboost",
            experiment_type="test",
            train_site="site_1",
            test_site="site_1",
            csmf_accuracy=0.0,
            cod_accuracy=0.0,
            execution_time_seconds=0.5,
            error="Test error",
        )
        tracker.update(failed_result)

        assert tracker.completed_experiments == 2
        assert tracker.failed_experiments == 1

        stats = tracker.get_statistics()
        assert stats["success_rate"] == 0.5

    def test_metric_statistics(self):
        """Test calculation of metric statistics."""
        tracker = ProgressTracker(total_experiments=3, show_progress_bar=False)

        # Add results with varying metrics
        for i, (csmf, cod) in enumerate([(0.7, 0.6), (0.8, 0.7), (0.9, 0.8)]):
            result = ExperimentResult(
                experiment_id=f"exp_{i}",
                model_name="xgboost",
                experiment_type="test",
                train_site="site_1",
                test_site="site_1",
                csmf_accuracy=csmf,
                cod_accuracy=cod,
                execution_time_seconds=1.0,
            )
            tracker.update(result)

        stats = tracker.get_statistics()
        assert abs(stats["avg_csmf_accuracy"] - 0.8) < 0.01
        assert abs(stats["avg_cod_accuracy"] - 0.7) < 0.01
        assert stats["min_csmf_accuracy"] == 0.7
        assert stats["max_csmf_accuracy"] == 0.9


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_performance_recording(self):
        """Test recording experiment performance."""
        monitor = PerformanceMonitor()

        # Record some experiments
        for i in range(5):
            result = ExperimentResult(
                experiment_id=f"exp_{i}",
                model_name="xgboost",
                experiment_type="test",
                train_site="site_1",
                test_site="site_1",
                csmf_accuracy=0.8,
                cod_accuracy=0.7,
                execution_time_seconds=1.0 + i * 0.1,
                worker_id=f"worker_{i % 2}",
            )
            monitor.record_experiment(result)

        summary = monitor.get_performance_summary()

        assert summary["total_experiments"] == 5
        assert abs(summary["avg_experiment_seconds"] - 1.2) < 0.01
        assert summary["min_experiment_seconds"] == 1.0
        assert summary["max_experiment_seconds"] == 1.4
        assert summary["worker_count"] == 2

    def test_worker_load_balancing(self):
        """Test worker load balancing metrics."""
        monitor = PerformanceMonitor()

        # Simulate uneven load distribution
        worker_assignments = ["worker_0"] * 8 + ["worker_1"] * 2

        for i, worker in enumerate(worker_assignments):
            result = ExperimentResult(
                experiment_id=f"exp_{i}",
                model_name="xgboost",
                experiment_type="test",
                train_site="site_1",
                test_site="site_1",
                csmf_accuracy=0.8,
                cod_accuracy=0.7,
                execution_time_seconds=1.0,
                worker_id=worker,
            )
            monitor.record_experiment(result)

        summary = monitor.get_performance_summary()
        load_balance = summary["worker_load_balance"]

        assert load_balance["min_experiments"] == 2
        assert load_balance["max_experiments"] == 8
        assert load_balance["avg_experiments"] == 5.0