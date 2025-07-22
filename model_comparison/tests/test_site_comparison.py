"""Tests for site comparison experiment."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from model_comparison.experiments.site_comparison import SiteComparisonExperiment
from model_comparison.experiments.experiment_config import ExperimentConfig


class TestSiteComparison:
    """Test site comparison experiment functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample VA data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_sites = 4
        n_causes = 10  # Reduced for testing

        # Create realistic site distribution
        sites = []
        for i in range(n_sites):
            sites.extend([f"site_{i}"] * (n_samples // n_sites))

        data = pd.DataFrame(
            {
                "site": sites[:n_samples],
                "cause": np.random.choice(
                    [f"cause_{i}" for i in range(n_causes)], n_samples
                ),
                "age": np.random.randint(0, 100, n_samples),
                "sex": np.random.choice(["male", "female"], n_samples),
            }
        )

        # Add numeric features (symptoms)
        for i in range(20):
            data[f"symptom_{i}"] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

        return data

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        return ExperimentConfig(
            data_path=str(tmp_path / "test_data.csv"),
            sites=["site_0", "site_1", "site_2"],
            training_sizes=[0.5, 1.0],
            models=["xgboost"],  # Only test one model for speed
            n_bootstrap=10,  # Reduced for testing
            output_dir=str(tmp_path / "results"),
            generate_plots=False,  # Skip plots in tests
        )

    def test_experiment_initialization(self, test_config):
        """Test experiment initialization."""
        experiment = SiteComparisonExperiment(test_config)

        assert experiment.config == test_config
        assert experiment.processor is not None
        assert Path(test_config.output_dir).exists()

    def test_load_data(self, sample_data, test_config):
        """Test data loading functionality."""
        # Save sample data
        sample_data.to_csv(test_config.data_path, index=False)

        experiment = SiteComparisonExperiment(test_config)
        loaded_data = experiment._load_data()

        # Check data is filtered to specified sites
        assert set(loaded_data["site"].unique()) == set(test_config.sites)
        assert "cause" in loaded_data.columns
        assert len(loaded_data) > 0

    def test_get_model(self, test_config):
        """Test model instantiation."""
        experiment = SiteComparisonExperiment(test_config)

        # Test valid models
        xgb_model = experiment._get_model("xgboost")
        assert xgb_model is not None

        insilico_model = experiment._get_model("insilico")
        assert insilico_model is not None

        # Test invalid model
        with pytest.raises(ValueError, match="Unknown model"):
            experiment._get_model("invalid_model")

    def test_in_domain_experiment(self, sample_data, test_config, tmp_path):
        """Test in-domain experiment execution."""
        # Prepare data
        sample_data.to_csv(test_config.data_path, index=False)

        experiment = SiteComparisonExperiment(test_config)
        experiment._load_data = lambda: sample_data[
            sample_data["site"].isin(test_config.sites)
        ]

        # Run in-domain experiments
        results = experiment._run_in_domain_experiments(
            sample_data[sample_data["site"].isin(test_config.sites)]
        )

        # Check results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(test_config.sites) * len(test_config.models)
        assert all(results["experiment_type"] == "in_domain")
        assert all(results["train_site"] == results["test_site"])

        # Check metrics are present
        required_metrics = ["csmf_accuracy", "cod_accuracy", "csmf_accuracy_ci_lower"]
        for metric in required_metrics:
            assert metric in results.columns

    def test_out_domain_experiment(self, sample_data, test_config):
        """Test out-domain experiment execution."""
        experiment = SiteComparisonExperiment(test_config)

        # Filter data to test sites
        test_data = sample_data[sample_data["site"].isin(test_config.sites)]

        # Run out-domain experiments
        results = experiment._run_out_domain_experiments(test_data)

        # Check results
        assert isinstance(results, pd.DataFrame)
        assert all(results["experiment_type"] == "out_domain")
        assert all(results["train_site"] != results["test_site"])

    def test_training_size_experiment(self, sample_data, test_config):
        """Test training size impact experiment."""
        experiment = SiteComparisonExperiment(test_config)

        # Use first site for testing
        site_data = sample_data[sample_data["site"] == test_config.sites[0]]

        # Run training size experiments
        results = experiment._run_training_size_experiments(site_data)

        if not results.empty:
            assert all(results["experiment_type"] == "training_size")
            assert "training_fraction" in results.columns
            assert all(results["training_fraction"].isin(test_config.training_sizes))

    def test_evaluate_model(self, sample_data, test_config):
        """Test model evaluation function."""
        experiment = SiteComparisonExperiment(test_config)

        # Prepare train/test data
        train_data = sample_data.iloc[:800]
        test_data = sample_data.iloc[800:]

        X_train = train_data.drop(columns=["cause", "site"])
        y_train = train_data["cause"]
        X_test = test_data.drop(columns=["cause", "site"])
        y_test = test_data["cause"]

        # Evaluate model
        result = experiment._evaluate_model(
            model_name="xgboost",
            train_data=(X_train, y_train),
            test_data=(X_test, y_test),
            experiment_type="test",
            train_site="site_0",
            test_site="site_1",
        )

        # Check result structure
        assert isinstance(result, dict)
        assert result["experiment_type"] == "test"
        assert result["model"] == "xgboost"
        assert "csmf_accuracy" in result
        assert "cod_accuracy" in result
        assert 0 <= result["csmf_accuracy"] <= 1
        assert 0 <= result["cod_accuracy"] <= 1

    def test_save_results(self, sample_data, test_config, tmp_path):
        """Test results saving functionality."""
        experiment = SiteComparisonExperiment(test_config)

        # Create dummy results
        results = pd.DataFrame(
            {
                "experiment_type": ["in_domain", "out_domain"],
                "model": ["xgboost", "xgboost"],
                "csmf_accuracy": [0.8, 0.7],
                "cod_accuracy": [0.75, 0.65],
                "train_site": ["site_0", "site_0"],
                "test_site": ["site_0", "site_1"],
                "n_train": [100, 100],
                "n_test": [50, 50],
            }
        )

        # Save results
        experiment._save_results(results)

        # Check files were created
        output_dir = Path(test_config.output_dir)
        assert (output_dir / "full_results.csv").exists()
        assert (output_dir / "in_domain_results.csv").exists()
        assert (output_dir / "out_domain_results.csv").exists()
        assert (output_dir / "summary_statistics.csv").exists()

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid training sizes
        with pytest.raises(ValueError, match="between 0 and 1"):
            ExperimentConfig(
                data_path="test.csv",
                sites=["site_1"],
                training_sizes=[0, 1.5],  # Invalid
            )

        # Test invalid model
        with pytest.raises(ValueError, match="Model"):
            ExperimentConfig(
                data_path="test.csv",
                sites=["site_1"],
                models=["invalid_model"],
            )

        # Test invalid label type
        with pytest.raises(ValueError, match="Label type"):
            ExperimentConfig(
                data_path="test.csv",
                sites=["site_1"],
                label_type="invalid",
            )
