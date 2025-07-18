"""Configuration management for VA data processing pipeline."""

import logging
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for VA data processing.

    This configuration class manages all settings for the VA data processing pipeline,
    including data paths, encoding options, and output configurations.
    """

    data_path: str = Field(..., description="Path to PHMRC CSV file")
    output_dir: str = Field("results/baseline/", description="Output directory")
    openva_encoding: bool = Field(
        False, description="Apply OpenVA encoding for InSilicoVA"
    )
    drop_columns: Optional[List[str]] = Field(
        default_factory=list, description="Columns to drop during processing"
    )
    stratify_by_site: bool = Field(True, description="Enable site-based stratification")
    
    # Data splitting parameters
    split_strategy: Literal["train_test", "cross_site", "stratified_site"] = Field(
        default="train_test", description="Strategy for splitting the data"
    )
    test_size: float = Field(
        default=0.3, ge=0.1, le=0.5, description="Proportion of test data (0.1-0.5)"
    )
    random_state: int = Field(
        default=42, description="Random seed for reproducible splits"
    )
    site_column: str = Field(
        default="site", description="Column name containing site information"
    )
    label_column: str = Field(
        default="va34", description="Column name containing target labels"
    )
    train_sites: Optional[List[str]] = Field(
        default=None, description="Specific sites to use for training (cross_site strategy)"
    )
    test_sites: Optional[List[str]] = Field(
        default=None, description="Specific sites to use for testing (cross_site strategy)"
    )
    min_samples_per_class: int = Field(
        default=5, ge=1, description="Minimum samples required per class"
    )
    handle_small_classes: Literal["error", "warn", "exclude"] = Field(
        default="warn", description="How to handle classes with insufficient samples"
    )

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, v: str) -> str:
        """Validate that the data path exists and is a CSV file."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Data file not found: {v}")
        if not path.suffix.lower() == ".csv":
            raise ValueError(f"Data file must be a CSV file, got: {path.suffix}")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output directory exists or can be created."""
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {v}")
        return v

    def get_output_path(self, dataset_name: str, encoding_type: str) -> Path:
        """Generate output path for processed data.

        Args:
            dataset_name: Name of the dataset (e.g., 'adult', 'child', 'neonate')
            encoding_type: Type of encoding ('numeric' or 'openva')

        Returns:
            Path object for the output file
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{encoding_type}_{timestamp}.csv"
        return Path(self.output_dir) / "processed_data" / filename

    def setup_logging(self, log_level: str = "INFO") -> None:
        """Configure logging for the pipeline.

        Args:
            log_level: Logging level (e.g., 'INFO', 'DEBUG', 'WARNING')
        """
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Also log to file
        log_dir = Path(self.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        log_file = (
            log_dir / f"va_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

        logging.info(f"Logging configured. Log file: {log_file}")

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Validate test size is within acceptable range."""
        if not 0.1 <= v <= 0.5:
            raise ValueError(f"test_size must be between 0.1 and 0.5, got {v}")
        return v

    @field_validator("split_strategy")
    @classmethod
    def validate_split_strategy(cls, v: str) -> str:
        """Validate split strategy is supported."""
        valid_strategies = ["train_test", "cross_site", "stratified_site"]
        if v not in valid_strategies:
            raise ValueError(f"split_strategy must be one of {valid_strategies}, got {v}")
        return v

    @field_validator("handle_small_classes")
    @classmethod
    def validate_handle_small_classes(cls, v: str) -> str:
        """Validate small class handling strategy."""
        valid_handlers = ["error", "warn", "exclude"]
        if v not in valid_handlers:
            raise ValueError(f"handle_small_classes must be one of {valid_handlers}, got {v}")
        return v
