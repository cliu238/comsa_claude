"""Configuration management for VA data processing pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Literal

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
    
    # Data splitting configuration
    split_strategy: Literal["train_test", "cross_site", "stratified_site"] = Field(
        "train_test", description="Data splitting strategy"
    )
    test_size: float = Field(0.3, description="Test size ratio (0.0-1.0)")
    random_state: int = Field(42, description="Random seed for reproducibility")
    site_column: str = Field("site", description="Column name containing site information")
    label_column: str = Field("va34", description="Column name containing target labels")
    train_sites: Optional[List[str]] = Field(
        None, description="Specific sites to use for training (cross_site mode)"
    )
    test_sites: Optional[List[str]] = Field(
        None, description="Specific sites to use for testing (cross_site mode)"
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

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Validate test size is between 0.0 and 1.0."""
        if not 0.0 < v < 1.0:
            raise ValueError(f"test_size must be between 0.0 and 1.0, got: {v}")
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
