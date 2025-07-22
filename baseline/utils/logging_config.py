"""Centralized logging configuration for VA pipeline.

This module provides standardized logging setup across all VA processing
components, ensuring consistent log formatting and organization.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    log_dir: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    component: Optional[str] = None,
) -> logging.Logger:
    """Set up standardized logging configuration.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_dir: Directory to save log files (defaults to 'logs/')
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        file: Whether to log to file
        component: Component subdirectory (e.g., 'baseline', 'model_comparison')

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        # Set up log directory
        if log_dir is None:
            log_dir = "logs"
        
        # Add component subdirectory if specified
        if component:
            log_dir = os.path.join(log_dir, component)
        
        # Create directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        module_name = name.split('.')[-1]  # Get last part of module name
        log_filename = f"{module_name}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Log the log file location at startup
        logger.info(f"Logging to: {log_path}")
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """Convenience function to get a logger with default settings.

    Args:
        name: Logger name
        **kwargs: Additional arguments passed to setup_logging

    Returns:
        Configured logger instance
    """
    return setup_logging(name, **kwargs)