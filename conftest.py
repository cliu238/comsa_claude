"""Pytest configuration file."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Also add va-data to the path for imports
va_data_path = project_root / "va-data"
if va_data_path.exists():
    # Add the va-data directory so imports work
    sys.path.insert(0, str(va_data_path))
    # Also create a symlink-like reference for the module
    import va_data