#!/usr/bin/env python
"""
TabICL Environment Configuration Utility

This module provides automatic environment configuration for TabICL compatibility,
especially on macOS ARM (M1/M2/M3) systems with Python 3.12.
"""

import os
import sys
import platform
import warnings
import subprocess
from typing import Dict, List, Optional, Tuple
import logging


class TabICLEnvironmentConfigurer:
    """Configures environment for optimal TabICL compatibility."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the environment configurer."""
        self.logger = logger or logging.getLogger(__name__)
        self.original_env = {}
        self.applied_configs = []
        
    def detect_system_info(self) -> Dict[str, str]:
        """Detect system information relevant to TabICL compatibility."""
        info = {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }
        
        # Check if we're on macOS ARM
        info['is_macos_arm'] = (
            info['platform'] == 'Darwin' and 
            ('arm' in info['processor'].lower() or info['architecture'] == '64bit')
        )
        
        # Check Python version compatibility
        major, minor = sys.version_info[:2]
        info['python_major_minor'] = f"{major}.{minor}"
        info['is_python_312_plus'] = (major == 3 and minor >= 12) or major > 3
        
        # Check for pytest environment
        info['in_pytest'] = 'pytest' in sys.modules
        
        return info
    
    def check_pytorch_compatibility(self) -> Dict[str, any]:
        """Check PyTorch installation and compatibility."""
        compat_info = {
            'torch_available': False,
            'torch_version': None,
            'cuda_available': False,
            'mps_available': False,
            'recommended_device': 'cpu'
        }
        
        try:
            import torch
            compat_info['torch_available'] = True
            compat_info['torch_version'] = torch.__version__
            
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                compat_info['cuda_available'] = True
                compat_info['recommended_device'] = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                compat_info['mps_available'] = True
                # Note: We don't recommend MPS due to compatibility issues
                compat_info['recommended_device'] = 'cpu'  # Force CPU for stability
            
        except ImportError:
            self.logger.warning("PyTorch not available")
        
        return compat_info
    
    def check_tabicl_availability(self) -> Dict[str, any]:
        """Check TabICL installation status."""
        tabicl_info = {
            'tabicl_available': False,
            'tabicl_version': None,
            'import_error': None
        }
        
        try:
            import tabicl
            tabicl_info['tabicl_available'] = True
            tabicl_info['tabicl_version'] = getattr(tabicl, '__version__', 'unknown')
        except ImportError as e:
            tabicl_info['import_error'] = str(e)
            self.logger.info(f"TabICL not available: {e}")
        
        return tabicl_info
    
    def get_recommended_env_vars(self, system_info: Dict[str, str]) -> Dict[str, str]:
        """Get recommended environment variables based on system info."""
        env_vars = {}
        
        # Base threading configuration for stability
        env_vars.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
        })
        
        # macOS specific configurations
        if system_info['platform'] == 'Darwin':
            env_vars.update({
                'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable MPS memory management
            })
            
            # For macOS ARM or Python 3.12+, disable MPS entirely for stability
            if system_info['is_macos_arm'] or system_info['is_python_312_plus']:
                env_vars['PYTORCH_MPS_DISABLE'] = '1'
                reason = []
                if system_info['is_macos_arm']:
                    reason.append("ARM processor")
                if system_info['is_python_312_plus']:
                    reason.append("Python 3.12+")
                self.logger.info(f"Disabling MPS for compatibility: {' + '.join(reason)}")
        
        # General Python 3.12+ configurations (any platform)
        if system_info['is_python_312_plus']:
            env_vars.update({
                'PYTORCH_DISABLE_MULTIPROCESSING': '1',  # Prevent multiprocessing issues
                'PYTHONFAULTHANDLER': '1',  # Better error reporting
            })
        
        # pytest-specific configurations
        if system_info['in_pytest']:
            env_vars.update({
                'PYTORCH_DISABLE_MULTIPROCESSING': '1',
                'PYTHONFAULTHANDLER': '1',
            })
        
        return env_vars
    
    def apply_environment_configuration(self, force: bool = False) -> bool:
        """Apply the recommended environment configuration."""
        system_info = self.detect_system_info()
        pytorch_info = self.check_pytorch_compatibility()
        recommended_vars = self.get_recommended_env_vars(system_info)
        
        self.logger.info("Applying TabICL environment configuration...")
        self.logger.info(f"System: {system_info['platform']} {system_info['architecture']}")
        self.logger.info(f"Python: {system_info['python_version']}")
        
        applied_count = 0
        for var_name, var_value in recommended_vars.items():
            current_value = os.environ.get(var_name)
            
            if current_value != var_value:
                if current_value is not None and not force:
                    self.logger.debug(f"Skipping {var_name} (already set to '{current_value}')")
                    continue
                
                # Store original value for restoration
                if var_name not in self.original_env:
                    self.original_env[var_name] = current_value
                
                os.environ[var_name] = var_value
                self.applied_configs.append(var_name)
                applied_count += 1
                self.logger.debug(f"Set {var_name}={var_value}")
        
        if applied_count > 0:
            self.logger.info(f"Applied {applied_count} environment configurations")
            
            # Configure PyTorch if available
            if pytorch_info['torch_available']:
                try:
                    import torch
                    torch.set_num_threads(1)
                    self.logger.debug("Set PyTorch thread count to 1")
                except Exception as e:
                    self.logger.warning(f"Failed to configure PyTorch: {e}")
        
        return applied_count > 0
    
    def restore_environment(self):
        """Restore original environment variables."""
        restored_count = 0
        for var_name in self.applied_configs:
            original_value = self.original_env.get(var_name)
            if original_value is None:
                if var_name in os.environ:
                    del os.environ[var_name]
                    restored_count += 1
            else:
                os.environ[var_name] = original_value
                restored_count += 1
        
        if restored_count > 0:
            self.logger.info(f"Restored {restored_count} environment variables")
        
        self.applied_configs.clear()
        self.original_env.clear()
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate the current configuration for TabICL compatibility."""
        issues = []
        system_info = self.detect_system_info()
        pytorch_info = self.check_pytorch_compatibility()
        tabicl_info = self.check_tabicl_availability()
        
        # Check system compatibility
        if system_info['is_macos_arm'] and system_info['is_python_312_plus']:
            if os.environ.get('PYTORCH_MPS_DISABLE') != '1':
                issues.append("MPS should be disabled for Python 3.12+ on macOS ARM")
        
        # Check threading configuration
        threading_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']
        for var in threading_vars:
            if os.environ.get(var) != '1':
                issues.append(f"{var} should be set to '1' for stability")
        
        # Check PyTorch configuration
        if not pytorch_info['torch_available']:
            issues.append("PyTorch not available - required for TabICL")
        
        # Check TabICL availability
        if not tabicl_info['tabicl_available']:
            issues.append(f"TabICL not available: {tabicl_info.get('import_error', 'Unknown error')}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_configuration_report(self) -> str:
        """Generate a comprehensive configuration report."""
        system_info = self.detect_system_info()
        pytorch_info = self.check_pytorch_compatibility()
        tabicl_info = self.check_tabicl_availability()
        is_valid, issues = self.validate_configuration()
        
        report_lines = [
            "TabICL Environment Configuration Report",
            "=" * 50,
            "",
            "System Information:",
            f"  Platform: {system_info['platform']} ({system_info['architecture']})",
            f"  Processor: {system_info['processor']}",
            f"  Python: {system_info['python_version']} ({system_info['python_implementation']})",
            f"  macOS ARM: {system_info['is_macos_arm']}",
            f"  Python 3.12+: {system_info['is_python_312_plus']}",
            f"  In pytest: {system_info['in_pytest']}",
            "",
            "PyTorch Information:",
            f"  Available: {pytorch_info['torch_available']}",
            f"  Version: {pytorch_info['torch_version']}",
            f"  CUDA Available: {pytorch_info['cuda_available']}",
            f"  MPS Available: {pytorch_info['mps_available']}",
            f"  Recommended Device: {pytorch_info['recommended_device']}",
            "",
            "TabICL Information:",
            f"  Available: {tabicl_info['tabicl_available']}",
            f"  Version: {tabicl_info['tabicl_version']}",
        ]
        
        if tabicl_info['import_error']:
            report_lines.append(f"  Import Error: {tabicl_info['import_error']}")
        
        report_lines.extend([
            "",
            "Environment Variables:",
        ])
        
        relevant_vars = [
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
            'OPENBLAS_NUM_THREADS', 'PYTORCH_ENABLE_MPS_FALLBACK', 
            'PYTORCH_MPS_DISABLE', 'PYTORCH_DISABLE_MULTIPROCESSING'
        ]
        
        for var in relevant_vars:
            value = os.environ.get(var, 'Not set')
            report_lines.append(f"  {var}: {value}")
        
        report_lines.extend([
            "",
            f"Configuration Status: {'✅ Valid' if is_valid else '❌ Issues Found'}",
        ])
        
        if issues:
            report_lines.extend([
                "",
                "Issues:",
            ])
            for issue in issues:
                report_lines.append(f"  - {issue}")
        
        return "\n".join(report_lines)
    
    def get_recovery_recommendations(self) -> List[str]:
        """Get specific recommendations to fix compatibility issues."""
        is_valid, issues = self.validate_configuration()
        recommendations = []
        
        if not is_valid:
            system_info = self.detect_system_info()
            pytorch_info = self.check_pytorch_compatibility()
            tabicl_info = self.check_tabicl_availability()
            
            # System-specific recommendations
            if system_info['is_macos_arm'] and system_info['is_python_312_plus']:
                recommendations.append(
                    "Set PYTORCH_MPS_DISABLE=1 to prevent segmentation faults on macOS ARM with Python 3.12+"
                )
            
            # Threading recommendations
            if any('NUM_THREADS' in issue for issue in issues):
                recommendations.append(
                    "Set thread limits: export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
                )
            
            # PyTorch recommendations
            if not pytorch_info['torch_available']:
                recommendations.append(
                    "Install PyTorch 2.2.0 or newer: pip install torch==2.2.0"
                )
            
            # TabICL recommendations
            if not tabicl_info['tabicl_available']:
                recommendations.append(
                    "Install TabICL: pip install tabicl"
                )
                recommendations.append(
                    "Or use SafeTabICLClassifier with fallback enabled (automatic with TabICLModel)"
                )
            
            # General stability recommendations
            if system_info['is_macos_arm'] or system_info['is_python_312_plus']:
                recommendations.append(
                    "Consider checking TabICL installation and compatibility"
                )
        
        return recommendations


# Convenience function for global configuration
_global_configurer = None

def configure_tabicl_environment(force: bool = False, logger: Optional[logging.Logger] = None) -> bool:
    """
    Configure environment for TabICL compatibility globally.
    
    Args:
        force: Whether to override existing environment variables
        logger: Optional logger for configuration messages
        
    Returns:
        True if configuration was applied, False otherwise
    """
    global _global_configurer
    
    if _global_configurer is None:
        _global_configurer = TabICLEnvironmentConfigurer(logger)
    
    return _global_configurer.apply_environment_configuration(force=force)


def get_tabicl_config_report() -> str:
    """Get a configuration report for TabICL environment."""
    configurer = TabICLEnvironmentConfigurer()
    return configurer.get_configuration_report()


def validate_tabicl_environment() -> Tuple[bool, List[str]]:
    """Validate the current environment for TabICL compatibility."""
    configurer = TabICLEnvironmentConfigurer()
    return configurer.validate_configuration()


if __name__ == "__main__":
    # Command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure environment for TabICL")
    parser.add_argument("--report", action="store_true", help="Show configuration report")
    parser.add_argument("--configure", action="store_true", help="Apply configuration")
    parser.add_argument("--force", action="store_true", help="Force override existing variables")
    parser.add_argument("--validate", action="store_true", help="Validate current configuration")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    configurer = TabICLEnvironmentConfigurer()
    
    if args.report or not any([args.configure, args.validate]):
        print(configurer.get_configuration_report())
    
    if args.validate:
        is_valid, issues = configurer.validate_configuration()
        print(f"\nValidation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
    
    if args.configure:
        success = configurer.apply_environment_configuration(force=args.force)
        if success:
            print("\n✅ Environment configured successfully")
        else:
            print("\nℹ️  No configuration changes needed")