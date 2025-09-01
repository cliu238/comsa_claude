#!/usr/bin/env python
"""
Clean and reorganize VA experiment results by removing CI columns and improving format.

This script:
1. Loads COD5 and VA34 result files
2. Removes confidence interval columns
3. Removes columns with all null values
4. Reorganizes columns in logical groups
5. Saves cleaned versions with better formatting
"""

import pandas as pd
from pathlib import Path


def clean_results(input_path: str, output_path: str, label_type: str) -> pd.DataFrame:
    """Clean and reorganize experiment results.
    
    Args:
        input_path: Path to raw results CSV
        output_path: Path for cleaned output CSV
        label_type: Either 'cod5' or 'va34' for display
        
    Returns:
        Cleaned DataFrame
    """
    print(f"\nProcessing {label_type.upper()} results...")
    print(f"Input: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    
    # Remove CI columns
    ci_columns = [
        'csmf_accuracy_ci', 'cod_accuracy_ci',
        'csmf_accuracy_ci_lower', 'csmf_accuracy_ci_upper',
        'cod_accuracy_ci_lower', 'cod_accuracy_ci_upper'
    ]
    df = df.drop(columns=[col for col in ci_columns if col in df.columns])
    
    # Remove columns with all nulls
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        print(f"Removing null columns: {null_columns}")
        df = df.drop(columns=null_columns)
    
    # Define logical column ordering
    column_order = [
        # Experiment identification
        'experiment_id',
        'experiment_type',
        'model',
        'random_seed',
        
        # Site configuration
        'train_site',
        'test_site',
        
        # Training configuration
        'training_fraction',
        
        # Performance metrics (main results)
        'csmf_accuracy',
        'cod_accuracy',
        
        # Dataset information
        'n_train',
        'n_test',
        
        # Execution timing
        'execution_time_seconds',
        'training_time_seconds',
        'inference_time_seconds',
        
        # Technical metadata
        'worker_id',
        'retry_count'
    ]
    
    # Reorder columns (keep only those that exist)
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # Sort by model, experiment_type, then sites for better organization
    df = df.sort_values(['model', 'experiment_type', 'train_site', 'test_site', 'random_seed'])
    df = df.reset_index(drop=True)
    
    # Save cleaned version
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Cleaned shape: {df.shape}")
    print(f"Output saved: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary for {label_type.upper()}:")
    print(f"- Total experiments: {len(df)}")
    print(f"- Unique models: {df['model'].nunique()} - {df['model'].unique().tolist()}")
    print(f"- Experiment types: {df['experiment_type'].unique().tolist()}")
    print(f"- Random seeds: {sorted(df['random_seed'].unique())}")
    print(f"- Mean CSMF accuracy: {df['csmf_accuracy'].mean():.4f}")
    print(f"- Mean COD accuracy: {df['cod_accuracy'].mean():.4f}")
    
    return df


def main():
    """Process both COD5 and VA34 results."""
    
    # Define file paths
    base_dir = Path("/Users/ericliu/projects5/context-engineering-intro/results")
    
    datasets = [
        {
            'input': base_dir / "cod5_multi_seed_5models_v2" / "cod5_comparison_results.csv",
            'output': base_dir / "cod5_multi_seed_5models_v2" / "cod5_comparison_results_clean.csv",
            'label': 'cod5'
        },
        {
            'input': base_dir / "va34_multi_seed_5models_v2" / "va34_comparison_results.csv",
            'output': base_dir / "va34_multi_seed_5models_v2" / "va34_comparison_results_clean.csv",
            'label': 'va34'
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        if dataset['input'].exists():
            clean_results(
                str(dataset['input']),
                str(dataset['output']),
                dataset['label']
            )
        else:
            print(f"Warning: File not found - {dataset['input']}")
    
    print("\n" + "="*60)
    print("✓ Cleaning complete! Cleaned files saved with '_clean' suffix")
    print("="*60)


if __name__ == "__main__":
    main()