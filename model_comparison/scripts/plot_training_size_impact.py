#!/usr/bin/env python
"""Generate plots showing the impact of training data size on model performance.

This script creates comprehensive visualizations of how training data size affects
both COD (individual) and CSMF (population) accuracy for XGBoost, TabICL, and InSilico models.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(csv_path):
    """Load and filter data for training size experiments."""
    df = pd.read_csv(csv_path)
    
    # Filter for AP site training_size experiments only
    training_size_df = df[
        (df['experiment_type'] == 'training_size') & 
        (df['train_site'] == 'AP') & 
        (df['test_site'] == 'AP')
    ].copy()
    
    # Sort by model and training fraction for consistent plotting
    training_size_df = training_size_df.sort_values(['model', 'training_fraction'])
    
    return training_size_df

def plot_learning_curves(df, output_dir):
    """Generate learning curves for CSMF and COD accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = df['model'].unique()
    colors = {'xgboost': '#1f77b4', 'tabicl': '#ff7f0e', 'insilico': '#2ca02c'}
    markers = {'xgboost': 'o', 'tabicl': 's', 'insilico': '^'}
    
    for model in models:
        model_df = df[df['model'] == model]
        
        # CSMF Accuracy
        yerr_lower = np.abs(model_df['csmf_accuracy'] - model_df['csmf_accuracy_ci_lower']).values
        yerr_upper = np.abs(model_df['csmf_accuracy_ci_upper'] - model_df['csmf_accuracy']).values
        
        ax1.errorbar(
            model_df['training_fraction'].values * 100,
            model_df['csmf_accuracy'].values,
            yerr=[yerr_lower, yerr_upper],
            label=model.capitalize(),
            marker=markers[model],
            markersize=8,
            linewidth=2,
            capsize=5,
            color=colors[model],
            alpha=0.8
        )
        
        # COD Accuracy
        yerr_lower = np.abs(model_df['cod_accuracy'] - model_df['cod_accuracy_ci_lower']).values
        yerr_upper = np.abs(model_df['cod_accuracy_ci_upper'] - model_df['cod_accuracy']).values
        
        ax2.errorbar(
            model_df['training_fraction'].values * 100,
            model_df['cod_accuracy'].values,
            yerr=[yerr_lower, yerr_upper],
            label=model.capitalize(),
            marker=markers[model],
            markersize=8,
            linewidth=2,
            capsize=5,
            color=colors[model],
            alpha=0.8
        )
    
    # Highlight the anomalous 5% point for XGBoost
    xgb_5pct = df[(df['model'] == 'xgboost') & (df['training_fraction'] == 0.05)]
    if not xgb_5pct.empty:
        ax1.scatter(5, xgb_5pct['csmf_accuracy'].values[0], 
                   s=200, color='red', alpha=0.3, zorder=5)
        ax1.annotate('Anomalous\nXGBoost\n5% point', 
                    xy=(5, xgb_5pct['csmf_accuracy'].values[0]),
                    xytext=(10, 0.55),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                    fontsize=10, color='red')
    
    # Formatting
    ax1.set_xlabel('Training Data Size (%)', fontsize=12)
    ax1.set_ylabel('CSMF Accuracy', fontsize=12)
    ax1.set_title('Population-Level Accuracy vs Training Size', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0.45, 0.95)
    
    ax2.set_xlabel('Training Data Size (%)', fontsize=12)
    ax2.set_ylabel('COD Accuracy', fontsize=12)
    ax2.set_title('Individual-Level Accuracy vs Training Size', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0.1, 0.55)
    
    plt.suptitle('Learning Curves: Impact of Training Data Size on Model Performance', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()

def plot_model_comparison_bars(df, output_dir):
    """Generate bar plots comparing models at each training size."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Prepare data for plotting
    training_sizes = sorted(df['training_fraction'].unique())
    models = ['xgboost', 'tabicl', 'insilico']
    
    x = np.arange(len(training_sizes))
    width = 0.25
    
    colors = {'xgboost': '#1f77b4', 'tabicl': '#ff7f0e', 'insilico': '#2ca02c'}
    
    # CSMF Accuracy bars
    for i, model in enumerate(models):
        model_df = df[df['model'] == model].sort_values('training_fraction')
        values = model_df['csmf_accuracy'].values
        errors = [
            np.abs(model_df['csmf_accuracy'].values - model_df['csmf_accuracy_ci_lower'].values),
            np.abs(model_df['csmf_accuracy_ci_upper'].values - model_df['csmf_accuracy'].values)
        ]
        
        bars = ax1.bar(x + i*width, values, width, label=model.capitalize(),
                      color=colors[model], alpha=0.8)
        ax1.errorbar(x + i*width, values, yerr=errors, fmt='none', 
                    color='black', alpha=0.5, capsize=3)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if training_sizes[j] == 0.05:  # Highlight 5% values
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold',
                        color='red' if model == 'xgboost' else 'black')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # COD Accuracy bars
    for i, model in enumerate(models):
        model_df = df[df['model'] == model].sort_values('training_fraction')
        values = model_df['cod_accuracy'].values
        errors = [
            np.abs(model_df['cod_accuracy'].values - model_df['cod_accuracy_ci_lower'].values),
            np.abs(model_df['cod_accuracy_ci_upper'].values - model_df['cod_accuracy'].values)
        ]
        
        bars = ax2.bar(x + i*width, values, width, label=model.capitalize(),
                      color=colors[model], alpha=0.8)
        ax2.errorbar(x + i*width, values, yerr=errors, fmt='none',
                    color='black', alpha=0.5, capsize=3)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax1.set_ylabel('CSMF Accuracy', fontsize=12)
    ax1.set_title('CSMF Accuracy Comparison Across Training Sizes', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{int(s*100)}%' for s in training_sizes])
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    ax2.set_xlabel('Training Data Size', fontsize=12)
    ax2.set_ylabel('COD Accuracy', fontsize=12)
    ax2.set_title('COD Accuracy Comparison Across Training Sizes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'{int(s*100)}%' for s in training_sizes])
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 0.6)
    
    plt.suptitle('Model Performance Comparison at Different Training Sizes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'model_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison bars to {output_path}")
    plt.close()

def plot_csmf_vs_cod_scatter(df, output_dir):
    """Generate scatter plot of CSMF vs COD accuracy."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'xgboost': '#1f77b4', 'tabicl': '#ff7f0e', 'insilico': '#2ca02c'}
    markers = {'xgboost': 'o', 'tabicl': 's', 'insilico': '^'}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # Create scatter plot with size based on training fraction
        sizes = model_df['training_fraction'] * 500
        scatter = ax.scatter(
            model_df['cod_accuracy'],
            model_df['csmf_accuracy'],
            s=sizes,
            c=[colors[model]],
            marker=markers[model],
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
            label=model.capitalize()
        )
        
        # Add annotations for 5% training points
        for _, row in model_df[model_df['training_fraction'] == 0.05].iterrows():
            ax.annotate(
                f'{model.capitalize()}\n5%',
                xy=(row['cod_accuracy'], row['csmf_accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color=colors[model],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        # Connect points for each model
        model_df_sorted = model_df.sort_values('training_fraction')
        ax.plot(model_df_sorted['cod_accuracy'], model_df_sorted['csmf_accuracy'],
               color=colors[model], alpha=0.3, linestyle='--', linewidth=1)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x (Equal accuracy)')
    
    # Highlight the anomalous region
    ax.axvspan(0.15, 0.22, ymin=0.6, ymax=0.7, alpha=0.1, color='red')
    ax.text(0.185, 0.52, 'Anomalous\nRegion', ha='center', fontsize=10, 
           color='red', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('COD Accuracy (Individual Predictions)', fontsize=12)
    ax.set_ylabel('CSMF Accuracy (Population Distribution)', fontsize=12)
    ax.set_title('Relationship Between Individual and Population Accuracy', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 0.55)
    ax.set_ylim(0.45, 0.95)
    
    # Add size legend
    sizes_legend = [0.05, 0.25, 0.5, 1.0]
    for size in sizes_legend:
        ax.scatter([], [], s=size*500, c='gray', alpha=0.5,
                  label=f'{int(size*100)}% training')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    output_path = output_dir / 'csmf_vs_cod_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved CSMF vs COD scatter plot to {output_path}")
    plt.close()

def plot_performance_gap(df, output_dir):
    """Plot the gap between CSMF and COD accuracy."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'xgboost': '#1f77b4', 'tabicl': '#ff7f0e', 'insilico': '#2ca02c'}
    markers = {'xgboost': 'o', 'tabicl': 's', 'insilico': '^'}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model].sort_values('training_fraction')
        
        # Calculate the gap
        gap = model_df['csmf_accuracy'] - model_df['cod_accuracy']
        
        ax.plot(
            model_df['training_fraction'] * 100,
            gap,
            label=model.capitalize(),
            marker=markers[model],
            markersize=8,
            linewidth=2,
            color=colors[model],
            alpha=0.8
        )
        
        # Highlight the 5% point
        if model == 'xgboost':
            gap_5pct = gap[model_df['training_fraction'] == 0.05].values
            if len(gap_5pct) > 0:
                ax.scatter(5, gap_5pct[0], s=200, color='red', alpha=0.3, zorder=5)
                ax.annotate(f'XGBoost Gap: {gap_5pct[0]:.3f}', 
                           xy=(5, gap_5pct[0]),
                           xytext=(15, gap_5pct[0] + 0.05),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                           fontsize=10, color='red', fontweight='bold')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='No gap')
    
    # Add shaded region for positive gap (CSMF > COD)
    ax.fill_between([0, 105], 0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.text(50, ax.get_ylim()[1] * 0.9, 'CSMF > COD\n(Population accuracy exceeds individual)',
           ha='center', fontsize=10, color='green', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Training Data Size (%)', fontsize=12)
    ax.set_ylabel('Accuracy Gap (CSMF - COD)', fontsize=12)
    ax.set_title('Performance Gap: Population vs Individual Accuracy', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    
    plt.tight_layout()
    
    output_path = output_dir / 'performance_gap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance gap plot to {output_path}")
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics for the analysis."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Training Size Impact Analysis")
    print("="*80)
    
    # 5% training data statistics
    print("\n5% Training Data Performance:")
    print("-" * 40)
    df_5pct = df[df['training_fraction'] == 0.05]
    for _, row in df_5pct.iterrows():
        print(f"{row['model'].capitalize():10s}: CSMF={row['csmf_accuracy']:.3f}, COD={row['cod_accuracy']:.3f}, "
              f"Gap={row['csmf_accuracy']-row['cod_accuracy']:.3f}")
    
    # 100% training data statistics
    print("\n100% Training Data Performance:")
    print("-" * 40)
    df_100pct = df[df['training_fraction'] == 1.0]
    for _, row in df_100pct.iterrows():
        print(f"{row['model'].capitalize():10s}: CSMF={row['csmf_accuracy']:.3f}, COD={row['cod_accuracy']:.3f}, "
              f"Gap={row['csmf_accuracy']-row['cod_accuracy']:.3f}")
    
    # Performance improvement from 5% to 100%
    print("\nPerformance Improvement (5% → 100%):")
    print("-" * 40)
    for model in df['model'].unique():
        model_5 = df[(df['model'] == model) & (df['training_fraction'] == 0.05)]
        model_100 = df[(df['model'] == model) & (df['training_fraction'] == 1.0)]
        
        if not model_5.empty and not model_100.empty:
            csmf_improvement = model_100['csmf_accuracy'].values[0] - model_5['csmf_accuracy'].values[0]
            cod_improvement = model_100['cod_accuracy'].values[0] - model_5['cod_accuracy'].values[0]
            print(f"{model.capitalize():10s}: CSMF +{csmf_improvement:.3f}, COD +{cod_improvement:.3f}")
    
    print("\nKey Finding:")
    print("-" * 40)
    xgb_5 = df[(df['model'] == 'xgboost') & (df['training_fraction'] == 0.05)]
    if not xgb_5.empty:
        print(f"XGBoost achieves {xgb_5['csmf_accuracy'].values[0]:.1%} CSMF accuracy with only 5% training data")
        print(f"This is {xgb_5['csmf_accuracy'].values[0] - xgb_5['cod_accuracy'].values[0]:.1%} higher than its COD accuracy!")
    print("="*80 + "\n")

def main():
    """Main function to generate all plots."""
    # Define paths
    csv_path = Path('results/insilico_xgboost_tabicl_comparison/va34_comparison_results.csv')
    output_dir = Path('results/insilico_xgboost_tabicl_comparison/plots')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("Loading data from:", csv_path)
    df = load_and_prepare_data(csv_path)
    
    print(f"Found {len(df)} training size experiments for AP site")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Training sizes: {', '.join([f'{int(x*100)}%' for x in sorted(df['training_fraction'].unique())])}")
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_learning_curves(df, output_dir)
    plot_model_comparison_bars(df, output_dir)
    plot_csmf_vs_cod_scatter(df, output_dir)
    plot_performance_gap(df, output_dir)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()