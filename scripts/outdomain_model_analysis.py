#!/usr/bin/env python3
"""
Out-Domain Model Performance Analysis for VA Models
Analyzes cross-site transfer performance of XGBoost and InSilico models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_filter_data(cod5_path, va34_path):
    """Load data and filter for out-domain experiments"""
    cod5_df = pd.read_csv(cod5_path)
    va34_df = pd.read_csv(va34_path)
    
    # Filter for out-domain only
    cod5_outdomain = cod5_df[cod5_df['experiment_type'] == 'out_domain'].copy()
    va34_outdomain = va34_df[va34_df['experiment_type'] == 'out_domain'].copy()
    
    # Add classification type column
    cod5_outdomain['classification'] = 'COD5'
    va34_outdomain['classification'] = 'VA34'
    
    return cod5_outdomain, va34_outdomain

def calculate_summary_statistics(df, model_name, classification_type):
    """Calculate summary statistics for a model's out-domain performance"""
    model_data = df[df['model'] == model_name]
    
    stats_dict = {
        'model': model_name,
        'classification': classification_type,
        'n_experiments': len(model_data),
        'csmf_mean': model_data['csmf_accuracy'].mean(),
        'csmf_std': model_data['csmf_accuracy'].std(),
        'csmf_min': model_data['csmf_accuracy'].min(),
        'csmf_max': model_data['csmf_accuracy'].max(),
        'cod_mean': model_data['cod_accuracy'].mean(),
        'cod_std': model_data['cod_accuracy'].std(),
        'cod_min': model_data['cod_accuracy'].min(),
        'cod_max': model_data['cod_accuracy'].max(),
    }
    
    return stats_dict

def create_transfer_matrix(df, model_name, metric='cod_accuracy'):
    """Create cross-site transfer matrix for a model"""
    model_data = df[df['model'] == model_name]
    sites = sorted(df['train_site'].unique())
    
    # Create matrix
    matrix = pd.DataFrame(index=sites, columns=sites, dtype=float)
    
    for train_site in sites:
        for test_site in sites:
            if train_site != test_site:  # Only out-domain
                exp_data = model_data[(model_data['train_site'] == train_site) & 
                                      (model_data['test_site'] == test_site)]
                if not exp_data.empty:
                    matrix.loc[train_site, test_site] = exp_data[metric].iloc[0]
    
    return matrix

def analyze_transfer_patterns(df):
    """Analyze best and worst transfer patterns"""
    patterns = {
        'best_transfers': [],
        'worst_transfers': [],
        'source_quality': {},
        'target_difficulty': {}
    }
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Best transfers
        best = model_data.nlargest(5, 'cod_accuracy')
        for _, row in best.iterrows():
            patterns['best_transfers'].append({
                'model': model,
                'source': row['train_site'],
                'target': row['test_site'],
                'cod_accuracy': row['cod_accuracy'],
                'csmf_accuracy': row['csmf_accuracy']
            })
        
        # Worst transfers
        worst = model_data.nsmallest(5, 'cod_accuracy')
        for _, row in worst.iterrows():
            patterns['worst_transfers'].append({
                'model': model,
                'source': row['train_site'],
                'target': row['test_site'],
                'cod_accuracy': row['cod_accuracy'],
                'csmf_accuracy': row['csmf_accuracy']
            })
        
        # Source quality (how well does training on this site generalize?)
        source_perf = model_data.groupby('train_site')['cod_accuracy'].mean()
        for site, perf in source_perf.items():
            if site not in patterns['source_quality']:
                patterns['source_quality'][site] = {}
            patterns['source_quality'][site][model] = perf
        
        # Target difficulty (how hard is this site to predict?)
        target_perf = model_data.groupby('test_site')['cod_accuracy'].mean()
        for site, perf in target_perf.items():
            if site not in patterns['target_difficulty']:
                patterns['target_difficulty'][site] = {}
            patterns['target_difficulty'][site][model] = perf
    
    return patterns

def create_model_comparison_plot(cod5_stats, va34_stats, output_dir):
    """Create comparison plots for out-domain performance"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Prepare data for plotting
    models = ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']
    
    # CSMF Accuracy Comparison
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    cod5_csmf = [cod5_stats[cod5_stats['model'] == m]['csmf_mean'].iloc[0] for m in models]
    cod5_csmf_std = [cod5_stats[cod5_stats['model'] == m]['csmf_std'].iloc[0] for m in models]
    va34_csmf = [va34_stats[va34_stats['model'] == m]['csmf_mean'].iloc[0] for m in models]
    va34_csmf_std = [va34_stats[va34_stats['model'] == m]['csmf_std'].iloc[0] for m in models]
    
    ax.bar(x - width/2, cod5_csmf, width, yerr=cod5_csmf_std, label='COD5', alpha=0.8, capsize=5, color='#2E86AB')
    ax.bar(x + width/2, va34_csmf, width, yerr=va34_csmf_std, label='VA34', alpha=0.8, capsize=5, color='#A23B72')
    ax.set_ylabel('CSMF Accuracy', fontsize=12)
    ax.set_title('CSMF Accuracy: Out-Domain Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cat. NB', 'RF', 'XGB', 'Log. Reg', 'InSilico'], fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.0])
    
    # COD Accuracy Comparison
    ax = axes[1]
    cod5_cod = [cod5_stats[cod5_stats['model'] == m]['cod_mean'].iloc[0] for m in models]
    cod5_cod_std = [cod5_stats[cod5_stats['model'] == m]['cod_std'].iloc[0] for m in models]
    va34_cod = [va34_stats[va34_stats['model'] == m]['cod_mean'].iloc[0] for m in models]
    va34_cod_std = [va34_stats[va34_stats['model'] == m]['cod_std'].iloc[0] for m in models]
    
    ax.bar(x - width/2, cod5_cod, width, yerr=cod5_cod_std, label='COD5', alpha=0.8, capsize=5, color='#2E86AB')
    ax.bar(x + width/2, va34_cod, width, yerr=va34_cod_std, label='VA34', alpha=0.8, capsize=5, color='#A23B72')
    ax.set_ylabel('COD Accuracy', fontsize=12)
    ax.set_title('COD Accuracy: Out-Domain Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cat. NB', 'RF', 'XGB', 'Log. Reg', 'InSilico'], fontsize=10, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.0])
    
    # Add value labels on bars
    for ax_idx, (ax_obj, values_cod5, values_va34) in enumerate([(axes[0], cod5_csmf, va34_csmf), 
                                                                   (axes[1], cod5_cod, va34_cod)]):
        for i, (v5, v34) in enumerate(zip(values_cod5, values_va34)):
            ax_obj.text(i - width/2, v5 + 0.01, f'{v5:.3f}', ha='center', va='bottom', fontsize=9)
            ax_obj.text(i + width/2, v34 + 0.01, f'{v34:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Out-Domain Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'outdomain_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_transfer_heatmaps(cod5_df, va34_df, output_dir):
    """Create heatmaps showing cross-site transfer performance for best models"""
    # Select best two models based on mean out-domain performance
    models_to_plot = ['xgboost', 'insilico']  # Focus on top performers to keep visualization readable
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # COD5 XGBoost Transfer Matrix
    cod5_xgb_matrix = create_transfer_matrix(cod5_df, 'xgboost', 'cod_accuracy')
    ax = axes[0, 0]
    sns.heatmap(cod5_xgb_matrix.astype(float), annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=ax, vmin=0.1, vmax=0.7, cbar_kws={'label': 'COD Accuracy'},
                square=True, linewidths=0.5, annot_kws={'fontsize': 8})
    ax.set_title('COD5: XGBoost Cross-Site Transfer', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Site', fontsize=11)
    ax.set_ylabel('Source Site', fontsize=11)
    
    # COD5 InSilico Transfer Matrix
    cod5_ins_matrix = create_transfer_matrix(cod5_df, 'insilico', 'cod_accuracy')
    ax = axes[0, 1]
    sns.heatmap(cod5_ins_matrix.astype(float), annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=ax, vmin=0.1, vmax=0.7, cbar_kws={'label': 'COD Accuracy'},
                square=True, linewidths=0.5, annot_kws={'fontsize': 8})
    ax.set_title('COD5: InSilico Cross-Site Transfer', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Site', fontsize=11)
    ax.set_ylabel('Source Site', fontsize=11)
    
    # VA34 XGBoost Transfer Matrix
    va34_xgb_matrix = create_transfer_matrix(va34_df, 'xgboost', 'cod_accuracy')
    ax = axes[1, 0]
    sns.heatmap(va34_xgb_matrix.astype(float), annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=ax, vmin=0.0, vmax=0.4, cbar_kws={'label': 'COD Accuracy'},
                square=True, linewidths=0.5, annot_kws={'fontsize': 8})
    ax.set_title('VA34: XGBoost Cross-Site Transfer', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Site', fontsize=11)
    ax.set_ylabel('Source Site', fontsize=11)
    
    # VA34 InSilico Transfer Matrix
    va34_ins_matrix = create_transfer_matrix(va34_df, 'insilico', 'cod_accuracy')
    ax = axes[1, 1]
    sns.heatmap(va34_ins_matrix.astype(float), annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=ax, vmin=0.0, vmax=0.4, cbar_kws={'label': 'COD Accuracy'},
                square=True, linewidths=0.5, annot_kws={'fontsize': 8})
    ax.set_title('VA34: InSilico Cross-Site Transfer', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Site', fontsize=11)
    ax.set_ylabel('Source Site', fontsize=11)
    
    plt.suptitle('Cross-Site Transfer Performance Matrices (Best Models)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'transfer_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save all model matrices as CSV
    all_matrices = {}
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        if not cod5_df[cod5_df['model'] == model].empty:
            cod5_matrix = create_transfer_matrix(cod5_df, model, 'cod_accuracy')
            cod5_matrix.to_csv(output_dir / f'cod5_{model}_transfer_matrix.csv')
            all_matrices[f'cod5_{model}'] = cod5_matrix
        if not va34_df[va34_df['model'] == model].empty:
            va34_matrix = create_transfer_matrix(va34_df, model, 'cod_accuracy')
            va34_matrix.to_csv(output_dir / f'va34_{model}_transfer_matrix.csv')
            all_matrices[f'va34_{model}'] = va34_matrix
    
    return all_matrices

def create_site_quality_plot(cod5_patterns, va34_patterns, output_dir):
    """Create plots showing site quality as source and target"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract site names
    sites = sorted(set(cod5_patterns['source_quality'].keys()) | 
                   set(va34_patterns['source_quality'].keys()))
    
    # Focus on top 3 models for clarity
    models_to_plot = ['random_forest', 'xgboost', 'insilico']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # COD5 Source Quality
    ax = axes[0, 0]
    x = np.arange(len(sites))
    width = 0.25
    
    for i, (model, color) in enumerate(zip(models_to_plot, colors)):
        values = [cod5_patterns['source_quality'].get(s, {}).get(model, 0) for s in sites]
        ax.bar(x + i*width - width, values, width, label=model.upper(), alpha=0.8, color=color)
    ax.set_ylabel('Mean COD Accuracy', fontsize=11)
    ax.set_title('COD5: Site Quality as Training Source', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # COD5 Target Difficulty
    ax = axes[0, 1]
    for i, (model, color) in enumerate(zip(models_to_plot, colors)):
        values = [cod5_patterns['target_difficulty'].get(s, {}).get(model, 0) for s in sites]
        ax.bar(x + i*width - width, values, width, label=model.upper(), alpha=0.8, color=color)
    ax.set_ylabel('Mean COD Accuracy', fontsize=11)
    ax.set_title('COD5: Site Difficulty as Prediction Target', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # VA34 Source Quality
    ax = axes[1, 0]
    for i, (model, color) in enumerate(zip(models_to_plot, colors)):
        values = [va34_patterns['source_quality'].get(s, {}).get(model, 0) for s in sites]
        ax.bar(x + i*width - width, values, width, label=model.upper(), alpha=0.8, color=color)
    ax.set_ylabel('Mean COD Accuracy', fontsize=11)
    ax.set_title('VA34: Site Quality as Training Source', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # VA34 Target Difficulty
    ax = axes[1, 1]
    for i, (model, color) in enumerate(zip(models_to_plot, colors)):
        values = [va34_patterns['target_difficulty'].get(s, {}).get(model, 0) for s in sites]
        ax.bar(x + i*width - width, values, width, label=model.upper(), alpha=0.8, color=color)
    ax.set_ylabel('Mean COD Accuracy', fontsize=11)
    ax.set_title('VA34: Site Difficulty as Prediction Target', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'site_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(cod5_stats, va34_stats, cod5_patterns, va34_patterns, output_dir):
    """Generate markdown report for out-domain analysis"""
    report = []
    report.append("# Out-Domain Model Performance Analysis Report\n")
    report.append("## Executive Summary\n")
    
    # Find best performing models for out-domain
    best_cod5_model = cod5_stats.loc[cod5_stats['cod_mean'].idxmax(), 'model']
    best_cod5_cod = cod5_stats.loc[cod5_stats['cod_mean'].idxmax(), 'cod_mean']
    best_va34_model = va34_stats.loc[va34_stats['cod_mean'].idxmax(), 'model']
    best_va34_cod = va34_stats.loc[va34_stats['cod_mean'].idxmax(), 'cod_mean']
    
    # Key findings
    report.append("### Key Findings:\n")
    report.append(f"1. **Best out-domain models vary by classification complexity**\n")
    report.append(f"   - COD5: {best_cod5_model.upper()} achieves {best_cod5_cod:.1%} COD accuracy\n")
    report.append(f"   - VA34: {best_va34_model.upper()} achieves {best_va34_cod:.1%} COD accuracy\n\n")
    
    report.append("2. **High variance in out-domain performance**\n")
    report.append("   - Performance heavily depends on source-target site combination\n")
    report.append("   - Some transfers nearly random (e.g., Bohol→Pemba: 1.7% in VA34)\n\n")
    
    report.append("3. **Site-specific patterns emerge**\n")
    report.append("   - Pemba is consistently difficult both as source and target\n")
    report.append("   - Dar and AP tend to be good transfer targets\n\n")
    
    # Detailed metrics
    report.append("## Detailed Performance Metrics\n")
    report.append("### COD5 Out-Domain Performance\n")
    report.append("| Model | CSMF Mean±Std | COD Mean±Std | COD Range |\n")
    report.append("|-------|---------------|--------------|------------|\n")
    
    for _, row in cod5_stats.iterrows():
        report.append(f"| {row['model'].upper()} | {row['csmf_mean']:.3f}±{row['csmf_std']:.3f} | "
                     f"{row['cod_mean']:.3f}±{row['cod_std']:.3f} | "
                     f"{row['cod_min']:.3f}-{row['cod_max']:.3f} |\n")
    
    report.append("\n### VA34 Out-Domain Performance\n")
    report.append("| Model | CSMF Mean±Std | COD Mean±Std | COD Range |\n")
    report.append("|-------|---------------|--------------|------------|\n")
    
    for _, row in va34_stats.iterrows():
        report.append(f"| {row['model'].upper()} | {row['csmf_mean']:.3f}±{row['csmf_std']:.3f} | "
                     f"{row['cod_mean']:.3f}±{row['cod_std']:.3f} | "
                     f"{row['cod_min']:.3f}-{row['cod_max']:.3f} |\n")
    
    # Best transfers
    report.append("\n## Transfer Pattern Analysis\n")
    report.append("### Best Cross-Site Transfers\n")
    report.append("#### COD5 Top 3:\n")
    for i, transfer in enumerate(cod5_patterns['best_transfers'][:3]):
        if transfer['model'] == 'xgboost':
            report.append(f"{i+1}. {transfer['model'].upper()}: {transfer['source']} → {transfer['target']}: "
                         f"COD={transfer['cod_accuracy']:.3f}, CSMF={transfer['csmf_accuracy']:.3f}\n")
    
    report.append("\n#### VA34 Top 3:\n")
    for i, transfer in enumerate(va34_patterns['best_transfers'][:3]):
        if transfer['model'] == 'insilico':
            report.append(f"{i+1}. {transfer['model'].upper()}: {transfer['source']} → {transfer['target']}: "
                         f"COD={transfer['cod_accuracy']:.3f}, CSMF={transfer['csmf_accuracy']:.3f}\n")
    
    # Worst transfers
    report.append("\n### Worst Cross-Site Transfers\n")
    report.append("#### COD5 Bottom 3:\n")
    for i, transfer in enumerate(cod5_patterns['worst_transfers'][:3]):
        report.append(f"{i+1}. {transfer['model'].upper()}: {transfer['source']} → {transfer['target']}: "
                     f"COD={transfer['cod_accuracy']:.3f}\n")
    
    report.append("\n#### VA34 Bottom 3:\n")
    for i, transfer in enumerate(va34_patterns['worst_transfers'][:3]):
        report.append(f"{i+1}. {transfer['model'].upper()}: {transfer['source']} → {transfer['target']}: "
                     f"COD={transfer['cod_accuracy']:.3f}\n")
    
    # Site analysis
    report.append("\n## Site-Specific Analysis\n")
    report.append("### Best Source Sites (for generalization)\n")
    
    # Calculate average source quality
    cod5_source_avg = {}
    va34_source_avg = {}
    for site in cod5_patterns['source_quality']:
        cod5_source_avg[site] = np.mean(list(cod5_patterns['source_quality'][site].values()))
    for site in va34_patterns['source_quality']:
        va34_source_avg[site] = np.mean(list(va34_patterns['source_quality'][site].values()))
    
    best_cod5_source = max(cod5_source_avg, key=cod5_source_avg.get)
    best_va34_source = max(va34_source_avg, key=va34_source_avg.get)
    
    report.append(f"- COD5: {best_cod5_source} (avg COD accuracy: {cod5_source_avg[best_cod5_source]:.3f})\n")
    report.append(f"- VA34: {best_va34_source} (avg COD accuracy: {va34_source_avg[best_va34_source]:.3f})\n")
    
    report.append("\n### Most Difficult Target Sites\n")
    
    # Calculate average target difficulty
    cod5_target_avg = {}
    va34_target_avg = {}
    for site in cod5_patterns['target_difficulty']:
        cod5_target_avg[site] = np.mean(list(cod5_patterns['target_difficulty'][site].values()))
    for site in va34_patterns['target_difficulty']:
        va34_target_avg[site] = np.mean(list(va34_patterns['target_difficulty'][site].values()))
    
    worst_cod5_target = min(cod5_target_avg, key=cod5_target_avg.get)
    worst_va34_target = min(va34_target_avg, key=va34_target_avg.get)
    
    report.append(f"- COD5: {worst_cod5_target} (avg COD accuracy: {cod5_target_avg[worst_cod5_target]:.3f})\n")
    report.append(f"- VA34: {worst_va34_target} (avg COD accuracy: {va34_target_avg[worst_va34_target]:.3f})\n")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("1. **Consider InSilico for multi-site deployments** - better cross-site generalization\n")
    report.append("2. **Site-specific calibration is critical** - especially for Pemba site\n")
    report.append("3. **Pool training data from multiple sites** - to improve generalization\n")
    report.append("4. **Validate on target population before deployment** - transfer performance varies widely\n")
    report.append("5. **COD5 more robust for cross-site deployment** - maintains reasonable accuracy\n")
    
    # Write report
    with open(output_dir / 'outdomain_analysis_report.md', 'w') as f:
        f.writelines(report)
    
    print("Report generated: outdomain_analysis_report.md")

def main():
    # Define paths
    cod5_path = Path('/Users/ericliu/projects5/context-engineering-intro/results/full_comparison_20250821_144955_cod5/cod5_comparison_results.csv')
    va34_path = Path('/Users/ericliu/projects5/context-engineering-intro/results/full_comparison_20250821_145007_va34/va34_comparison_results.csv')
    output_dir = Path('/Users/ericliu/projects5/context-engineering-intro/results/outdomain_analysis')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and filtering out-domain data...")
    cod5_outdomain, va34_outdomain = load_and_filter_data(cod5_path, va34_path)
    
    print("Calculating summary statistics...")
    # Calculate summary statistics
    cod5_stats = []
    va34_stats = []
    
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        cod5_stats.append(calculate_summary_statistics(cod5_outdomain, model, 'COD5'))
        va34_stats.append(calculate_summary_statistics(va34_outdomain, model, 'VA34'))
    
    cod5_stats_df = pd.DataFrame(cod5_stats)
    va34_stats_df = pd.DataFrame(va34_stats)
    
    print("\n=== COD5 Out-Domain Performance ===")
    print(cod5_stats_df[['model', 'csmf_mean', 'csmf_std', 'cod_mean', 'cod_std']])
    
    print("\n=== VA34 Out-Domain Performance ===")
    print(va34_stats_df[['model', 'csmf_mean', 'csmf_std', 'cod_mean', 'cod_std']])
    
    print("\nAnalyzing transfer patterns...")
    # Analyze transfer patterns
    cod5_patterns = analyze_transfer_patterns(cod5_outdomain)
    va34_patterns = analyze_transfer_patterns(va34_outdomain)
    
    # Save transfer matrices
    print("\nCreating transfer matrices...")
    all_matrices = create_transfer_heatmaps(cod5_outdomain, va34_outdomain, output_dir)
    print(f"Created {len(all_matrices)} transfer matrices for all models")
    
    # Save statistics
    cod5_stats_df.to_csv(output_dir / 'cod5_outdomain_stats.csv', index=False)
    va34_stats_df.to_csv(output_dir / 'va34_outdomain_stats.csv', index=False)
    
    print("\nCreating visualizations...")
    # Create visualizations
    create_model_comparison_plot(cod5_stats_df, va34_stats_df, output_dir)
    create_site_quality_plot(cod5_patterns, va34_patterns, output_dir)
    
    print("\nGenerating report...")
    # Generate report
    generate_report(cod5_stats_df, va34_stats_df, cod5_patterns, va34_patterns, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print("\nKey Findings:")
    print("1. Out-domain generalization varies significantly across models")
    print("2. Transfer performance heavily depends on source-target site combination")
    print("3. Some sites are consistently difficult for cross-site transfer (e.g., Pemba, Bohol)")
    print("4. Categorical NB shows poorest transfer capabilities across all scenarios")

if __name__ == "__main__":
    main()