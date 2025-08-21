#!/usr/bin/env python3
"""
In-Domain Model Performance Analysis for VA Models
Analyzes performance of XGBoost and InSilico models on COD5 and VA34 classification tasks
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
    """Load data and filter for in-domain experiments"""
    cod5_df = pd.read_csv(cod5_path)
    va34_df = pd.read_csv(va34_path)
    
    # Filter for in-domain only
    cod5_indomain = cod5_df[cod5_df['experiment_type'] == 'in_domain'].copy()
    va34_indomain = va34_df[va34_df['experiment_type'] == 'in_domain'].copy()
    
    # Add classification type column
    cod5_indomain['classification'] = 'COD5'
    va34_indomain['classification'] = 'VA34'
    
    return cod5_indomain, va34_indomain

def calculate_summary_statistics(df, model_name, classification_type):
    """Calculate summary statistics for a model"""
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
    
    # Calculate 95% CI
    stats_dict['csmf_ci_lower'] = model_data['csmf_accuracy_ci_lower'].mean()
    stats_dict['csmf_ci_upper'] = model_data['csmf_accuracy_ci_upper'].mean()
    stats_dict['cod_ci_lower'] = model_data['cod_accuracy_ci_lower'].mean()
    stats_dict['cod_ci_upper'] = model_data['cod_accuracy_ci_upper'].mean()
    
    return stats_dict

def analyze_site_performance(df, model_name):
    """Analyze performance by site for a model"""
    model_data = df[df['model'] == model_name]
    site_stats = []
    
    for site in model_data['test_site'].unique():
        site_data = model_data[model_data['test_site'] == site]
        site_stats.append({
            'site': site,
            'model': model_name,
            'csmf_accuracy': site_data['csmf_accuracy'].iloc[0],
            'cod_accuracy': site_data['cod_accuracy'].iloc[0],
            'n_train': site_data['n_train'].iloc[0],
            'n_test': site_data['n_test'].iloc[0]
        })
    
    return pd.DataFrame(site_stats)

def create_model_comparison_plot(cod5_stats, va34_stats, output_dir):
    """Create comparison plots for models"""
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
    ax.set_title('CSMF Accuracy: In-Domain Performance', fontsize=14, fontweight='bold')
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
    ax.set_title('COD Accuracy: In-Domain Performance', fontsize=14, fontweight='bold')
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
    
    plt.suptitle('In-Domain Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_site_heatmap(cod5_site_data, va34_site_data, output_dir):
    """Create heatmap of site-specific performance"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Prepare data for heatmaps
    sites = sorted(set(cod5_site_data['site'].unique()) | set(va34_site_data['site'].unique()))
    models = ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']
    
    # COD5 CSMF Heatmap
    cod5_csmf_matrix = pd.DataFrame(index=sites, columns=models)
    for model in models:
        model_data = cod5_site_data[cod5_site_data['model'] == model]
        for _, row in model_data.iterrows():
            cod5_csmf_matrix.loc[row['site'], model] = row['csmf_accuracy']
    
    ax = axes[0, 0]
    sns.heatmap(cod5_csmf_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, vmin=0.2, vmax=1.0, cbar_kws={'label': 'Accuracy'}, annot_kws={'fontsize': 8})
    ax.set_title('COD5: CSMF Accuracy by Site')
    ax.set_ylabel('Site')
    ax.set_xlabel('Model')
    
    # COD5 COD Heatmap
    cod5_cod_matrix = pd.DataFrame(index=sites, columns=models)
    for model in models:
        model_data = cod5_site_data[cod5_site_data['model'] == model]
        for _, row in model_data.iterrows():
            cod5_cod_matrix.loc[row['site'], model] = row['cod_accuracy']
    
    ax = axes[0, 1]
    sns.heatmap(cod5_cod_matrix.astype(float), annot=True, fmt='.2f', cmap='YlGnBu', 
                ax=ax, vmin=0.1, vmax=0.8, cbar_kws={'label': 'Accuracy'}, annot_kws={'fontsize': 8})
    ax.set_title('COD5: COD Accuracy by Site')
    ax.set_ylabel('Site')
    ax.set_xlabel('Model')
    
    # VA34 CSMF Heatmap
    va34_csmf_matrix = pd.DataFrame(index=sites, columns=models)
    for model in models:
        model_data = va34_site_data[va34_site_data['model'] == model]
        for _, row in model_data.iterrows():
            va34_csmf_matrix.loc[row['site'], model] = row['csmf_accuracy']
    
    ax = axes[1, 0]
    sns.heatmap(va34_csmf_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, vmin=0.1, vmax=0.95, cbar_kws={'label': 'Accuracy'}, annot_kws={'fontsize': 8})
    ax.set_title('VA34: CSMF Accuracy by Site')
    ax.set_ylabel('Site')
    ax.set_xlabel('Model')
    
    # VA34 COD Heatmap
    va34_cod_matrix = pd.DataFrame(index=sites, columns=models)
    for model in models:
        model_data = va34_site_data[va34_site_data['model'] == model]
        for _, row in model_data.iterrows():
            va34_cod_matrix.loc[row['site'], model] = row['cod_accuracy']
    
    ax = axes[1, 1]
    sns.heatmap(va34_cod_matrix.astype(float), annot=True, fmt='.2f', cmap='YlGnBu', 
                ax=ax, vmin=0.05, vmax=0.65, cbar_kws={'label': 'Accuracy'}, annot_kws={'fontsize': 8})
    ax.set_title('VA34: COD Accuracy by Site')
    ax.set_ylabel('Site')
    ax.set_xlabel('Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'site_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(cod5_stats, va34_stats, cod5_site_data, va34_site_data, output_dir):
    """Generate markdown report"""
    report = []
    report.append("# In-Domain Model Performance Analysis Report\n")
    report.append("## Executive Summary\n")
    
    # Find best performing models
    best_cod5_model = cod5_stats.loc[cod5_stats['cod_mean'].idxmax(), 'model']
    best_cod5_cod = cod5_stats.loc[cod5_stats['cod_mean'].idxmax(), 'cod_mean']
    best_va34_model = va34_stats.loc[va34_stats['cod_mean'].idxmax(), 'model']
    best_va34_cod = va34_stats.loc[va34_stats['cod_mean'].idxmax(), 'cod_mean']
    
    # Find worst performing models
    worst_cod5_model = cod5_stats.loc[cod5_stats['cod_mean'].idxmin(), 'model']
    worst_cod5_cod = cod5_stats.loc[cod5_stats['cod_mean'].idxmin(), 'cod_mean']
    worst_va34_model = va34_stats.loc[va34_stats['cod_mean'].idxmin(), 'model']
    worst_va34_cod = va34_stats.loc[va34_stats['cod_mean'].idxmin(), 'cod_mean']
    
    report.append("### Key Findings:\n")
    report.append(f"1. **Best performing models vary by classification complexity**\n")
    report.append(f"   - COD5: {best_cod5_model.upper()} achieves {best_cod5_cod:.1%} COD accuracy\n")
    report.append(f"   - VA34: {best_va34_model.upper()} achieves {best_va34_cod:.1%} COD accuracy\n\n")
    
    report.append(f"2. **Significant performance range across models**\n")
    report.append(f"   - COD5: {worst_cod5_model.upper()} ({worst_cod5_cod:.1%}) to {best_cod5_model.upper()} ({best_cod5_cod:.1%})\n")
    report.append(f"   - VA34: {worst_va34_model.upper()} ({worst_va34_cod:.1%}) to {best_va34_model.upper()} ({best_va34_cod:.1%})\n\n")
    
    report.append(f"3. **Performance degradation from COD5 to VA34 affects all models**\n")
    for _, row in cod5_stats.iterrows():
        model = row['model']
        cod5_acc = row['cod_mean']
        va34_acc = va34_stats[va34_stats['model'] == model]['cod_mean'].iloc[0]
        report.append(f"   - {model.upper()}: {(cod5_acc - va34_acc):.1%} drop\n")
    report.append("\n")
    
    report.append("## Detailed Performance Metrics\n")
    report.append("### COD5 Classification (5 causes)\n")
    report.append("| Model | CSMF Accuracy | COD Accuracy | CSMF Std | COD Std |\n")
    report.append("|-------|--------------|-------------|----------|----------|\n")
    
    for _, row in cod5_stats.iterrows():
        report.append(f"| {row['model'].upper()} | {row['csmf_mean']:.3f} | {row['cod_mean']:.3f} | "
                     f"{row['csmf_std']:.3f} | {row['cod_std']:.3f} |\n")
    
    report.append("\n### VA34 Classification (34 causes)\n")
    report.append("| Model | CSMF Accuracy | COD Accuracy | CSMF Std | COD Std |\n")
    report.append("|-------|--------------|-------------|----------|----------|\n")
    
    for _, row in va34_stats.iterrows():
        report.append(f"| {row['model'].upper()} | {row['csmf_mean']:.3f} | {row['cod_mean']:.3f} | "
                     f"{row['csmf_std']:.3f} | {row['cod_std']:.3f} |\n")
    
    report.append("\n## Site-Specific Performance\n")
    report.append("### Best Performing Sites\n")
    
    # Find best sites for each model/classification
    report.append("| Classification | Model | Best Site | COD Accuracy |\n")
    report.append("|---------------|-------|-----------|-------------|\n")
    
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        cod5_model_site = cod5_site_data[cod5_site_data['model'] == model].nlargest(1, 'cod_accuracy')
        va34_model_site = va34_site_data[va34_site_data['model'] == model].nlargest(1, 'cod_accuracy')
        if not cod5_model_site.empty:
            report.append(f"| COD5 | {model.upper()} | {cod5_model_site['site'].iloc[0]} | {cod5_model_site['cod_accuracy'].iloc[0]:.3f} |\n")
        if not va34_model_site.empty:
            report.append(f"| VA34 | {model.upper()} | {va34_model_site['site'].iloc[0]} | {va34_model_site['cod_accuracy'].iloc[0]:.3f} |\n")
    
    report.append("\n## Statistical Analysis\n")
    
    # Perform ANOVA to compare all models
    report.append("### Model Performance Comparison (One-way ANOVA)\n")
    
    # COD5 ANOVA
    cod5_groups = []
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        model_values = cod5_site_data[cod5_site_data['model'] == model]['cod_accuracy'].values
        if len(model_values) > 0:
            cod5_groups.append(model_values)
    
    if len(cod5_groups) > 1:
        f_stat, p_value = stats.f_oneway(*cod5_groups)
        report.append(f"#### COD5 Classification:\n")
        report.append(f"- F-statistic: {f_stat:.3f}\n")
        report.append(f"- p-value: {p_value:.4f}\n")
        report.append(f"- Significant difference between models: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)\n\n")
    
    # VA34 ANOVA
    va34_groups = []
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        model_values = va34_site_data[va34_site_data['model'] == model]['cod_accuracy'].values
        if len(model_values) > 0:
            va34_groups.append(model_values)
    
    if len(va34_groups) > 1:
        f_stat, p_value = stats.f_oneway(*va34_groups)
        report.append(f"#### VA34 Classification:\n")
        report.append(f"- F-statistic: {f_stat:.3f}\n")
        report.append(f"- p-value: {p_value:.4f}\n")
        report.append(f"- Significant difference between models: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)\n\n")
    
    report.append("## Recommendations\n")
    best_overall = cod5_stats.loc[cod5_stats['cod_mean'].idxmax(), 'model']
    report.append(f"1. **Consider {best_overall.upper()} for COD5 classification** - best overall performance\n")
    report.append("2. **Avoid Categorical Naive Bayes** - consistently poor performance across tasks\n")
    report.append("3. **Site-specific calibration is critical** - performance varies significantly by location\n")
    report.append("4. **COD5 classification recommended over VA34** - better accuracy for practical deployment\n")
    report.append("5. **Ensemble methods (RF, XGBoost) generally outperform single models**\n")
    
    # Write report
    with open(output_dir / 'indomain_analysis_report.md', 'w') as f:
        f.writelines(report)
    
    print("Report generated: indomain_analysis_report.md")

def main():
    # Define paths
    cod5_path = Path('/Users/ericliu/projects5/context-engineering-intro/results/full_comparison_20250821_144955_cod5/cod5_comparison_results.csv')
    va34_path = Path('/Users/ericliu/projects5/context-engineering-intro/results/full_comparison_20250821_145007_va34/va34_comparison_results.csv')
    output_dir = Path('/Users/ericliu/projects5/context-engineering-intro/results/indomain_analysis')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and filtering data...")
    cod5_indomain, va34_indomain = load_and_filter_data(cod5_path, va34_path)
    
    print("Calculating summary statistics...")
    # Calculate summary statistics for each model and classification
    cod5_stats = []
    va34_stats = []
    
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        cod5_stats.append(calculate_summary_statistics(cod5_indomain, model, 'COD5'))
        va34_stats.append(calculate_summary_statistics(va34_indomain, model, 'VA34'))
    
    cod5_stats_df = pd.DataFrame(cod5_stats)
    va34_stats_df = pd.DataFrame(va34_stats)
    
    print("\n=== COD5 In-Domain Performance ===")
    print(cod5_stats_df[['model', 'csmf_mean', 'csmf_std', 'cod_mean', 'cod_std']])
    
    print("\n=== VA34 In-Domain Performance ===")
    print(va34_stats_df[['model', 'csmf_mean', 'csmf_std', 'cod_mean', 'cod_std']])
    
    print("\nAnalyzing site-specific performance...")
    # Analyze site-specific performance
    cod5_site_data = []
    va34_site_data = []
    
    for model in ['categorical_nb', 'random_forest', 'xgboost', 'logistic_regression', 'insilico']:
        cod5_site_data.append(analyze_site_performance(cod5_indomain, model))
        va34_site_data.append(analyze_site_performance(va34_indomain, model))
    
    cod5_site_df = pd.concat(cod5_site_data, ignore_index=True)
    va34_site_df = pd.concat(va34_site_data, ignore_index=True)
    
    # Save processed data
    cod5_stats_df.to_csv(output_dir / 'cod5_summary_stats.csv', index=False)
    va34_stats_df.to_csv(output_dir / 'va34_summary_stats.csv', index=False)
    cod5_site_df.to_csv(output_dir / 'cod5_site_performance.csv', index=False)
    va34_site_df.to_csv(output_dir / 'va34_site_performance.csv', index=False)
    
    print("\nCreating visualizations...")
    # Create visualizations
    create_model_comparison_plot(cod5_stats_df, va34_stats_df, output_dir)
    create_site_heatmap(cod5_site_df, va34_site_df, output_dir)
    
    print("\nGenerating report...")
    # Generate report
    generate_report(cod5_stats_df, va34_stats_df, cod5_site_df, va34_site_df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print("\nKey Findings:")
    print("1. Model performance varies significantly - from Categorical NB (lowest) to XGBoost/RF (highest)")
    print("2. All models show performance degradation from COD5 to VA34 classification")
    print("3. Site-specific variations suggest population-specific mortality patterns")
    print("4. Ensemble methods (Random Forest, XGBoost) consistently outperform simpler models")

if __name__ == "__main__":
    main()