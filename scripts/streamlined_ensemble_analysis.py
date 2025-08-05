#!/usr/bin/env python
"""
Streamlined Ensemble vs Individual Baseline Model Analysis

This script provides the essential analysis comparing ensemble models against 
individual baseline models with proper matching.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def load_and_filter_data():
    """Load and filter data for fair comparison."""
    print("Loading and filtering data...")
    
    # Load data
    ensemble_df = pd.read_csv("results/ensemble_with_names-v2/va34_comparison_results.csv")
    baseline_df = pd.read_csv("results/full_comparison_20250729_155434/va34_comparison_results.csv")
    
    # Extract baseline model names
    baseline_df['model_name'] = baseline_df['experiment_id'].str.split('_').str[0]
    
    # Filter baseline to training_fraction=1.0 only
    baseline_df = baseline_df[baseline_df['training_fraction'] == 1.0].copy()
    
    # Extract ensemble size
    ensemble_df['ensemble_size'] = ensemble_df['experiment_id'].str.extract(r'size(\d+)_').astype(int)
    
    # Find common site combinations
    ensemble_combinations = set(zip(ensemble_df['train_site'], ensemble_df['test_site']))
    baseline_combinations = set(zip(baseline_df['train_site'], baseline_df['test_site']))
    common_combinations = ensemble_combinations & baseline_combinations
    
    # Filter to common combinations only
    ensemble_mask = ensemble_df.apply(lambda x: (x['train_site'], x['test_site']) in common_combinations, axis=1)
    baseline_mask = baseline_df.apply(lambda x: (x['train_site'], x['test_site']) in common_combinations, axis=1)
    
    ensemble_df = ensemble_df[ensemble_mask].copy()
    baseline_df = baseline_df[baseline_mask].copy()
    
    print(f"Final datasets: {len(ensemble_df)} ensemble, {len(baseline_df)} baseline experiments")
    print(f"Common site combinations: {len(common_combinations)}")
    
    return ensemble_df, baseline_df, common_combinations

def calculate_performance_rankings(ensemble_df, baseline_df):
    """Calculate performance rankings for all models."""
    print("Calculating performance rankings...")
    
    rankings = {}
    
    # Individual baseline models
    for model in baseline_df['model_name'].unique():
        model_data = baseline_df[baseline_df['model_name'] == model]
        rankings[model] = {
            'type': 'individual',
            'csmf_mean': model_data['csmf_accuracy'].mean(),
            'csmf_std': model_data['csmf_accuracy'].std(),
            'cod_mean': model_data['cod_accuracy'].mean(),
            'cod_std': model_data['cod_accuracy'].std(),
            'experiments': len(model_data)
        }
    
    # Ensemble models by size
    for ensemble_size in [3, 5]:
        ensemble_data = ensemble_df[ensemble_df['ensemble_size'] == ensemble_size]
        if len(ensemble_data) > 0:
            model_name = f'{ensemble_size}-model_ensemble'
            rankings[model_name] = {
                'type': 'ensemble',
                'csmf_mean': ensemble_data['csmf_accuracy'].mean(),
                'csmf_std': ensemble_data['csmf_accuracy'].std(),
                'cod_mean': ensemble_data['cod_accuracy'].mean(),
                'cod_std': ensemble_data['cod_accuracy'].std(),
                'experiments': len(ensemble_data)
            }
    
    # Convert to DataFrame and sort by CSMF accuracy
    rankings_df = pd.DataFrame(rankings).T
    rankings_df = rankings_df.sort_values('csmf_mean', ascending=False)
    
    return rankings_df

def analyze_head_to_head_comparisons(ensemble_df, baseline_df):
    """Analyze head-to-head comparisons between ensembles and individual models."""
    print("Analyzing head-to-head comparisons...")
    
    results = {}
    
    # Create match keys
    ensemble_df['match_key'] = ensemble_df['train_site'] + '_' + ensemble_df['test_site']
    baseline_df['match_key'] = baseline_df['train_site'] + '_' + baseline_df['test_site']
    
    for ensemble_size in [3, 5]:
        ensemble_subset = ensemble_df[ensemble_df['ensemble_size'] == ensemble_size]
        if len(ensemble_subset) == 0:
            continue
            
        results[f'{ensemble_size}_model'] = {}
        
        for model in baseline_df['model_name'].unique():
            baseline_subset = baseline_df[baseline_df['model_name'] == model]
            
            # Calculate average performance by site combination
            ensemble_by_site = ensemble_subset.groupby('match_key')['csmf_accuracy'].mean()
            baseline_by_site = baseline_subset.groupby('match_key')['csmf_accuracy'].mean()
            
            # Find common sites
            common_sites = set(ensemble_by_site.index) & set(baseline_by_site.index)
            
            if len(common_sites) > 0:
                ensemble_values = [ensemble_by_site[site] for site in common_sites]
                baseline_values = [baseline_by_site[site] for site in common_sites]
                differences = np.array(ensemble_values) - np.array(baseline_values)
                
                # Statistics
                wins = (differences > 0).sum()
                losses = (differences < 0).sum()
                ties = (differences == 0).sum()
                
                results[f'{ensemble_size}_model'][model] = {
                    'sites_compared': len(common_sites),
                    'ensemble_mean': np.mean(ensemble_values),
                    'baseline_mean': np.mean(baseline_values),
                    'mean_difference': np.mean(differences),
                    'wins': wins,
                    'losses': losses,
                    'ties': ties,
                    'win_rate': wins / len(common_sites) if len(common_sites) > 0 else 0
                }
    
    return results

def generate_summary_report(rankings_df, head_to_head_results):
    """Generate a comprehensive summary report."""
    
    output_dir = Path('results/final_corrected_ensemble_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'corrected_analysis_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Final Corrected Ensemble vs Individual Baseline Model Analysis\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This analysis compares ensemble VA models against individual baseline models ")
        f.write("using properly matched train/test site combinations and fair comparison methodology.\n\n")
        
        # Performance Rankings
        f.write("## Overall Performance Rankings (by CSMF Accuracy)\n\n")
        f.write("| Rank | Model | Type | CSMF Accuracy | COD Accuracy | Experiments |\n")
        f.write("|------|-------|------|---------------|--------------|-------------|\n")
        
        for i, (model, row) in enumerate(rankings_df.iterrows(), 1):
            f.write(f"| {i} | {model} | {row['type']} | ")
            f.write(f"{row['csmf_mean']:.4f} ± {row['csmf_std']:.4f} | ")
            f.write(f"{row['cod_mean']:.4f} ± {row['cod_std']:.4f} | ")
            f.write(f"{int(row['experiments'])} |\n")
        
        f.write("\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        
        # Best individual model
        individual_rankings = rankings_df[rankings_df['type'] == 'individual']
        best_individual = individual_rankings.iloc[0]
        f.write(f"### 1. Best Individual Model: {best_individual.name}\n")
        f.write(f"- CSMF Accuracy: {best_individual['csmf_mean']:.4f} ± {best_individual['csmf_std']:.4f}\n")
        f.write(f"- COD Accuracy: {best_individual['cod_mean']:.4f} ± {best_individual['cod_std']:.4f}\n\n")
        
        # Best ensemble
        ensemble_rankings = rankings_df[rankings_df['type'] == 'ensemble']
        if len(ensemble_rankings) > 0:
            best_ensemble = ensemble_rankings.iloc[0]
            f.write(f"### 2. Best Ensemble: {best_ensemble.name}\n")
            f.write(f"- CSMF Accuracy: {best_ensemble['csmf_mean']:.4f} ± {best_ensemble['csmf_std']:.4f}\n")
            f.write(f"- COD Accuracy: {best_ensemble['cod_mean']:.4f} ± {best_ensemble['cod_std']:.4f}\n\n")
            
            # Compare best ensemble vs best individual
            improvement = best_ensemble['csmf_mean'] - best_individual['csmf_mean']
            f.write(f"### 3. Best Ensemble vs Best Individual\n")
            f.write(f"- CSMF Improvement: {improvement:.4f} ({improvement/best_individual['csmf_mean']*100:.1f}%)\n\n")
        
        # XGBoost analysis
        if 'xgboost' in individual_rankings.index:
            xgb_rank = individual_rankings.index.get_loc('xgboost') + 1
            f.write(f"### 4. XGBoost Performance\n")
            f.write(f"- Rank among individual models: {xgb_rank}\n")
            f.write(f"- Is best individual model: {'Yes' if xgb_rank == 1 else 'No'}\n\n")
        
        # Head-to-head comparisons
        f.write("## Head-to-Head Comparisons\n\n")
        
        for ensemble_size, comparisons in head_to_head_results.items():
            f.write(f"### {ensemble_size.replace('_', ' ').title()} vs Individual Models\n\n")
            f.write("| Individual Model | Sites Compared | Win Rate | Mean Difference (CSMF) |\n")
            f.write("|------------------|----------------|----------|----------------------|\n")
            
            for model, stats in comparisons.items():
                f.write(f"| {model} | {stats['sites_compared']} | ")
                f.write(f"{stats['win_rate']:.3f} | {stats['mean_difference']:.4f} |\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        overall_best = rankings_df.iloc[0]
        f.write(f"1. **Best Overall Model**: {overall_best.name} ")
        f.write(f"(CSMF: {overall_best['csmf_mean']:.4f})\n\n")
        
        f.write(f"2. **Best Individual Model**: {best_individual.name} ")
        f.write("for scenarios requiring single models\n\n")
        
        if len(ensemble_rankings) > 0:
            f.write(f"3. **Best Ensemble**: {best_ensemble.name} ")
            f.write("when computational resources allow\n\n")
        
        f.write("4. **Usage Guidelines**:\n")
        f.write("   - Use individual models for faster inference and simpler deployment\n")
        f.write("   - Use ensembles when maximum accuracy is required and computational cost is acceptable\n")
        f.write("   - Consider the specific train/test site combination when choosing models\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("- **Fair Comparison**: Used only training_fraction=1.0 for baseline models\n")
        f.write("- **Site Matching**: Compared only on common train/test site combinations\n")
        f.write("- **Aggregation**: Averaged performance across site combinations for head-to-head comparisons\n")
        f.write("- **Primary Metric**: CSMF accuracy (main VA evaluation metric)\n")
        f.write("- **Secondary Metric**: COD accuracy\n")
    
    print(f"Summary report saved to: {report_path}")
    return report_path

def identify_files_to_delete():
    """Identify legacy/incorrect analysis files that should be deleted."""
    
    files_to_delete = [
        # Flawed analysis files
        "analyze_ensemble_vs_baseline_flawed.py",
        "ensemble_comparison_report_flawed.md",
        
        # Legacy analysis files with incorrect methodology
        "comprehensive_ensemble_analysis.py",
        "corrected_ensemble_analysis.py", 
        "ensemble_vs_baseline_corrected_analysis.py",
        
        # Old analysis reports with wrong conclusions
        "ensemble_comparison_report.md",
        "corrected_ensemble_comparison_report.md",
        "ensemble_analysis_diagnosis.md",
        
        # Summary files that may be outdated
        "comparison_summary.py",
        "ANALYSIS_SUMMARY.md",
        "FINAL_ENSEMBLE_ANALYSIS_SUMMARY.md",
        "METHODOLOGY_CORRECTIONS.md",
        
        # Data structure analysis (temporary)
        "data_structure_analysis.py"
    ]
    
    # Check which files actually exist
    existing_files = []
    for file_path in files_to_delete:
        if Path(file_path).exists():
            existing_files.append(file_path)
    
    report_path = Path('results/final_corrected_ensemble_analysis/files_to_delete.md')
    
    with open(report_path, 'w') as f:
        f.write("# Files to Delete - Legacy/Incorrect Analysis\n\n")
        f.write("The following files contain incorrect analysis or outdated methodology ")
        f.write("and should be deleted to avoid confusion:\n\n")
        
        f.write("## Files with Flawed Methodology\n\n")
        for file_path in existing_files:
            if 'flawed' in file_path or 'corrected' in file_path:
                f.write(f"- `{file_path}` - Contains incorrect analysis methodology\n")
        
        f.write("\n## Legacy Analysis Files\n\n")
        for file_path in existing_files:
            if 'comprehensive' in file_path or 'ensemble_vs_baseline' in file_path:
                f.write(f"- `{file_path}` - Superseded by final corrected analysis\n")
        
        f.write("\n## Outdated Reports\n\n")
        for file_path in existing_files:
            if file_path.endswith('.md') and 'final' not in file_path.lower():
                f.write(f"- `{file_path}` - Contains outdated conclusions\n")
        
        f.write("\n## Summary\n\n")
        f.write(f"**Total files to delete**: {len(existing_files)}\n\n")
        f.write("**Reason for deletion**: These files contain analysis based on:\n")
        f.write("- Incorrect site matching (comparing different site combinations)\n")
        f.write("- Unfair comparisons (different training fractions)\n")
        f.write("- Methodological errors in ensemble vs baseline matching\n")
        f.write("- Outdated conclusions that have been corrected\n\n")
        f.write("**Files to keep**:\n")
        f.write("- `final_corrected_ensemble_analysis.py` - Definitive analysis script\n")
        f.write("- `streamlined_ensemble_analysis.py` - Streamlined version\n")
        f.write("- Results in `results/final_corrected_ensemble_analysis/` - Corrected outputs\n")
    
    print(f"Files to delete list saved to: {report_path}")
    return existing_files

def main():
    """Main analysis execution."""
    print("=== STREAMLINED ENSEMBLE VS BASELINE ANALYSIS ===\n")
    
    # Load and filter data
    ensemble_df, baseline_df, common_combinations = load_and_filter_data()
    
    # Calculate performance rankings
    rankings_df = calculate_performance_rankings(ensemble_df, baseline_df)
    
    print("\nPerformance Rankings:")
    for i, (model, row) in enumerate(rankings_df.iterrows(), 1):
        print(f"{i:2d}. {model:20s} - CSMF: {row['csmf_mean']:.4f} ± {row['csmf_std']:.4f}")
    
    # Head-to-head comparisons
    head_to_head_results = analyze_head_to_head_comparisons(ensemble_df, baseline_df)
    
    print("\nHead-to-Head Win Rates (Ensembles vs Individual Models):")
    for ensemble_type, comparisons in head_to_head_results.items():
        print(f"\n{ensemble_type.replace('_', ' ').title()}:")
        for model, stats in comparisons.items():
            print(f"  vs {model:15s}: {stats['win_rate']:.3f} ({stats['wins']}/{stats['sites_compared']} sites)")
    
    # Generate summary report
    report_path = generate_summary_report(rankings_df, head_to_head_results)
    
    # Identify files to delete
    files_to_delete = identify_files_to_delete()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Summary report: {report_path}")
    print(f"Files to delete: {len(files_to_delete)}")
    
    return {
        'rankings': rankings_df,
        'head_to_head': head_to_head_results,
        'report_path': report_path,
        'files_to_delete': files_to_delete
    }

if __name__ == "__main__":
    results = main()