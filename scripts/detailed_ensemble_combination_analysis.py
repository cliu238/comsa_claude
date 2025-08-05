#!/usr/bin/env python
"""
DETAILED ENSEMBLE COMBINATION ANALYSIS

This script provides detailed analysis of specific 3-model ensemble combinations
rather than treating all 3-model ensembles as a single category.

Key Features:
1. Separates different 3-model combinations (e.g., XGB+RF+CNB vs XGB+CNB+InSilico)
2. Analyzes performance of each specific combination
3. Compares each combination against individual baseline models
4. Identifies synergies and patterns in model combinations
5. Provides actionable insights on which combinations work best

Usage: python scripts/detailed_ensemble_combination_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

class DetailedEnsembleCombinationAnalyzer:
    """Detailed analyzer for specific ensemble model combinations."""
    
    def __init__(self):
        self.ensemble_path = "results/ensemble_with_names-v2/va34_comparison_results.csv"
        self.baseline_path = "results/full_comparison_20250729_155434/va34_comparison_results.csv"
        self.output_dir = Path('results/detailed_ensemble_combination_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.ensemble_df = None
        self.baseline_df = None
        self.combination_analysis = {}
        
        # Model name mapping for cleaner display
        self.model_name_map = {
            'xgb': 'XGBoost',
            'rf': 'Random Forest', 
            'cnb': 'Categorical NB',
            'ins': 'InSilicoVA',
            'lr': 'Logistic Regression'
        }
        
    def load_data(self):
        """Load and preprocess ensemble and baseline data."""
        print("Loading data...")
        
        # Load ensemble data
        self.ensemble_df = pd.read_csv(self.ensemble_path)
        print(f"Loaded {len(self.ensemble_df)} ensemble experiments")
        
        # Load baseline data - filter to training_fraction=1.0 for fair comparison
        baseline_full = pd.read_csv(self.baseline_path)
        self.baseline_df = baseline_full[baseline_full['training_fraction'] == 1.0].copy()
        print(f"Loaded {len(self.baseline_df)} baseline experiments (training_fraction=1.0)")
        
        # Extract model combinations from ensemble experiment IDs
        self.ensemble_df['model_combination'] = self.ensemble_df['experiment_id'].apply(
            self._extract_model_combination
        )
        
        # Filter to common sites for fair comparison
        common_sites = self._get_common_sites()
        print(f"Common train/test sites: {sorted(common_sites)}")
        
        # Filter both datasets to common sites
        self.ensemble_df = self._filter_to_common_sites(self.ensemble_df, common_sites)
        self.baseline_df = self._filter_to_common_sites(self.baseline_df, common_sites)
        
        print(f"After site filtering: {len(self.ensemble_df)} ensemble, {len(self.baseline_df)} baseline")
        
    def _extract_model_combination(self, experiment_id: str) -> str:
        """Extract model combination from experiment ID."""
        # Pattern: ensemble_soft_[weight]_size[N]_[models]_[domain]_[train]_[test]_size[frac]
        parts = experiment_id.split('_')
        
        # Find size3 or size5 and extract models after it
        size_idx = None
        for i, part in enumerate(parts):
            if part.startswith('size') and part[4:].isdigit():
                size_idx = i
                break
                
        if size_idx is None:
            return "unknown"
            
        # Extract models between size[N] and domain type
        models = []
        for i in range(size_idx + 1, len(parts)):
            if parts[i] in ['in', 'out']:  # domain indicators
                break
            models.append(parts[i])
            
        return '_'.join(models)
    
    def _get_common_sites(self) -> set:
        """Get sites common to both ensemble and baseline data."""
        ensemble_sites = set()
        baseline_sites = set()
        
        for _, row in self.ensemble_df.iterrows():
            ensemble_sites.add((row['train_site'], row['test_site']))
            
        for _, row in self.baseline_df.iterrows():
            baseline_sites.add((row['train_site'], row['test_site']))
            
        return ensemble_sites.intersection(baseline_sites)
    
    def _filter_to_common_sites(self, df: pd.DataFrame, common_sites: set) -> pd.DataFrame:
        """Filter dataframe to only include common site combinations."""
        mask = df.apply(lambda row: (row['train_site'], row['test_site']) in common_sites, axis=1)
        return df[mask].copy()
    
    def analyze_combinations(self):
        """Analyze each specific 3-model combination."""
        print("Analyzing specific model combinations...")
        
        # Get all unique 3-model combinations
        three_model_combos = self.ensemble_df[
            self.ensemble_df['experiment_id'].str.contains('size3')
        ]['model_combination'].unique()
        
        print(f"Found {len(three_model_combos)} unique 3-model combinations:")
        for combo in sorted(three_model_combos):
            print(f"  - {combo} ({self._format_combination_name(combo)})")
        
        # Analyze each combination
        for combo in three_model_combos:
            self._analyze_single_combination(combo)
            
        # Also analyze 5-model combinations
        five_model_combos = self.ensemble_df[
            self.ensemble_df['experiment_id'].str.contains('size5')
        ]['model_combination'].unique()
        
        print(f"\nFound {len(five_model_combos)} unique 5-model combinations:")
        for combo in sorted(five_model_combos):
            print(f"  - {combo} ({self._format_combination_name(combo)})")
            
        for combo in five_model_combos:
            self._analyze_single_combination(combo)
    
    def _format_combination_name(self, combo: str) -> str:
        """Convert abbreviated combination to readable name."""
        parts = combo.split('_')
        readable_parts = [self.model_name_map.get(part, part.upper()) for part in parts]
        return ' + '.join(readable_parts)
    
    def _analyze_single_combination(self, combination: str):
        """Analyze a single model combination against all individual baselines."""
        print(f"\nAnalyzing combination: {combination}")
        
        # Get data for this combination
        combo_data = self.ensemble_df[
            self.ensemble_df['model_combination'] == combination
        ].copy()
        
        if len(combo_data) == 0:
            print(f"  No data found for combination {combination}")
            return
            
        # Calculate combination performance statistics
        combo_stats = {
            'combination': combination,
            'formatted_name': self._format_combination_name(combination),
            'n_experiments': len(combo_data),
            'csmf_mean': combo_data['csmf_accuracy'].mean(),
            'csmf_std': combo_data['csmf_accuracy'].std(),
            'cod_mean': combo_data['cod_accuracy'].mean(),
            'cod_std': combo_data['cod_accuracy'].std(),
            'individual_comparisons': {}
        }
        
        # Compare against each individual baseline model
        individual_models = self.baseline_df['model'].unique()
        
        for ind_model in individual_models:
            comparison = self._compare_combination_vs_individual(combo_data, ind_model)
            combo_stats['individual_comparisons'][ind_model] = comparison
            
        self.combination_analysis[combination] = combo_stats
        
        print(f"  CSMF Accuracy: {combo_stats['csmf_mean']:.4f} ± {combo_stats['csmf_std']:.4f}")
        print(f"  COD Accuracy: {combo_stats['cod_mean']:.4f} ± {combo_stats['cod_std']:.4f}")
        print(f"  Experiments: {combo_stats['n_experiments']}")
    
    def _compare_combination_vs_individual(self, combo_data: pd.DataFrame, individual_model: str) -> Dict[str, Any]:
        """Compare a specific combination against an individual model."""
        # Get individual model data
        ind_data = self.baseline_df[self.baseline_df['model'] == individual_model].copy()
        
        if len(ind_data) == 0:
            return {'error': f'No data for individual model {individual_model}'}
        
        # Match by site combinations for head-to-head comparison
        matched_comparisons = []
        
        for _, combo_row in combo_data.iterrows():
            train_site = combo_row['train_site']
            test_site = combo_row['test_site']
            
            # Find matching individual experiment
            ind_match = ind_data[
                (ind_data['train_site'] == train_site) & 
                (ind_data['test_site'] == test_site)
            ]
            
            if len(ind_match) > 0:
                ind_row = ind_match.iloc[0]  # Take first match if multiple
                matched_comparisons.append({
                    'train_site': train_site,
                    'test_site': test_site,
                    'combo_csmf': combo_row['csmf_accuracy'],
                    'combo_cod': combo_row['cod_accuracy'],
                    'ind_csmf': ind_row['csmf_accuracy'],
                    'ind_cod': ind_row['cod_accuracy'],
                    'csmf_diff': combo_row['csmf_accuracy'] - ind_row['csmf_accuracy'],
                    'cod_diff': combo_row['cod_accuracy'] - ind_row['cod_accuracy']
                })
        
        if not matched_comparisons:
            return {'error': f'No matching site combinations for {individual_model}'}
        
        # Calculate comparison statistics
        csmf_diffs = [comp['csmf_diff'] for comp in matched_comparisons]
        cod_diffs = [comp['cod_diff'] for comp in matched_comparisons]
        
        # Win rate (combination better than individual)
        wins = sum(1 for diff in csmf_diffs if diff > 0)
        win_rate = wins / len(csmf_diffs) if csmf_diffs else 0
        
        # Statistical test
        if len(csmf_diffs) > 1:
            t_stat, p_value = stats.ttest_1samp(csmf_diffs, 0)
        else:
            t_stat, p_value = 0, 1
        
        return {
            'individual_model': individual_model,
            'n_comparisons': len(matched_comparisons),
            'win_rate': win_rate,
            'csmf_mean_diff': np.mean(csmf_diffs),
            'csmf_std_diff': np.std(csmf_diffs),
            'cod_mean_diff': np.mean(cod_diffs), 
            'cod_std_diff': np.std(cod_diffs),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'matched_comparisons': matched_comparisons
        }
    
    def generate_summary_tables(self):
        """Generate summary tables for all combinations."""
        print("\nGenerating summary tables...")
        
        # Overall combination performance table
        combo_performance = []
        for combo, stats in self.combination_analysis.items():
            combo_performance.append({
                'Combination': stats['formatted_name'], 
                'Short_Name': combo,
                'N_Experiments': stats['n_experiments'],
                'CSMF_Accuracy': f"{stats['csmf_mean']:.4f} ± {stats['csmf_std']:.4f}",
                'COD_Accuracy': f"{stats['cod_mean']:.4f} ± {stats['cod_std']:.4f}",
                'CSMF_Mean': stats['csmf_mean'],
                'COD_Mean': stats['cod_mean']
            })
        
        combo_df = pd.DataFrame(combo_performance)
        combo_df = combo_df.sort_values('CSMF_Mean', ascending=False)
        combo_df['Rank'] = range(1, len(combo_df) + 1)
        
        # Save combination performance table
        combo_output_path = self.output_dir / 'combination_performance_rankings.csv'
        combo_df.to_csv(combo_output_path, index=False)
        print(f"Saved combination performance rankings to {combo_output_path}")
        
        # Head-to-head comparison matrix
        self._generate_head_to_head_matrix()
        
        # Individual model beating rates
        self._generate_individual_beating_rates()
        
        return combo_df
    
    def _generate_head_to_head_matrix(self):
        """Generate matrix showing win rates of each combination vs each individual model."""
        print("Generating head-to-head comparison matrix...")
        
        individual_models = list(self.baseline_df['model'].unique())
        combinations = list(self.combination_analysis.keys())
        
        # Create win rate matrix
        win_rate_matrix = []
        mean_diff_matrix = []
        
        for combo in combinations:
            win_row = []
            diff_row = []
            for ind_model in individual_models:
                if ind_model in self.combination_analysis[combo]['individual_comparisons']:
                    comp_data = self.combination_analysis[combo]['individual_comparisons'][ind_model]
                    if 'error' not in comp_data:
                        win_row.append(comp_data['win_rate'])
                        diff_row.append(comp_data['csmf_mean_diff'])
                    else:
                        win_row.append(np.nan)
                        diff_row.append(np.nan)
                else:
                    win_row.append(np.nan)
                    diff_row.append(np.nan)
            win_rate_matrix.append(win_row)
            mean_diff_matrix.append(diff_row)
        
        # Convert to DataFrames
        win_df = pd.DataFrame(win_rate_matrix, 
                            index=[self._format_combination_name(c) for c in combinations],
                            columns=individual_models)
        
        diff_df = pd.DataFrame(mean_diff_matrix,
                             index=[self._format_combination_name(c) for c in combinations], 
                             columns=individual_models)
        
        # Save matrices
        win_df.to_csv(self.output_dir / 'combination_vs_individual_win_rates.csv')
        diff_df.to_csv(self.output_dir / 'combination_vs_individual_mean_differences.csv')
        
        print(f"Saved head-to-head matrices to {self.output_dir}")
        
    def _generate_individual_beating_rates(self):
        """Generate table showing how often each individual model beats combinations."""
        print("Generating individual model beating rates...")
        
        individual_models = list(self.baseline_df['model'].unique())
        beating_data = []
        
        for ind_model in individual_models:
            model_stats = {
                'Individual_Model': ind_model,
                'Total_Comparisons': 0,
                'Total_Wins': 0,
                'Overall_Win_Rate': 0,
                'Combinations_Beaten': [],
                'Average_Performance_Advantage': 0
            }
            
            total_comparisons = 0
            total_wins = 0
            total_diff = 0
            combinations_beaten = []
            
            for combo, combo_stats in self.combination_analysis.items():
                if ind_model in combo_stats['individual_comparisons']:
                    comp_data = combo_stats['individual_comparisons'][ind_model]
                    if 'error' not in comp_data and comp_data['n_comparisons'] > 0:
                        total_comparisons += comp_data['n_comparisons']
                        # Individual wins when combination has negative performance diff
                        ind_wins = comp_data['n_comparisons'] - (comp_data['win_rate'] * comp_data['n_comparisons'])
                        total_wins += ind_wins
                        total_diff += (-comp_data['csmf_mean_diff'] * comp_data['n_comparisons'])
                        
                        # Check if individual model beats this combination more often than not
                        if comp_data['win_rate'] < 0.5:
                            combinations_beaten.append(self._format_combination_name(combo))
            
            if total_comparisons > 0:
                model_stats['Total_Comparisons'] = total_comparisons
                model_stats['Total_Wins'] = total_wins
                model_stats['Overall_Win_Rate'] = total_wins / total_comparisons
                model_stats['Combinations_Beaten'] = combinations_beaten
                model_stats['Average_Performance_Advantage'] = total_diff / total_comparisons
            
            beating_data.append(model_stats)
        
        # Convert to DataFrame
        beating_df = pd.DataFrame(beating_data)
        beating_df = beating_df.sort_values('Overall_Win_Rate', ascending=False)
        
        # Save
        beating_df.to_csv(self.output_dir / 'individual_model_beating_rates.csv', index=False)
        print(f"Saved individual model beating rates to {self.output_dir}")
        
        return beating_df
    
    def generate_detailed_report(self):
        """Generate detailed markdown report with all findings."""
        print("Generating detailed report...")
        
        report_path = self.output_dir / 'detailed_combination_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Detailed Ensemble Combination Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This analysis separates different 3-model and 5-model ensemble combinations ")
            f.write("to provide detailed insights into which specific model combinations work best.\n\n")
            
            # Combination performance rankings
            f.write("## Model Combination Performance Rankings\n\n")
            combo_df = self.generate_summary_tables()
            
            f.write("| Rank | Combination | CSMF Accuracy | COD Accuracy | Experiments |\n")
            f.write("|------|-------------|---------------|--------------|-------------|\n")
            
            for _, row in combo_df.iterrows():
                f.write(f"| {row['Rank']} | {row['Combination']} | {row['CSMF_Accuracy']} | ")
                f.write(f"{row['COD_Accuracy']} | {row['N_Experiments']} |\n")
            
            # Detailed combination analysis
            f.write("\n## Detailed Combination Analysis\n\n")
            
            # Sort combinations by performance for reporting
            sorted_combos = sorted(self.combination_analysis.items(),
                                 key=lambda x: x[1]['csmf_mean'], reverse=True)
            
            for combo, stats in sorted_combos:
                f.write(f"### {stats['formatted_name']} ({combo})\n\n")
                f.write(f"**Performance:**\n")
                f.write(f"- CSMF Accuracy: {stats['csmf_mean']:.4f} ± {stats['csmf_std']:.4f}\n")
                f.write(f"- COD Accuracy: {stats['cod_mean']:.4f} ± {stats['cod_std']:.4f}\n")
                f.write(f"- Number of experiments: {stats['n_experiments']}\n\n")
                
                f.write("**Head-to-Head vs Individual Models:**\n\n")
                f.write("| Individual Model | Win Rate | Mean CSMF Difference | Significance |\n")
                f.write("|------------------|----------|---------------------|-------------|\n")
                
                for ind_model, comp_data in stats['individual_comparisons'].items():
                    if 'error' not in comp_data:
                        sig_marker = "✓" if comp_data['significant'] else ""
                        f.write(f"| {ind_model} | {comp_data['win_rate']:.1%} | ")
                        f.write(f"{comp_data['csmf_mean_diff']:+.4f} | {sig_marker} |\n")
                
                f.write("\n")
            
            # Key insights
            f.write("## Key Insights\n\n")
            
            # Best combinations
            best_combo = sorted_combos[0]
            f.write(f"### Best Performing Combination\n")
            f.write(f"**{best_combo[1]['formatted_name']}** achieves the highest CSMF accuracy ")
            f.write(f"at {best_combo[1]['csmf_mean']:.4f} ± {best_combo[1]['csmf_std']:.4f}\n\n")
            
            # Synergy analysis
            f.write("### Model Synergies\n")
            self._write_synergy_analysis(f)
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            self._write_recommendations(f)
        
        print(f"Detailed report saved to {report_path}")
        return report_path
    
    def _write_synergy_analysis(self, f):
        """Write synergy analysis to report file."""
        f.write("Analysis of which model combinations create synergistic effects:\n\n")
        
        # Group combinations by performance tiers
        sorted_combos = sorted(self.combination_analysis.items(),
                             key=lambda x: x[1]['csmf_mean'], reverse=True)
        
        top_tier = sorted_combos[:len(sorted_combos)//3] if len(sorted_combos) > 3 else sorted_combos[:1]
        
        f.write("**High-Performing Combinations:**\n")
        for combo, stats in top_tier:
            f.write(f"- {stats['formatted_name']}: {stats['csmf_mean']:.4f} CSMF accuracy\n")
        
        f.write("\n**Common Patterns in High-Performing Combinations:**\n")
        
        # Analyze model frequency in top combinations
        model_counts = defaultdict(int)
        for combo, stats in top_tier:
            models = combo.split('_')
            for model in models:
                model_counts[model] += 1
        
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            readable_name = self.model_name_map.get(model, model.upper())
            f.write(f"- {readable_name} appears in {count}/{len(top_tier)} top combinations\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f):
        """Write recommendations to report file."""
        sorted_combos = sorted(self.combination_analysis.items(),
                             key=lambda x: x[1]['csmf_mean'], reverse=True)
        
        best_combo = sorted_combos[0]
        
        f.write(f"1. **Best Overall Combination**: Use {best_combo[1]['formatted_name']} ")
        f.write(f"for maximum ensemble performance ({best_combo[1]['csmf_mean']:.4f} CSMF accuracy)\n\n")
        
        f.write("2. **Avoid Low-Performing Combinations**: ")
        if len(sorted_combos) > 1:
            worst_combo = sorted_combos[-1]
            f.write(f"Avoid {worst_combo[1]['formatted_name']} ")
            f.write(f"({worst_combo[1]['csmf_mean']:.4f} CSMF accuracy)\n\n")
        
        f.write("3. **Individual vs Ensemble Trade-offs**: ")
        f.write("Consider computational cost vs performance improvement when choosing between ")
        f.write("individual models and ensemble combinations\n\n")
        
        f.write("4. **Site-Specific Performance**: ")
        f.write("Performance varies by train/test site combination - consider site-specific ")
        f.write("model selection for optimal results\n\n")

    def run_analysis(self):
        """Run the complete detailed combination analysis."""
        print("Starting detailed ensemble combination analysis...")
        
        # Load and preprocess data
        self.load_data()
        
        # Analyze each combination
        self.analyze_combinations()
        
        # Generate summary tables
        self.generate_summary_tables()
        
        # Generate detailed report
        report_path = self.generate_detailed_report()
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Detailed report: {report_path}")
        
        return self.combination_analysis

def main():
    """Main execution function."""
    analyzer = DetailedEnsembleCombinationAnalyzer()
    results = analyzer.run_analysis()
    return results

if __name__ == "__main__":
    main()