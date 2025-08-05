#!/usr/bin/env python
"""
FINAL CORRECTED ENSEMBLE VS INDIVIDUAL BASELINE MODEL ANALYSIS

This script provides the definitive analysis comparing ensemble models against 
INDIVIDUAL baseline models with proper matching of train/test site combinations.

Key Requirements Addressed:
1. Compare ensembles with INDIVIDUAL baseline models (not just "best baseline")
2. Exact train/test site combination matching
3. Separate analysis for 3-model vs 5-model ensembles
4. Head-to-head comparisons for each model type
5. Clear performance rankings and recommendations

Critical Fixes:
- Only use matching site combinations (9 combinations: AP, Mexico, UP)
- Compare each ensemble against each individual baseline model separately
- Filter baseline data to only training_fraction=1.0 for fair comparison
- Provide statistical significance testing for each comparison
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
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

class FinalCorrectedEnsembleAnalyzer:
    """Final corrected analyzer for ensemble vs individual baseline model comparison."""
    
    def __init__(self):
        self.ensemble_path = "results/ensemble_with_names-v2/va34_comparison_results.csv"
        self.baseline_path = "results/full_comparison_20250729_155434/va34_comparison_results.csv"
        self.output_dir = Path('results/final_corrected_ensemble_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.ensemble_df = None
        self.baseline_df = None
        self.matched_data = None
        self.individual_comparisons = {}
        self.performance_rankings = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with proper filtering for fair comparison."""
        print("=== LOADING AND PREPARING DATA ===")
        
        # Load raw data
        self.ensemble_df = pd.read_csv(self.ensemble_path)
        self.baseline_df = pd.read_csv(self.baseline_path)
        
        # Extract baseline model names
        self.baseline_df['model_name'] = self.baseline_df['experiment_id'].str.split('_').str[0]
        
        # CRITICAL: Filter baseline to only training_fraction=1.0 for fair comparison
        self.baseline_df = self.baseline_df[self.baseline_df['training_fraction'] == 1.0].copy()
        
        # Parse ensemble information
        self.ensemble_df = self._parse_ensemble_info(self.ensemble_df)
        
        # Filter to common train/test site combinations only
        common_combinations = self._get_common_combinations()
        
        # Filter both datasets to only common combinations
        ensemble_mask = self.ensemble_df.apply(
            lambda x: (x['train_site'], x['test_site']) in common_combinations, axis=1
        )
        baseline_mask = self.baseline_df.apply(
            lambda x: (x['train_site'], x['test_site']) in common_combinations, axis=1
        )
        
        self.ensemble_df = self.ensemble_df[ensemble_mask].copy()
        self.baseline_df = self.baseline_df[baseline_mask].copy()
        
        # Add domain classification
        self.ensemble_df['domain_type'] = self.ensemble_df.apply(
            lambda x: 'in_domain' if x['train_site'] == x['test_site'] else 'out_domain', axis=1
        )
        self.baseline_df['domain_type'] = self.baseline_df.apply(
            lambda x: 'in_domain' if x['train_site'] == x['test_site'] else 'out_domain', axis=1
        )
        
        # Create match keys
        self.ensemble_df['match_key'] = (
            self.ensemble_df['train_site'] + '_' + self.ensemble_df['test_site']
        )
        self.baseline_df['match_key'] = (
            self.baseline_df['train_site'] + '_' + self.baseline_df['test_site']
        )
        
        print(f"Final ensemble experiments: {len(self.ensemble_df)}")
        print(f"Final baseline experiments: {len(self.baseline_df)}")
        print(f"Common site combinations: {len(common_combinations)}")
        print(f"Individual baseline models: {sorted(self.baseline_df['model_name'].unique())}")
        print(f"Ensemble types: 3-model ({(self.ensemble_df['ensemble_size'] == 3).sum()}), "
              f"5-model ({(self.ensemble_df['ensemble_size'] == 5).sum()})")
        
        return self.ensemble_df, self.baseline_df
    
    def _parse_ensemble_info(self, df):
        """Parse ensemble information from experiment_id."""
        df = df.copy()
        
        # Extract ensemble size
        df['ensemble_size'] = df['experiment_id'].str.extract(r'size(\d+)_').astype(int)
        
        # Extract selection strategy
        df['selection_strategy'] = df['experiment_id'].str.extract(r'ensemble_soft_(\w+)_size')
        
        # Extract ensemble models
        def extract_models(exp_id):
            parts = exp_id.split('_')
            models = []
            found_size = False
            for part in parts:
                if part.startswith('size') and part[4:].isdigit():
                    found_size = True
                    continue
                if found_size and part in ['in', 'out']:
                    break
                if found_size:
                    models.append(part)
            return '_'.join(sorted(models))
        
        df['ensemble_models'] = df['experiment_id'].apply(extract_models)
        df['ensemble_type'] = df['ensemble_size'].astype(str) + '-model_' + df['ensemble_models']
        
        return df
    
    def _get_common_combinations(self):
        """Get train/test site combinations common to both datasets."""
        ensemble_combinations = set(zip(self.ensemble_df['train_site'], self.ensemble_df['test_site']))
        baseline_combinations = set(zip(self.baseline_df['train_site'], self.baseline_df['test_site']))
        return ensemble_combinations & baseline_combinations
    
    def perform_individual_model_comparisons(self):
        """Perform head-to-head comparisons between ensembles and individual baseline models."""
        print("=== PERFORMING INDIVIDUAL MODEL COMPARISONS ===")
        
        baseline_models = sorted(self.baseline_df['model_name'].unique())
        metrics = ['csmf_accuracy', 'cod_accuracy']
        
        # Storage for all comparisons
        self.individual_comparisons = {}
        
        # Compare each ensemble type against each individual baseline model
        for ensemble_size in [3, 5]:
            ensemble_subset = self.ensemble_df[self.ensemble_df['ensemble_size'] == ensemble_size].copy()
            
            self.individual_comparisons[f'{ensemble_size}_model'] = {}
            
            for baseline_model in baseline_models:
                baseline_subset = self.baseline_df[self.baseline_df['model_name'] == baseline_model].copy()
                
                print(f"Comparing {ensemble_size}-model ensembles vs {baseline_model}...")
                
                # Perform matching by site combination
                matched_results = self._match_experiments(ensemble_subset, baseline_subset)
                
                if len(matched_results) == 0:
                    print(f"  No matches found for {baseline_model}")
                    continue
                
                # Calculate comparison statistics
                comparison_stats = self._calculate_comparison_stats(matched_results, metrics)
                
                self.individual_comparisons[f'{ensemble_size}_model'][baseline_model] = {
                    'matched_pairs': len(matched_results),
                    'stats': comparison_stats,
                    'raw_data': matched_results
                }
                
                print(f"  Matched pairs: {len(matched_results)}")
                for metric in metrics:
                    ensemble_mean = comparison_stats[metric]['ensemble_mean']
                    baseline_mean = comparison_stats[metric]['baseline_mean']
                    p_value = comparison_stats[metric]['p_value']
                    effect_size = comparison_stats[metric]['effect_size']
                    
                    print(f"    {metric}: Ensemble={ensemble_mean:.4f}, "
                          f"{baseline_model}={baseline_mean:.4f}, p={p_value:.4f}, d={effect_size:.3f}")
        
        return self.individual_comparisons
    
    def _match_experiments(self, ensemble_df, baseline_df):
        """Match ensemble and baseline experiments by site combination."""
        matched_results = []
        
        # Group by match key (train_site_test_site)
        ensemble_groups = ensemble_df.groupby('match_key')
        baseline_groups = baseline_df.groupby('match_key')
        
        for match_key in ensemble_groups.groups.keys():
            if match_key in baseline_groups.groups:
                ensemble_group = ensemble_groups.get_group(match_key)
                baseline_group = baseline_groups.get_group(match_key)
                
                # For each ensemble experiment, find the corresponding baseline experiment
                for _, ensemble_row in ensemble_group.iterrows():
                    # Match with all baseline experiments for this site combination
                    for _, baseline_row in baseline_group.iterrows():
                        matched_results.append({
                            'match_key': match_key,
                            'train_site': ensemble_row['train_site'],
                            'test_site': ensemble_row['test_site'],
                            'domain_type': ensemble_row['domain_type'],
                            'ensemble_type': ensemble_row['ensemble_type'],
                            'ensemble_size': ensemble_row['ensemble_size'],
                            'ensemble_models': ensemble_row['ensemble_models'],
                            'baseline_model': baseline_row['model_name'],
                            'ensemble_csmf': ensemble_row['csmf_accuracy'],
                            'baseline_csmf': baseline_row['csmf_accuracy'],
                            'ensemble_cod': ensemble_row['cod_accuracy'],
                            'baseline_cod': baseline_row['cod_accuracy'],
                            'csmf_diff': ensemble_row['csmf_accuracy'] - baseline_row['csmf_accuracy'],
                            'cod_diff': ensemble_row['cod_accuracy'] - baseline_row['cod_accuracy']
                        })
        
        return pd.DataFrame(matched_results)
    
    def _calculate_comparison_stats(self, matched_df, metrics):
        """Calculate statistical comparisons between ensemble and baseline performance."""
        stats_results = {}
        
        for metric in metrics:
            ensemble_col = f'ensemble_{metric.split("_")[0]}'
            baseline_col = f'baseline_{metric.split("_")[0]}'
            diff_col = f'{metric.split("_")[0]}_diff'
            
            ensemble_values = matched_df[ensemble_col].values
            baseline_values = matched_df[baseline_col].values
            differences = matched_df[diff_col].values
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(ensemble_values, baseline_values)
            
            # Effect size (Cohen's d for paired samples)
            effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
            
            # Win/Loss/Tie counts
            wins = (differences > 0).sum()
            losses = (differences < 0).sum()
            ties = (differences == 0).sum()
            
            stats_results[metric] = {
                'ensemble_mean': np.mean(ensemble_values),
                'ensemble_std': np.std(ensemble_values),
                'baseline_mean': np.mean(baseline_values),
                'baseline_std': np.std(baseline_values),
                'mean_difference': np.mean(differences),
                'std_difference': np.std(differences),
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'win_rate': wins / len(differences)
            }
        
        return stats_results
    
    def generate_performance_rankings(self):
        """Generate comprehensive performance rankings for all models."""
        print("=== GENERATING PERFORMANCE RANKINGS ===")
        
        metrics = ['csmf_accuracy', 'cod_accuracy']
        
        # Calculate overall performance for each model type
        rankings = {}
        
        # Individual baseline models
        for model in self.baseline_df['model_name'].unique():
            model_data = self.baseline_df[self.baseline_df['model_name'] == model]
            rankings[model] = {
                'type': 'individual',
                'experiments': len(model_data),
                'csmf_mean': model_data['csmf_accuracy'].mean(),
                'csmf_std': model_data['csmf_accuracy'].std(),
                'cod_mean': model_data['cod_accuracy'].mean(),
                'cod_std': model_data['cod_accuracy'].std(),
                'in_domain_csmf': model_data[model_data['domain_type'] == 'in_domain']['csmf_accuracy'].mean(),
                'out_domain_csmf': model_data[model_data['domain_type'] == 'out_domain']['csmf_accuracy'].mean(),
                'in_domain_cod': model_data[model_data['domain_type'] == 'in_domain']['cod_accuracy'].mean(),
                'out_domain_cod': model_data[model_data['domain_type'] == 'out_domain']['cod_accuracy'].mean()
            }
        
        # Ensemble models by size and type
        for ensemble_size in [3, 5]:
            ensemble_data = self.ensemble_df[self.ensemble_df['ensemble_size'] == ensemble_size]
            model_name = f'{ensemble_size}-model_ensemble'
            
            rankings[model_name] = {
                'type': 'ensemble',
                'experiments': len(ensemble_data),
                'csmf_mean': ensemble_data['csmf_accuracy'].mean(),
                'csmf_std': ensemble_data['csmf_accuracy'].std(),
                'cod_mean': ensemble_data['cod_accuracy'].mean(),
                'cod_std': ensemble_data['cod_accuracy'].std(),
                'in_domain_csmf': ensemble_data[ensemble_data['domain_type'] == 'in_domain']['csmf_accuracy'].mean(),
                'out_domain_csmf': ensemble_data[ensemble_data['domain_type'] == 'out_domain']['csmf_accuracy'].mean(),
                'in_domain_cod': ensemble_data[ensemble_data['domain_type'] == 'in_domain']['cod_accuracy'].mean(),
                'out_domain_cod': ensemble_data[ensemble_data['domain_type'] == 'out_domain']['cod_accuracy'].mean()
            }
        
        # Convert to DataFrame for easier manipulation
        rankings_df = pd.DataFrame(rankings).T
        rankings_df = rankings_df.fillna(0)  # Fill NaN values with 0
        
        # Sort by overall CSMF accuracy (primary metric)
        rankings_df = rankings_df.sort_values('csmf_mean', ascending=False)
        
        self.performance_rankings = rankings_df
        
        print("Performance Rankings (by CSMF accuracy):")
        for i, (model, row) in enumerate(rankings_df.iterrows(), 1):
            print(f"{i:2d}. {model:20s} - CSMF: {row['csmf_mean']:.4f} ± {row['csmf_std']:.4f}, "
                  f"COD: {row['cod_mean']:.4f} ± {row['cod_std']:.4f}")
        
        return rankings_df
    
    def analyze_specific_comparisons(self):
        """Analyze specific head-to-head comparisons to answer key questions."""
        print("=== ANALYZING SPECIFIC COMPARISONS ===")
        
        analysis_results = {}
        
        # Question 1: Which individual model performs best overall?
        best_individual = self.performance_rankings[
            self.performance_rankings['type'] == 'individual'
        ].iloc[0]
        analysis_results['best_individual_model'] = {
            'model': best_individual.name,
            'csmf_accuracy': best_individual['csmf_mean'],
            'cod_accuracy': best_individual['cod_mean']
        }
        
        # Question 2: Do any ensemble combinations beat the best individual model?
        best_individual_csmf = best_individual['csmf_mean']
        ensemble_rankings = self.performance_rankings[
            self.performance_rankings['type'] == 'ensemble'
        ]
        
        better_ensembles = ensemble_rankings[
            ensemble_rankings['csmf_mean'] > best_individual_csmf
        ]
        
        analysis_results['ensembles_beating_best_individual'] = {
            'count': len(better_ensembles),
            'ensembles': better_ensembles.index.tolist(),
            'improvements': (better_ensembles['csmf_mean'] - best_individual_csmf).tolist()
        }
        
        # Question 3: Is XGBoost consistently the best individual model?
        xgboost_rank = None
        individual_rankings = self.performance_rankings[
            self.performance_rankings['type'] == 'individual'
        ]
        for i, (model, _) in enumerate(individual_rankings.iterrows(), 1):
            if model == 'xgboost':
                xgboost_rank = i
                break
        
        analysis_results['xgboost_performance'] = {
            'rank': xgboost_rank,
            'is_best_individual': xgboost_rank == 1,
            'csmf_accuracy': individual_rankings.loc['xgboost', 'csmf_mean'] if 'xgboost' in individual_rankings.index else None
        }
        
        # Question 4: Which individual models do ensembles typically beat/lose to?
        win_loss_summary = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
        
        for ensemble_size in [3, 5]:
            if f'{ensemble_size}_model' in self.individual_comparisons:
                for baseline_model, comparison in self.individual_comparisons[f'{ensemble_size}_model'].items():
                    csmf_stats = comparison['stats']['csmf_accuracy']
                    wins = csmf_stats['wins']
                    losses = csmf_stats['losses']
                    total = wins + losses + csmf_stats['ties']
                    
                    win_loss_summary[baseline_model]['wins'] += wins
                    win_loss_summary[baseline_model]['losses'] += losses
                    win_loss_summary[baseline_model]['total'] += total
        
        # Calculate win rates
        for model in win_loss_summary:
            total = win_loss_summary[model]['total']
            if total > 0:
                win_loss_summary[model]['win_rate'] = win_loss_summary[model]['wins'] / total
            else:
                win_loss_summary[model]['win_rate'] = 0
        
        analysis_results['ensemble_vs_individual_summary'] = dict(win_loss_summary)
        
        # Question 5: Site-specific patterns
        site_analysis = {}
        for match_key in self.ensemble_df['match_key'].unique():
            train_site, test_site = match_key.split('_')
            domain_type = 'in_domain' if train_site == test_site else 'out_domain'
            
            ensemble_perf = self.ensemble_df[self.ensemble_df['match_key'] == match_key]['csmf_accuracy'].mean()
            baseline_perf = self.baseline_df[self.baseline_df['match_key'] == match_key]['csmf_accuracy'].mean()
            
            site_analysis[match_key] = {
                'train_site': train_site,
                'test_site': test_site,
                'domain_type': domain_type,
                'ensemble_csmf': ensemble_perf,
                'baseline_csmf': baseline_perf,
                'ensemble_advantage': ensemble_perf - baseline_perf
            }
        
        analysis_results['site_specific_patterns'] = site_analysis
        
        return analysis_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations for the analysis."""
        print("=== CREATING VISUALIZATIONS ===")
        
        # 1. Overall Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Ensemble vs Individual Model Performance Analysis', fontsize=16)
        
        # Performance rankings bar plot
        ax1 = axes[0, 0]
        models = self.performance_rankings.index
        csmf_means = self.performance_rankings['csmf_mean']
        colors = ['red' if 'ensemble' in model else 'blue' for model in models]
        
        bars = ax1.bar(range(len(models)), csmf_means, color=colors, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('CSMF Accuracy')
        ax1.set_title('Overall Model Performance Rankings')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add legend
        ax1.legend(['Ensemble Models', 'Individual Models'], loc='upper right')
        
        # 2. Domain-specific performance comparison
        ax2 = axes[0, 1]
        
        # Prepare data for domain comparison
        domain_data = []
        for model in self.performance_rankings.index:
            row = self.performance_rankings.loc[model]
            domain_data.append({
                'model': model,
                'in_domain': row['in_domain_csmf'],
                'out_domain': row['out_domain_csmf'],
                'type': row['type']
            })
        
        domain_df = pd.DataFrame(domain_data)
        
        # Create grouped bar plot
        x = np.arange(len(domain_df['model']))
        width = 0.35
        
        ax2.bar(x - width/2, domain_df['in_domain'], width, label='In-Domain', alpha=0.7)
        ax2.bar(x + width/2, domain_df['out_domain'], width, label='Out-Domain', alpha=0.7)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('CSMF Accuracy')
        ax2.set_title('In-Domain vs Out-Domain Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(domain_df['model'], rotation=45, ha='right')
        ax2.legend()
        
        # 3. Ensemble size comparison
        ax3 = axes[1, 0]
        
        ensemble_3_data = self.ensemble_df[self.ensemble_df['ensemble_size'] == 3]
        ensemble_5_data = self.ensemble_df[self.ensemble_df['ensemble_size'] == 5]
        
        ax3.boxplot([ensemble_3_data['csmf_accuracy'], ensemble_5_data['csmf_accuracy']], 
                   labels=['3-Model Ensembles', '5-Model Ensembles'])
        ax3.set_ylabel('CSMF Accuracy')
        ax3.set_title('Ensemble Size Performance Comparison')
        
        # 4. Win rate against individual models
        ax4 = axes[1, 1]
        
        # Calculate win rates for each ensemble size against each individual model
        win_rates_data = []
        
        for ensemble_size in [3, 5]:
            if f'{ensemble_size}_model' in self.individual_comparisons:
                for baseline_model, comparison in self.individual_comparisons[f'{ensemble_size}_model'].items():
                    win_rate = comparison['stats']['csmf_accuracy']['win_rate']
                    win_rates_data.append({
                        'ensemble_size': f'{ensemble_size}-model',
                        'baseline_model': baseline_model,
                        'win_rate': win_rate
                    })
        
        if win_rates_data:
            win_rates_df = pd.DataFrame(win_rates_data)
            
            # Create pivot table for heatmap
            pivot_data = win_rates_df.pivot(index='baseline_model', 
                                          columns='ensemble_size', 
                                          values='win_rate')
            
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       ax=ax4, vmin=0, vmax=1)
            ax4.set_title('Ensemble Win Rate vs Individual Models')
            ax4.set_xlabel('Ensemble Type')
            ax4.set_ylabel('Individual Model')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_performance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_final_report(self, analysis_results):
        """Generate the final comprehensive analysis report."""
        print("=== GENERATING FINAL REPORT ===")
        
        report_path = self.output_dir / 'final_corrected_ensemble_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Final Corrected Ensemble vs Individual Baseline Model Analysis\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report provides a comprehensive comparison between ensemble VA models ")
            f.write("and individual baseline models using properly matched train/test site combinations.\n\n")
            
            # Key findings
            f.write("### Key Findings\n\n")
            
            best_model = analysis_results['best_individual_model']['model']
            best_csmf = analysis_results['best_individual_model']['csmf_accuracy']
            
            f.write(f"1. **Best Individual Model**: {best_model} (CSMF Accuracy: {best_csmf:.4f})\n")
            
            ensemble_count = analysis_results['ensembles_beating_best_individual']['count']
            f.write(f"2. **Ensemble Performance**: {ensemble_count} ensemble configurations ")
            f.write(f"outperform the best individual model\n")
            
            xgb_best = analysis_results['xgboost_performance']['is_best_individual']
            xgb_rank = analysis_results['xgboost_performance']['rank']
            f.write(f"3. **XGBoost Performance**: {'Is' if xgb_best else 'Is not'} the best individual model ")
            f.write(f"(Rank: {xgb_rank})\n\n")
            
            # Performance rankings
            f.write("## Complete Performance Rankings\n\n")
            f.write("| Rank | Model | Type | CSMF Accuracy | COD Accuracy | Experiments |\n")
            f.write("|------|-------|------|---------------|--------------|-------------|\n")
            
            for i, (model, row) in enumerate(self.performance_rankings.iterrows(), 1):
                f.write(f"| {i} | {model} | {row['type']} | ")
                f.write(f"{row['csmf_mean']:.4f} ± {row['csmf_std']:.4f} | ")
                f.write(f"{row['cod_mean']:.4f} ± {row['cod_std']:.4f} | ")
                f.write(f"{int(row['experiments'])} |\n")
            
            f.write("\n")
            
            # Individual model comparisons
            f.write("## Head-to-Head Comparisons\n\n")
            
            for ensemble_size in [3, 5]:
                if f'{ensemble_size}_model' in self.individual_comparisons:
                    f.write(f"### {ensemble_size}-Model Ensembles vs Individual Models\n\n")
                    f.write("| Individual Model | Matched Pairs | Win Rate | Mean Difference (CSMF) | p-value | Effect Size |\n")
                    f.write("|------------------|---------------|----------|----------------------|---------|-------------|\n")
                    
                    for baseline_model, comparison in self.individual_comparisons[f'{ensemble_size}_model'].items():
                        stats = comparison['stats']['csmf_accuracy']
                        f.write(f"| {baseline_model} | {comparison['matched_pairs']} | ")
                        f.write(f"{stats['win_rate']:.3f} | {stats['mean_difference']:.4f} | ")
                        f.write(f"{stats['p_value']:.4f} | {stats['effect_size']:.3f} |\n")
                    
                    f.write("\n")
            
            # Site-specific analysis
            f.write("## Site-Specific Performance Patterns\n\n")
            f.write("| Train Site | Test Site | Domain Type | Ensemble CSMF | Baseline CSMF | Advantage |\n")
            f.write("|------------|-----------|------------|---------------|---------------|----------|\n")
            
            for site_key, site_data in analysis_results['site_specific_patterns'].items():
                f.write(f"| {site_data['train_site']} | {site_data['test_site']} | ")
                f.write(f"{site_data['domain_type']} | {site_data['ensemble_csmf']:.4f} | ")
                f.write(f"{site_data['baseline_csmf']:.4f} | {site_data['ensemble_advantage']:.4f} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Best overall model
            best_overall = self.performance_rankings.iloc[0]
            f.write(f"1. **Best Overall Model**: {best_overall.name} ")
            f.write(f"(CSMF: {best_overall['csmf_mean']:.4f}, COD: {best_overall['cod_mean']:.4f})\n")
            
            # Best individual model
            f.write(f"2. **Best Individual Model**: {best_model} for scenarios requiring single models\n")
            
            # Ensemble recommendations
            best_ensemble = self.performance_rankings[
                self.performance_rankings['type'] == 'ensemble'
            ].iloc[0]
            f.write(f"3. **Best Ensemble**: {best_ensemble.name} when computational resources allow\n")
            
            # Domain-specific recommendations
            f.write("4. **Domain-Specific Usage**:\n")
            for model in ['3-model_ensemble', '5-model_ensemble']:
                if model in self.performance_rankings.index:
                    row = self.performance_rankings.loc[model]
                    in_domain = row['in_domain_csmf']
                    out_domain = row['out_domain_csmf']
                    if in_domain > out_domain:
                        f.write(f"   - {model}: Better for in-domain tasks (gap: {in_domain - out_domain:.4f})\n")
                    else:
                        f.write(f"   - {model}: Better for out-domain tasks (gap: {out_domain - in_domain:.4f})\n")
            
            f.write("\n")
            
            # Methodology notes
            f.write("## Methodology Notes\n\n")
            f.write("- **Data Matching**: Only used exact train/test site combinations present in both datasets\n")
            f.write("- **Fair Comparison**: Filtered baseline data to training_fraction=1.0 only\n")
            f.write("- **Statistical Testing**: Used paired t-tests for significance testing\n")
            f.write("- **Effect Sizes**: Calculated Cohen's d for practical significance\n")
            f.write("- **Metrics**: Primary focus on CSMF accuracy, secondary on COD accuracy\n")
        
        print(f"Final report saved to: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete corrected analysis pipeline."""
        print("=== STARTING FINAL CORRECTED ENSEMBLE ANALYSIS ===\n")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Perform individual model comparisons
        self.perform_individual_model_comparisons()
        
        # Generate performance rankings
        self.generate_performance_rankings()
        
        # Analyze specific comparisons
        analysis_results = self.analyze_specific_comparisons()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate final report
        report_path = self.generate_final_report(analysis_results)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Results saved to: {self.output_dir}")
        print(f"Final report: {report_path}")
        
        return {
            'performance_rankings': self.performance_rankings,
            'individual_comparisons': self.individual_comparisons,
            'analysis_results': analysis_results,
            'report_path': report_path
        }

def main():
    """Main execution function."""
    analyzer = FinalCorrectedEnsembleAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    results = main()