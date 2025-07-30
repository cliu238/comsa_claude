#!/usr/bin/env python
"""
Model Complexity Analysis Script
Analyzes trained XGBoost models to understand overfitting patterns
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import permutation_importance

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model(model_path: Path) -> xgb.XGBClassifier:
    """Load a saved XGBoost model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def analyze_tree_structure(model: xgb.XGBClassifier) -> Dict:
    """Analyze the tree structure of an XGBoost model."""
    booster = model.get_booster()
    
    # Get tree dumps as dictionary
    trees = booster.get_dump(dump_format='json')
    
    tree_depths = []
    leaf_counts = []
    split_counts = []
    feature_usage = {}
    
    for tree_str in trees:
        tree_dict = json.loads(tree_str)
        
        # Analyze tree depth and structure
        depth = get_tree_depth(tree_dict)
        leaves = count_leaves(tree_dict)
        splits = count_splits(tree_dict)
        
        tree_depths.append(depth)
        leaf_counts.append(leaves)
        split_counts.append(splits)
        
        # Track feature usage
        collect_feature_usage(tree_dict, feature_usage)
    
    return {
        'n_trees': len(trees),
        'avg_tree_depth': np.mean(tree_depths),
        'max_tree_depth': np.max(tree_depths),
        'min_tree_depth': np.min(tree_depths),
        'std_tree_depth': np.std(tree_depths),
        'avg_leaves_per_tree': np.mean(leaf_counts),
        'avg_splits_per_tree': np.mean(split_counts),
        'n_unique_features': len(feature_usage),
        'feature_usage': feature_usage,
        'tree_depths': tree_depths,
        'leaf_counts': leaf_counts
    }


def get_tree_depth(node: Dict, current_depth: int = 0) -> int:
    """Recursively calculate tree depth."""
    if 'leaf' in node:
        return current_depth
    
    left_depth = get_tree_depth(node['children'][0], current_depth + 1)
    right_depth = get_tree_depth(node['children'][1], current_depth + 1)
    
    return max(left_depth, right_depth)


def count_leaves(node: Dict) -> int:
    """Count number of leaves in a tree."""
    if 'leaf' in node:
        return 1
    
    return sum(count_leaves(child) for child in node['children'])


def count_splits(node: Dict) -> int:
    """Count number of splits in a tree."""
    if 'leaf' in node:
        return 0
    
    return 1 + sum(count_splits(child) for child in node['children'])


def collect_feature_usage(node: Dict, feature_usage: Dict) -> None:
    """Collect feature usage statistics."""
    if 'leaf' in node:
        return
    
    feature = node.get('split', '')
    if feature:
        feature_usage[feature] = feature_usage.get(feature, 0) + 1
    
    for child in node.get('children', []):
        collect_feature_usage(child, feature_usage)


def analyze_feature_importance(model: xgb.XGBClassifier, X_sample: np.ndarray, 
                             feature_names: List[str]) -> Dict:
    """Analyze feature importance using multiple methods."""
    # Get built-in feature importance
    importance_gain = model.feature_importances_
    
    # Get importance by different metrics
    booster = model.get_booster()
    importance_weight = booster.get_score(importance_type='weight')
    importance_cover = booster.get_score(importance_type='cover')
    
    # Normalize scores
    def normalize_importance(imp_dict, feature_names):
        scores = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            scores[i] = imp_dict.get(f'f{i}', 0)
        if scores.sum() > 0:
            scores = scores / scores.sum()
        return scores
    
    weight_scores = normalize_importance(importance_weight, feature_names)
    cover_scores = normalize_importance(importance_cover, feature_names)
    
    return {
        'gain': importance_gain,
        'weight': weight_scores,
        'cover': cover_scores,
        'feature_names': feature_names
    }


def compare_model_complexities(model_paths: Dict[str, Path], 
                             data_path: Path,
                             output_dir: Path) -> None:
    """Compare complexity metrics across different models."""
    
    results = []
    
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            continue
            
        print(f"Analyzing {model_name}...")
        model = load_model(model_path)
        
        # Analyze tree structure
        tree_stats = analyze_tree_structure(model)
        
        # Create result entry
        result = {
            'model': model_name,
            **{k: v for k, v in tree_stats.items() if k not in ['feature_usage', 'tree_depths', 'leaf_counts']}
        }
        results.append(result)
        
        # Plot tree depth distribution
        plt.figure(figsize=(10, 6))
        plt.hist(tree_stats['tree_depths'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Tree Depth')
        plt.ylabel('Number of Trees')
        plt.title(f'Tree Depth Distribution - {model_name}')
        plt.savefig(output_dir / f'tree_depth_dist_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot top features
        feature_usage = tree_stats['feature_usage']
        if feature_usage:
            top_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)[:20]
            
            plt.figure(figsize=(10, 8))
            features, counts = zip(*top_features)
            y_pos = np.arange(len(features))
            plt.barh(y_pos, counts)
            plt.yticks(y_pos, features)
            plt.xlabel('Number of Splits')
            plt.title(f'Top 20 Most Used Features - {model_name}')
            plt.tight_layout()
            plt.savefig(output_dir / f'top_features_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create comparison DataFrame
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / 'model_complexity_comparison.csv', index=False)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['avg_tree_depth', 'n_unique_features', 'avg_leaves_per_tree', 'avg_splits_per_tree']
        titles = ['Average Tree Depth', 'Number of Unique Features Used', 
                 'Average Leaves per Tree', 'Average Splits per Tree']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            if metric in df.columns:
                df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
                ax.set_title(title)
                ax.set_xlabel('')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Complexity Comparison')
        plt.tight_layout()
        plt.savefig(output_dir / 'complexity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComplexity metrics saved to: {output_dir}")
        print("\nSummary:")
        print(df.to_string(index=False))


def analyze_overfitting_indicators(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze overfitting indicators from experiment results."""
    
    # Calculate overfitting score for each model configuration
    overfitting_data = []
    
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        in_domain = model_df[model_df['experiment_type'] == 'in_domain']['csmf_accuracy'].mean()
        out_domain = model_df[model_df['experiment_type'] == 'out_domain']['csmf_accuracy'].mean()
        
        if in_domain > 0:
            overfitting_score = (in_domain - out_domain) / in_domain
            overfitting_data.append({
                'model': model,
                'in_domain_csmf': in_domain,
                'out_domain_csmf': out_domain,
                'overfitting_score': overfitting_score,
                'performance_gap': overfitting_score * 100
            })
    
    overfitting_df = pd.DataFrame(overfitting_data)
    overfitting_df.to_csv(output_dir / 'overfitting_analysis.csv', index=False)
    
    # Visualize overfitting
    plt.figure(figsize=(10, 6))
    x = np.arange(len(overfitting_df))
    width = 0.35
    
    plt.bar(x - width/2, overfitting_df['in_domain_csmf'], width, label='In-Domain', alpha=0.8)
    plt.bar(x + width/2, overfitting_df['out_domain_csmf'], width, label='Out-Domain', alpha=0.8)
    
    plt.xlabel('Model Configuration')
    plt.ylabel('CSMF Accuracy')
    plt.title('In-Domain vs Out-Domain Performance')
    plt.xticks(x, overfitting_df['model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nOverfitting Analysis:")
    print(overfitting_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Analyze XGBoost model complexity and overfitting patterns')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--model-dir', type=str,
                       help='Directory containing saved models (optional)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--data-path', type=str,
                       help='Path to data file for feature analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    results_path = Path(args.results_dir) / 'va34_comparison_results.csv'
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        
        # Analyze overfitting patterns
        analyze_overfitting_indicators(results_df, output_dir)
    else:
        print(f"Warning: Results file not found at {results_path}")
    
    # Analyze model complexity if models are available
    if args.model_dir:
        model_dir = Path(args.model_dir)
        
        # Find saved models
        model_paths = {}
        for model_file in model_dir.glob('**/xgboost_*.pkl'):
            # Extract configuration from path
            config_name = model_file.parent.name
            model_paths[config_name] = model_file
        
        if model_paths:
            compare_model_complexities(model_paths, Path(args.data_path) if args.data_path else None, output_dir)
        else:
            print(f"No XGBoost models found in {model_dir}")
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()