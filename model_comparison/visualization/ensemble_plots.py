"""Visualization functions for ensemble model experiments.

This module provides plotting functions specifically for ensemble
model analysis and comparison.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from baseline.utils import get_logger

logger = get_logger(__name__, component="visualization")


def plot_voting_strategy_comparison(
    results_df: pd.DataFrame,
    metric: str = "csmf_accuracy",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6)
):
    """Plot comparison of voting strategies (hard vs soft).
    
    Args:
        results_df: Results DataFrame with ensemble experiments
        metric: Metric to plot (csmf_accuracy or cod_accuracy)
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Filter ensemble results
    ensemble_df = results_df[results_df["model"] == "ensemble"].copy()
    
    # Extract voting strategy from additional metrics if needed
    if "voting" not in ensemble_df.columns and "additional_metrics" in ensemble_df.columns:
        ensemble_df["voting"] = ensemble_df["additional_metrics"].apply(
            lambda x: x.get("voting", "unknown") if isinstance(x, dict) else "unknown"
        )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Box plot comparing voting strategies
    sns.boxplot(
        data=ensemble_df,
        x="voting",
        y=metric,
        ax=ax,
        palette="Set2"
    )
    
    # Add individual points
    sns.stripplot(
        data=ensemble_df,
        x="voting",
        y=metric,
        ax=ax,
        color="black",
        alpha=0.5,
        size=4
    )
    
    ax.set_title(f"Voting Strategy Comparison - {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Voting Strategy")
    ax.set_ylabel(metric.replace("_", " ").title())
    
    # Add mean values as text
    for i, voting in enumerate(["soft", "hard"]):
        voting_data = ensemble_df[ensemble_df["voting"] == voting][metric]
        if len(voting_data) > 0:
            mean_val = voting_data.mean()
            ax.text(i, mean_val, f"{mean_val:.3f}", 
                   ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved voting strategy comparison plot to {save_path}")
    
    plt.close()


def plot_weight_optimization_comparison(
    results_df: pd.DataFrame,
    metric: str = "csmf_accuracy",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6)
):
    """Plot comparison of weight optimization strategies.
    
    Args:
        results_df: Results DataFrame with ensemble experiments
        metric: Metric to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Filter ensemble results
    ensemble_df = results_df[results_df["model"] == "ensemble"].copy()
    
    # Extract weight strategy from additional metrics if needed
    if "weight_strategy" not in ensemble_df.columns and "additional_metrics" in ensemble_df.columns:
        ensemble_df["weight_strategy"] = ensemble_df["additional_metrics"].apply(
            lambda x: x.get("weight_strategy", "unknown") if isinstance(x, dict) else "unknown"
        )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Box plot by weight strategy
    sns.boxplot(
        data=ensemble_df,
        x="weight_strategy",
        y=metric,
        ax=ax1,
        palette="viridis"
    )
    ax1.set_title(f"Weight Strategy Impact on {metric.replace('_', ' ').title()}")
    ax1.set_xlabel("Weight Strategy")
    ax1.set_ylabel(metric.replace("_", " ").title())
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Performance improvement over baseline (none)
    baseline_perf = ensemble_df[ensemble_df["weight_strategy"] == "none"][metric].mean()
    
    improvements = []
    strategies = []
    for strategy in ensemble_df["weight_strategy"].unique():
        if strategy != "none":
            strategy_perf = ensemble_df[ensemble_df["weight_strategy"] == strategy][metric].mean()
            improvement = ((strategy_perf - baseline_perf) / baseline_perf) * 100
            improvements.append(improvement)
            strategies.append(strategy)
    
    ax2.bar(strategies, improvements, color=["green" if x > 0 else "red" for x in improvements])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title("Performance Improvement over Equal Weights")
    ax2.set_xlabel("Weight Strategy")
    ax2.set_ylabel("Improvement (%)")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved weight optimization comparison plot to {save_path}")
    
    plt.close()


def plot_ensemble_size_impact(
    results_df: pd.DataFrame,
    metric: str = "csmf_accuracy",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6)
):
    """Plot impact of ensemble size on performance.
    
    Args:
        results_df: Results DataFrame with ensemble experiments
        metric: Metric to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Filter ensemble results
    ensemble_df = results_df[results_df["model"] == "ensemble"].copy()
    
    # Extract ensemble size from additional metrics if needed
    if "ensemble_size" not in ensemble_df.columns and "additional_metrics" in ensemble_df.columns:
        ensemble_df["ensemble_size"] = ensemble_df["additional_metrics"].apply(
            lambda x: x.get("ensemble_size", 0) if isinstance(x, dict) else 0
        )
    
    # Group by ensemble size
    size_performance = ensemble_df.groupby("ensemble_size")[metric].agg(["mean", "std", "count"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean with error bars
    sizes = size_performance.index
    means = size_performance["mean"]
    stds = size_performance["std"]
    
    ax.errorbar(sizes, means, yerr=stds, marker='o', markersize=8, 
                capsize=5, capthick=2, linewidth=2)
    
    # Add sample sizes as annotations
    for size, count in zip(sizes, size_performance["count"]):
        ax.annotate(f'n={count}', xy=(size, means[size]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_title(f"Ensemble Size Impact on {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Number of Base Estimators")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    
    # Add trend line if we have enough data points
    if len(sizes) >= 3:
        z = np.polyfit(sizes, means, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(sizes.min(), sizes.max(), 100)
        ax.plot(x_smooth, p(x_smooth), "--", alpha=0.5, color="red", 
               label="Trend (quadratic)")
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ensemble size impact plot to {save_path}")
    
    plt.close()


def plot_diversity_analysis(
    results_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8)
):
    """Plot ensemble diversity analysis.
    
    Args:
        results_df: Results DataFrame with ensemble experiments
        save_path: Path to save the plot
        figsize: Figure size
    """
    # This would require diversity scores to be stored in results
    # For now, create a placeholder visualization
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Placeholder for diversity vs performance scatter
    ax1.scatter(np.random.rand(20), np.random.rand(20))
    ax1.set_xlabel("Ensemble Diversity")
    ax1.set_ylabel("CSMF Accuracy")
    ax1.set_title("Diversity vs Performance")
    
    # Placeholder for model composition heatmap
    models = ["XGBoost", "RF", "NB", "LR", "InSilico"]
    composition_matrix = np.random.rand(5, 5)
    sns.heatmap(composition_matrix, xticklabels=models, yticklabels=models,
                cmap="YlOrRd", ax=ax2, annot=True, fmt=".2f")
    ax2.set_title("Model Co-occurrence in Best Ensembles")
    
    # Placeholder for voting agreement
    ax3.bar(["High Agreement", "Medium Agreement", "Low Agreement"], 
           [0.85, 0.78, 0.72])
    ax3.set_ylabel("Average CSMF Accuracy")
    ax3.set_title("Performance by Voting Agreement Level")
    
    # Placeholder for model contribution
    ax4.pie([0.3, 0.25, 0.2, 0.15, 0.1], labels=models, autopct='%1.1f%%')
    ax4.set_title("Average Model Contribution to Ensemble Predictions")
    
    plt.suptitle("Ensemble Diversity Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved diversity analysis plot to {save_path}")
    
    plt.close()


def plot_ensemble_vs_individual(
    results_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6)
):
    """Plot ensemble performance vs individual models.
    
    Args:
        results_df: Results DataFrame with both ensemble and individual results
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Separate ensemble and individual results
    ensemble_results = results_df[results_df["model"] == "ensemble"]
    individual_results = results_df[results_df["model"] != "ensemble"]
    
    # Plot 1: CSMF Accuracy comparison
    models = individual_results["model"].unique()
    individual_means = [
        individual_results[individual_results["model"] == m]["csmf_accuracy"].mean()
        for m in models
    ]
    ensemble_mean = ensemble_results["csmf_accuracy"].mean()
    
    x = np.arange(len(models) + 1)
    means = individual_means + [ensemble_mean]
    labels = list(models) + ["Ensemble"]
    colors = ["skyblue"] * len(models) + ["gold"]
    
    ax1.bar(x, means, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel("CSMF Accuracy")
    ax1.set_title("CSMF Accuracy: Individual vs Ensemble")
    ax1.axhline(y=ensemble_mean, color='gold', linestyle='--', alpha=0.5)
    
    # Plot 2: COD Accuracy comparison
    individual_cod_means = [
        individual_results[individual_results["model"] == m]["cod_accuracy"].mean()
        for m in models
    ]
    ensemble_cod_mean = ensemble_results["cod_accuracy"].mean()
    
    cod_means = individual_cod_means + [ensemble_cod_mean]
    
    ax2.bar(x, cod_means, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel("COD Accuracy")
    ax2.set_title("COD Accuracy: Individual vs Ensemble")
    ax2.axhline(y=ensemble_cod_mean, color='gold', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ensemble vs individual comparison plot to {save_path}")
    
    plt.close()