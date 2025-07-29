"""Visualization functions for model comparison results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_model_comparison(results: pd.DataFrame, output_path: str):
    """Create comprehensive comparison plots.

    Args:
        results: DataFrame with experiment results
        output_path: Path to save the plot
    """
    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. In-domain vs Out-domain comparison
    plot_domain_comparison(results, axes[0, 0])

    # 2. Training size impact
    plot_training_size_impact(results, axes[0, 1])

    # 3. Site-specific performance heatmap
    plot_site_heatmap(results, axes[1, 0])

    # 4. Statistical significance
    plot_metric_distributions(results, axes[1, 1])

    plt.suptitle("VA34 Model Comparison Results", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Generate additional plots
    output_dir = Path(output_path).parent

    # Performance by model plot
    plot_model_performance(results, output_dir / "model_performance.png")

    # Generalization gap analysis
    plot_generalization_gap(results, output_dir / "generalization_gap.png")


def plot_domain_comparison(results: pd.DataFrame, ax):
    """Plot in-domain vs out-domain performance."""
    # Filter data and exclude failed experiments
    domain_data = results[
        (results["experiment_type"].isin(["in_domain", "out_domain"])) &
        (results["csmf_accuracy"] > 0)  # Exclude failed experiments
    ]

    if domain_data.empty:
        ax.text(0.5, 0.5, "No domain comparison data", ha="center", va="center")
        ax.set_title("CSMF Accuracy: In-domain vs Out-domain")
        return

    # Get unique models that have successful experiments
    successful_models = sorted(domain_data["model"].unique())
    
    if not successful_models:
        ax.text(0.5, 0.5, "No successful experiments", ha="center", va="center")
        ax.set_title("CSMF Accuracy: In-domain vs Out-domain")
        return

    # Calculate means and confidence intervals
    summary_data = []
    for exp_type in ["in_domain", "out_domain"]:
        for model in successful_models:
            model_data = domain_data[
                (domain_data["experiment_type"] == exp_type) & 
                (domain_data["model"] == model)
            ]
            
            if not model_data.empty:
                # Calculate mean and CI bounds
                mean_acc = model_data["csmf_accuracy"].mean()
                
                # Check if CI values are available
                ci_lower_vals = model_data["csmf_accuracy_ci_lower"].dropna()
                ci_upper_vals = model_data["csmf_accuracy_ci_upper"].dropna()
                
                if len(ci_lower_vals) > 0 and len(ci_upper_vals) > 0:
                    ci_lower_mean = ci_lower_vals.mean()
                    ci_upper_mean = ci_upper_vals.mean()
                else:
                    # Use standard error if no CI available
                    std_err = model_data["csmf_accuracy"].std() / np.sqrt(len(model_data))
                    ci_lower_mean = mean_acc - 1.96 * std_err
                    ci_upper_mean = mean_acc + 1.96 * std_err
                
                summary_data.append({
                    "experiment_type": exp_type,
                    "model": model,
                    "csmf_accuracy_mean": mean_acc,
                    "csmf_accuracy_ci_lower_mean": ci_lower_mean,
                    "csmf_accuracy_ci_upper_mean": ci_upper_mean,
                })
    
    if not summary_data:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.set_title("CSMF Accuracy: In-domain vs Out-domain")
        return
        
    summary = pd.DataFrame(summary_data)

    # Create grouped bar plot
    x = np.arange(len(successful_models))
    width = 0.35

    for i, exp_type in enumerate(["in_domain", "out_domain"]):
        data = summary[summary["experiment_type"] == exp_type]
        
        if not data.empty:
            offset = width * (i - 0.5)
            
            # Calculate error bars (ensure non-negative)
            yerr_lower = np.maximum(0, data["csmf_accuracy_mean"] - data["csmf_accuracy_ci_lower_mean"])
            yerr_upper = np.maximum(0, data["csmf_accuracy_ci_upper_mean"] - data["csmf_accuracy_mean"])

            bars = ax.bar(
                x + offset,
                data["csmf_accuracy_mean"],
                width,
                label=exp_type.replace("_", " ").title(),
                yerr=[yerr_lower, yerr_upper],
                capsize=5,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("CSMF Accuracy")
    ax.set_title("CSMF Accuracy: In-domain vs Out-domain")
    ax.set_xticks(x)
    ax.set_xticklabels(successful_models)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)


def plot_training_size_impact(results: pd.DataFrame, ax):
    """Plot impact of training data size on performance."""
    # Filter data and exclude failed experiments
    size_data = results[
        (results["experiment_type"] == "training_size") &
        (results["csmf_accuracy"] > 0)
    ]

    if size_data.empty:
        ax.text(0.5, 0.5, "No training size data", ha="center", va="center")
        ax.set_title("Impact of Training Data Size")
        return

    # Group by training fraction and model
    for model in sorted(size_data["model"].unique()):
        model_data = size_data[size_data["model"] == model]

        # Sort by training fraction
        model_data = model_data.sort_values("training_fraction")

        # Check if CI values are available
        has_ci = model_data["csmf_accuracy_ci_lower"].notna().any() and model_data["csmf_accuracy_ci_upper"].notna().any()
        
        if has_ci:
            # Calculate error bars (ensure non-negative)
            yerr_lower = np.maximum(0, model_data["csmf_accuracy"] - model_data["csmf_accuracy_ci_lower"])
            yerr_upper = np.maximum(0, model_data["csmf_accuracy_ci_upper"] - model_data["csmf_accuracy"])
            yerr = [yerr_lower, yerr_upper]
        else:
            yerr = None

        # Plot with error bars
        ax.errorbar(
            model_data["training_fraction"],
            model_data["csmf_accuracy"],
            yerr=yerr,
            label=model,
            marker="o",
            capsize=5,
            capthick=2,
        )

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("CSMF Accuracy")
    ax.set_title("Impact of Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)


def plot_site_heatmap(results: pd.DataFrame, ax):
    """Plot site-specific performance heatmap."""
    # Filter out-domain results and exclude failed experiments
    out_domain = results[
        (results["experiment_type"] == "out_domain") &
        (results["csmf_accuracy"] > 0)
    ]

    if out_domain.empty:
        ax.text(0.5, 0.5, "No out-domain data", ha="center", va="center")
        ax.set_title("Cross-Site CSMF Accuracy")
        return

    # Get unique models with successful experiments
    models = sorted(out_domain["model"].unique())
    
    if not models:
        ax.text(0.5, 0.5, "No successful out-domain experiments", ha="center", va="center")
        ax.set_title("Cross-Site CSMF Accuracy")
        return

    # Create pivot table for the first model with data
    for i, model in enumerate(models):
        model_data = out_domain[out_domain["model"] == model]

        if not model_data.empty:
            # Pivot to create matrix
            pivot = model_data.pivot_table(
                values="csmf_accuracy",
                index="test_site",
                columns="train_site",
                aggfunc="mean",
            )

            # Create subplot (only show first model)
            im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Training Site")
            ax.set_ylabel("Test Site")
            ax.set_title(f"Cross-Site CSMF Accuracy: {model}")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("CSMF Accuracy")
            break  # Only show first model


def plot_metric_distributions(results: pd.DataFrame, ax):
    """Plot distribution of metrics across experiments."""
    # Prepare data for box plot, excluding failed experiments
    plot_data = []

    for exp_type in ["in_domain", "out_domain"]:
        exp_data = results[
            (results["experiment_type"] == exp_type) &
            (results["csmf_accuracy"] > 0)
        ]

        for model in exp_data["model"].unique():
            model_data = exp_data[exp_data["model"] == model]

            for _, row in model_data.iterrows():
                plot_data.append(
                    {
                        "Model": model,
                        "Type": exp_type.replace("_", " ").title(),
                        "CSMF Accuracy": row["csmf_accuracy"],
                    }
                )

    if not plot_data:
        ax.text(0.5, 0.5, "No distribution data", ha="center", va="center")
        ax.set_title("CSMF Accuracy Distribution")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create box plot
    sns.boxplot(data=plot_df, x="Model", y="CSMF Accuracy", hue="Type", ax=ax)
    ax.set_title("CSMF Accuracy Distribution")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)


def plot_model_performance(results: pd.DataFrame, output_path: str):
    """Create detailed model performance comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out failed experiments
    successful_results = results[results["csmf_accuracy"] > 0]
    
    if successful_results.empty:
        ax1.text(0.5, 0.5, "No successful experiments", ha="center", va="center")
        ax2.text(0.5, 0.5, "No successful experiments", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # CSMF Accuracy comparison
    model_summary = (
        successful_results.groupby("model")
        .agg(
            {
                "csmf_accuracy": ["mean", "std", "min", "max"],
                "cod_accuracy": ["mean", "std", "min", "max"],
            }
        )
        .round(3)
    )

    # Plot CSMF accuracy
    models = model_summary.index
    csmf_means = model_summary[("csmf_accuracy", "mean")]
    csmf_stds = model_summary[("csmf_accuracy", "std")]

    ax1.bar(models, csmf_means, yerr=csmf_stds, capsize=10)
    ax1.set_ylabel("CSMF Accuracy")
    ax1.set_title("Average CSMF Accuracy by Model")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    # Plot COD accuracy
    cod_means = model_summary[("cod_accuracy", "mean")]
    cod_stds = model_summary[("cod_accuracy", "std")]

    ax2.bar(models, cod_means, yerr=cod_stds, capsize=10)
    ax2.set_ylabel("COD Accuracy")
    ax2.set_title("Average COD Accuracy by Model")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_generalization_gap(results: pd.DataFrame, output_path: str):
    """Plot generalization gap analysis."""
    # Filter out failed experiments
    successful_results = results[results["csmf_accuracy"] > 0]
    
    # Calculate generalization gap
    in_domain = (
        successful_results[successful_results["experiment_type"] == "in_domain"]
        .groupby("model")["csmf_accuracy"]
        .mean()
    )
    out_domain = (
        successful_results[successful_results["experiment_type"] == "out_domain"]
        .groupby("model")["csmf_accuracy"]
        .mean()
    )

    gap = in_domain - out_domain

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(gap.index, gap.values)

    # Color bars based on gap size
    for bar, g in zip(bars, gap.values):
        if g > 0.1:
            bar.set_color("red")
        elif g > 0.05:
            bar.set_color("orange")
        else:
            bar.set_color("green")

    ax.set_ylabel("Generalization Gap (In-domain - Out-domain)")
    ax.set_title("Model Generalization Gap Analysis")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (model, value) in enumerate(gap.items()):
        ax.text(
            i,
            value + 0.01 if value > 0 else value - 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value > 0 else "top",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
