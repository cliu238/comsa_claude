#!/usr/bin/env python
"""
Visualization of Training Size Impact on VA Model Performance
Highlights the CSMF vs COD accuracy divergence phenomenon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
data_path = Path("../insilico_xgboost_tabicl_comparison/va34_comparison_results.csv")
df = pd.read_csv(data_path)

# Create output directory
output_dir = Path(".")
output_dir.mkdir(exist_ok=True)

# Filter for training size experiments
training_df = df[df['experiment_type'] == 'training_size'].copy()

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Impact of Training Data Size on VA Model Performance', fontsize=16, fontweight='bold')

models = ['xgboost', 'insilico', 'tabicl']
colors = {'xgboost': '#1f77b4', 'insilico': '#ff7f0e', 'tabicl': '#2ca02c'}

# Plot 1: CSMF and COD accuracy by training size for each model
for idx, model in enumerate(models):
    ax = axes[0, idx]
    model_df = training_df[training_df['model'] == model].sort_values('training_fraction')
    
    # Plot CSMF and COD
    ax.plot(model_df['training_fraction'] * 100, model_df['csmf_accuracy'], 
            'o-', label='CSMF Accuracy', linewidth=2, markersize=8, color=colors[model])
    ax.plot(model_df['training_fraction'] * 100, model_df['cod_accuracy'], 
            's--', label='COD Accuracy', linewidth=2, markersize=8, color=colors[model], alpha=0.7)
    
    # Fill between
    ax.fill_between(model_df['training_fraction'] * 100, 
                     model_df['csmf_accuracy'], 
                     model_df['cod_accuracy'],
                     alpha=0.2, color=colors[model])
    
    ax.set_xlabel('Training Data (%)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add annotation for gap at 5%
    if model == 'xgboost':
        small_data = model_df[model_df['training_fraction'] == 0.05].iloc[0]
        gap = small_data['csmf_accuracy'] - small_data['cod_accuracy']
        ax.annotate(f'Gap: {gap:.3f}', 
                    xy=(5, (small_data['csmf_accuracy'] + small_data['cod_accuracy'])/2),
                    xytext=(15, (small_data['csmf_accuracy'] + small_data['cod_accuracy'])/2),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')

# Plot 2: CSMF/COD Ratio comparison
ax = axes[1, 0]
for model in models:
    model_df = training_df[training_df['model'] == model].sort_values('training_fraction')
    ratio = model_df['csmf_accuracy'] / (model_df['cod_accuracy'] + 0.001)  # Avoid division by zero
    ax.plot(model_df['training_fraction'] * 100, ratio, 
            'o-', label=model.upper(), linewidth=2, markersize=8, color=colors[model])

ax.set_xlabel('Training Data (%)', fontsize=11)
ax.set_ylabel('CSMF/COD Ratio', fontsize=11)
ax.set_title('CSMF/COD Ratio by Training Size', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax.text(50, 1.1, 'Balanced Performance', fontsize=9, alpha=0.7)

# Plot 3: Performance at 5% training data
ax = axes[1, 1]
small_data = training_df[training_df['training_fraction'] == 0.05]
x = np.arange(len(models))
width = 0.35

csmf_vals = [small_data[small_data['model'] == m]['csmf_accuracy'].iloc[0] for m in models]
cod_vals = [small_data[small_data['model'] == m]['cod_accuracy'].iloc[0] for m in models]

bars1 = ax.bar(x - width/2, csmf_vals, width, label='CSMF Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, cod_vals, width, label='COD Accuracy', alpha=0.8)

# Color bars by model
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    bar1.set_color(colors[models[i]])
    bar2.set_color(colors[models[i]])
    bar2.set_alpha(0.6)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Performance at 5% Training Data (59 samples)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in models])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Improvement from 5% to 100%
ax = axes[1, 2]
improvements = []
for model in models:
    model_df = training_df[training_df['model'] == model]
    small = model_df[model_df['training_fraction'] == 0.05].iloc[0]
    large = model_df[model_df['training_fraction'] == 1.0].iloc[0]
    
    csmf_imp = (large['csmf_accuracy'] - small['csmf_accuracy']) / small['csmf_accuracy'] * 100
    cod_imp = (large['cod_accuracy'] - small['cod_accuracy']) / small['cod_accuracy'] * 100
    improvements.append((csmf_imp, cod_imp))

x = np.arange(len(models))
width = 0.35

csmf_imps = [imp[0] for imp in improvements]
cod_imps = [imp[1] for imp in improvements]

bars1 = ax.bar(x - width/2, csmf_imps, width, label='CSMF Improvement', alpha=0.8)
bars2 = ax.bar(x + width/2, cod_imps, width, label='COD Improvement', alpha=0.8)

# Color bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    bar1.set_color(colors[models[i]])
    bar2.set_color(colors[models[i]])
    bar2.set_alpha(0.6)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Improvement (%)', fontsize=11)
ax.set_title('Performance Improvement (5% → 100% Data)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in models])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'training_size_impact.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'training_size_impact.pdf', bbox_inches='tight')
print(f"Plots saved to {output_dir}")

# Create additional focused plot on the phenomenon
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('XGBoost CSMF vs COD Divergence with Small Training Data', fontsize=14, fontweight='bold')

# Left plot: All models CSMF vs COD at different training sizes
ax = axes2[0]
for model in models:
    model_df = training_df[training_df['model'] == model].sort_values('training_fraction')
    ax.scatter(model_df['cod_accuracy'], model_df['csmf_accuracy'], 
               s=model_df['training_fraction']*500, 
               alpha=0.6, label=model.upper(), color=colors[model])
    
    # Connect points
    ax.plot(model_df['cod_accuracy'], model_df['csmf_accuracy'], 
            '--', alpha=0.3, color=colors[model])

# Add diagonal line
ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='CSMF = COD')
ax.set_xlabel('COD Accuracy', fontsize=11)
ax.set_ylabel('CSMF Accuracy', fontsize=11)
ax.set_title('CSMF vs COD Accuracy Relationship', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.55])
ax.set_ylim([0.4, 1])

# Annotate XGBoost 5% point
xgb_small = training_df[(training_df['model'] == 'xgboost') & 
                        (training_df['training_fraction'] == 0.05)].iloc[0]
ax.annotate('XGBoost\n5% data', 
            xy=(xgb_small['cod_accuracy'], xgb_small['csmf_accuracy']),
            xytext=(xgb_small['cod_accuracy']-0.05, xgb_small['csmf_accuracy']+0.08),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

# Right plot: In-domain vs Out-domain CSMF/COD ratio
ax = axes2[1]
domain_data = []
for model in models:
    in_domain = df[(df['model'] == model) & (df['experiment_type'] == 'in_domain')]
    out_domain = df[(df['model'] == model) & (df['experiment_type'] == 'out_domain')]
    
    in_ratio = in_domain['csmf_accuracy'].mean() / (in_domain['cod_accuracy'].mean() + 0.001)
    out_ratio = out_domain['csmf_accuracy'].mean() / (out_domain['cod_accuracy'].mean() + 0.001)
    
    domain_data.append({
        'model': model.upper(),
        'In-Domain': in_ratio,
        'Out-Domain': out_ratio
    })

domain_df = pd.DataFrame(domain_data)
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, domain_df['In-Domain'], width, label='In-Domain', alpha=0.8)
bars2 = ax.bar(x + width/2, domain_df['Out-Domain'], width, label='Out-Domain', alpha=0.8)

for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    bar1.set_color(colors[models[i]])
    bar2.set_color(colors[models[i]])
    bar2.set_alpha(0.6)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('CSMF/COD Ratio', fontsize=11)
ax.set_title('Domain Transfer Impact on CSMF/COD Balance', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(domain_df['model'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'csmf_cod_divergence.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'csmf_cod_divergence.pdf', bbox_inches='tight')
print(f"Additional plots saved to {output_dir}")

# Print summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

print("\nPerformance at 5% Training Data:")
print("-"*40)
small_stats = training_df[training_df['training_fraction'] == 0.05]
for _, row in small_stats.iterrows():
    ratio = row['csmf_accuracy'] / (row['cod_accuracy'] + 0.001)
    print(f"{row['model'].upper():10} CSMF: {row['csmf_accuracy']:.3f}, "
          f"COD: {row['cod_accuracy']:.3f}, Ratio: {ratio:.2f}")

print("\nPerformance at 100% Training Data:")
print("-"*40)
full_stats = training_df[training_df['training_fraction'] == 1.0]
for _, row in full_stats.iterrows():
    ratio = row['csmf_accuracy'] / (row['cod_accuracy'] + 0.001)
    print(f"{row['model'].upper():10} CSMF: {row['csmf_accuracy']:.3f}, "
          f"COD: {row['cod_accuracy']:.3f}, Ratio: {ratio:.2f}")

print("\n✅ Visualization complete!")