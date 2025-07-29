import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('/Users/ericliu/projects5/context-engineering-intro/results/full_comparison_20250729_115445/va34_comparison_results.csv')

# Basic statistics
print("=== Dataset Overview ===")
print(f"Total experiments: {len(df)}")
print(f"Unique models: {df['model'].unique()}")
print(f"Unique sites: {df['train_site'].unique() if 'train_site' in df.columns else df['test_site'].unique()}")
print(f"Experiment types: {df['experiment_type'].unique()}")

# Check for errors
error_count = df['error'].notna().sum()
print(f"\nExperiments with errors: {error_count} ({error_count/len(df)*100:.1f}%)")

# Performance by model
print("\n=== Performance by Model ===")
model_perf = df.groupby('model')[['csmf_accuracy', 'cod_accuracy']].agg(['mean', 'std', 'min', 'max'])
print(model_perf)

# Performance by experiment type
print("\n=== Performance by Experiment Type ===")
exp_type_perf = df.groupby('experiment_type')[['csmf_accuracy', 'cod_accuracy']].agg(['mean', 'std'])
print(exp_type_perf)

# In-domain vs Out-domain comparison
print("\n=== In-domain vs Out-domain Performance ===")
in_domain = df[df['experiment_type'] == 'in_domain']
out_domain = df[df['experiment_type'] == 'out_domain']

for model in df['model'].unique():
    in_perf = in_domain[in_domain['model'] == model][['csmf_accuracy', 'cod_accuracy']].mean()
    out_perf = out_domain[out_domain['model'] == model][['csmf_accuracy', 'cod_accuracy']].mean()
    print(f"\n{model}:")
    print(f"  In-domain - CSMF: {in_perf['csmf_accuracy']:.3f}, COD: {in_perf['cod_accuracy']:.3f}")
    print(f"  Out-domain - CSMF: {out_perf['csmf_accuracy']:.3f}, COD: {out_perf['cod_accuracy']:.3f}")
    print(f"  Drop - CSMF: {in_perf['csmf_accuracy'] - out_perf['csmf_accuracy']:.3f}, COD: {in_perf['cod_accuracy'] - out_perf['cod_accuracy']:.3f}")

# Site-specific analysis
print("\n=== Site-specific Performance (In-domain) ===")
for site in df['test_site'].unique():
    site_data = in_domain[in_domain['test_site'] == site]
    print(f"\n{site}:")
    for model in ['xgboost', 'categorical_nb', 'insilico']:
        model_data = site_data[site_data['model'] == model]
        if not model_data.empty:
            print(f"  {model}: CSMF={model_data['csmf_accuracy'].values[0]:.3f}, COD={model_data['cod_accuracy'].values[0]:.3f}")

# Extreme performance cases
print("\n=== Extreme Performance Cases ===")
print("\nWorst COD accuracy (<0.05):")
worst_cod = df[df['cod_accuracy'] < 0.05].sort_values('cod_accuracy')
for _, row in worst_cod.iterrows():
    print(f"  {row['model']} {row['train_site']}->{row['test_site']}: COD={row['cod_accuracy']:.3f}")

print("\nBest COD accuracy (>0.5):")
best_cod = df[df['cod_accuracy'] > 0.5].sort_values('cod_accuracy', ascending=False)
for _, row in best_cod.iterrows():
    print(f"  {row['model']} {row['train_site']}->{row['test_site']}: COD={row['cod_accuracy']:.3f}")

# Sample size analysis
print("\n=== Sample Size Analysis ===")
sample_sizes = df.groupby('test_site')[['n_test']].first()
print(sample_sizes.sort_values('n_test'))

# Cross-site transfer matrix for each model
print("\n=== Cross-site Transfer Performance ===")
for model in ['xgboost', 'categorical_nb', 'insilico']:
    print(f"\n{model.upper()} - COD Accuracy Matrix:")
    model_data = df[df['model'] == model]
    
    sites = sorted(df['test_site'].unique())
    matrix = pd.DataFrame(index=sites, columns=sites)
    
    for train_site in sites:
        for test_site in sites:
            exp_data = model_data[(model_data['train_site'] == train_site) & 
                                  (model_data['test_site'] == test_site)]
            if not exp_data.empty:
                matrix.loc[train_site, test_site] = exp_data['cod_accuracy'].values[0]
    
    print(matrix.to_string())

# Statistical analysis of failures
print("\n=== Root Cause Analysis ===")

# 1. Model-specific failure patterns
print("\n1. Model-specific patterns:")
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    low_perf = model_data[model_data['cod_accuracy'] < 0.1]
    print(f"\n{model}:")
    print(f"  Total experiments: {len(model_data)}")
    print(f"  Low performance (<0.1 COD): {len(low_perf)} ({len(low_perf)/len(model_data)*100:.1f}%)")
    if len(low_perf) > 0:
        print(f"  Common failure sites: {low_perf['test_site'].value_counts().head(3).to_dict()}")

# 2. Site-specific challenges
print("\n2. Site-specific challenges:")
site_perf = df.groupby('test_site')[['cod_accuracy']].agg(['mean', 'std', 'count'])
print(site_perf.sort_values(('cod_accuracy', 'mean')))

# 3. Transfer learning patterns
print("\n3. Transfer learning patterns:")
transfer_patterns = defaultdict(list)
for _, row in out_domain.iterrows():
    transfer_patterns[f"{row['train_site']}->{row['test_site']}"].append(row['cod_accuracy'])

worst_transfers = sorted([(k, np.mean(v)) for k, v in transfer_patterns.items()], 
                        key=lambda x: x[1])[:10]
print("\nWorst transfer pairs:")
for pair, avg_cod in worst_transfers:
    print(f"  {pair}: {avg_cod:.3f}")

# Save detailed analysis
print("\n=== Saving Detailed Analysis ===")
analysis_results = {
    'model_performance': model_perf,
    'experiment_type_performance': exp_type_perf,
    'site_performance': site_perf,
    'worst_cod_cases': worst_cod[['model', 'train_site', 'test_site', 'cod_accuracy']],
    'best_cod_cases': best_cod[['model', 'train_site', 'test_site', 'cod_accuracy']]
}

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model performance comparison
ax1 = axes[0, 0]
model_means = df.groupby('model')[['csmf_accuracy', 'cod_accuracy']].mean()
model_means.plot(kind='bar', ax=ax1)
ax1.set_title('Average Performance by Model')
ax1.set_ylabel('Accuracy')
ax1.legend(['CSMF Accuracy', 'COD Accuracy'])

# 2. In-domain vs Out-domain
ax2 = axes[0, 1]
exp_type_means = df.groupby(['experiment_type', 'model'])[['cod_accuracy']].mean().unstack()
exp_type_means.plot(kind='bar', ax=ax2)
ax2.set_title('COD Accuracy: In-domain vs Out-domain')
ax2.set_ylabel('COD Accuracy')

# 3. Site difficulty
ax3 = axes[1, 0]
site_means = df.groupby('test_site')[['cod_accuracy']].mean().sort_values('cod_accuracy')
site_means.plot(kind='bar', ax=ax3)
ax3.set_title('Average COD Accuracy by Test Site')
ax3.set_ylabel('COD Accuracy')

# 4. Sample size vs performance
ax4 = axes[1, 1]
site_data = df.groupby('test_site').agg({'n_test': 'first', 'cod_accuracy': 'mean'})
ax4.scatter(site_data['n_test'], site_data['cod_accuracy'])
for site, row in site_data.iterrows():
    ax4.annotate(site, (row['n_test'], row['cod_accuracy']))
ax4.set_xlabel('Test Set Size')
ax4.set_ylabel('Average COD Accuracy')
ax4.set_title('Sample Size vs Performance')

plt.tight_layout()
plt.savefig('/Users/ericliu/projects5/context-engineering-intro/va_analysis_plots.png', dpi=300)
plt.close()

print("\nAnalysis complete. Plots saved to va_analysis_plots.png")