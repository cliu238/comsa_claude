#!/usr/bin/env python
"""
Analyze train/test split percentages by site from the cleaned results.
"""

import pandas as pd
from pathlib import Path


def analyze_splits(file_path: str, dataset_name: str):
    """Analyze train/test splits for each site."""
    print(f"\n=== {dataset_name.upper()} TRAIN/TEST SPLITS ===")
    
    df = pd.read_csv(file_path)
    
    # Get unique train/test combinations for in-domain experiments
    in_domain = df[(df['experiment_type'] == 'in_domain') & (df['train_site'] == df['test_site'])]
    
    splits_by_site = []
    
    for site in sorted(in_domain['train_site'].unique()):
        site_data = in_domain[in_domain['train_site'] == site]
        n_train = site_data['n_train'].iloc[0]  # Should be same for all experiments
        n_test = site_data['n_test'].iloc[0]
        total = n_train + n_test
        
        train_pct = (n_train / total) * 100
        test_pct = (n_test / total) * 100
        
        splits_by_site.append({
            'site': site,
            'n_train': n_train,
            'n_test': n_test,
            'total': total,
            'train_pct': train_pct,
            'test_pct': test_pct
        })
        
        print(f"{site:8} | Train: {n_train:4d} ({train_pct:5.1f}%) | Test: {n_test:3d} ({test_pct:4.1f}%) | Total: {total:4d}")
    
    # Calculate overall statistics
    total_train = sum(s['n_train'] for s in splits_by_site)
    total_test = sum(s['n_test'] for s in splits_by_site)
    grand_total = total_train + total_test
    
    print(f"{'OVERALL':8} | Train: {total_train:4d} ({(total_train/grand_total)*100:5.1f}%) | Test: {total_test:3d} ({(total_test/grand_total)*100:4.1f}%) | Total: {grand_total:4d}")
    
    # Check if splits are consistent
    train_percentages = [s['train_pct'] for s in splits_by_site]
    test_percentages = [s['test_pct'] for s in splits_by_site]
    
    if len(set([round(p, 1) for p in train_percentages])) == 1:
        print(f"\n✓ Consistent split across all sites: ~{train_percentages[0]:.0f}% train, ~{test_percentages[0]:.0f}% test")
    else:
        print(f"\n⚠ Variable splits detected:")
        for split in splits_by_site:
            print(f"  {split['site']}: {split['train_pct']:.1f}% train")
    
    return splits_by_site


def main():
    base_dir = Path("/Users/ericliu/projects5/context-engineering-intro/results")
    
    # Analyze both datasets
    datasets = [
        (base_dir / "cod5_multi_seed_5models_v2" / "cod5_comparison_results_clean.csv", "COD5"),
        (base_dir / "va34_multi_seed_5models_v2" / "va34_comparison_results_clean.csv", "VA34")
    ]
    
    all_splits = {}
    
    for file_path, name in datasets:
        if file_path.exists():
            splits = analyze_splits(str(file_path), name)
            all_splits[name] = splits
        else:
            print(f"File not found: {file_path}")
    
    # Check if splits are identical between datasets
    if len(all_splits) == 2:
        cod5_splits = {s['site']: (s['n_train'], s['n_test']) for s in all_splits['COD5']}
        va34_splits = {s['site']: (s['n_train'], s['n_test']) for s in all_splits['VA34']}
        
        if cod5_splits == va34_splits:
            print(f"\n✓ COD5 and VA34 use identical train/test splits")
        else:
            print(f"\n⚠ COD5 and VA34 have different splits")


if __name__ == "__main__":
    main()