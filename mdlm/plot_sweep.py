#!/usr/bin/env python3
"""Plot sweep results from sweep_cr_fr.csv.

For each metric, produces a line plot where:
  - X-axis: fill_ratio
  - Each line: a different correction_ratio
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser(description='Plot sweep results')
    parser.add_argument('--csv', type=str, default='./eval_results/sweep_cr_fr.csv',
                        help='Path to sweep CSV file')
    parser.add_argument('--out_dir', type=str, default='./eval_results/plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = {
        'gen_ppl': {'label': 'Generative Perplexity', 'lower_better': True},
        'entropy': {'label': 'Shannon Entropy', 'lower_better': False},
        'rep_n':   {'label': 'N-gram Repetition', 'lower_better': True},
    }

    correction_ratios = sorted(df['correction_ratio'].unique())

    for metric, info in metrics.items():
        if metric not in df.columns or df[metric].dropna().empty:
            print(f'Skipping {metric} (no data)')
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for cr in correction_ratios:
            subset = df[df['correction_ratio'] == cr].sort_values('fill_ratio')
            ax.plot(subset['fill_ratio'], subset[metric],
                    marker='o', label=f'CR={cr}')

        ax.set_xlabel('Fill Ratio')
        ax.set_ylabel(info['label'])
        ax.set_title(f'{info["label"]} vs Fill Ratio')
        ax.legend(title='Correction Ratio')
        ax.grid(True, alpha=0.3)

        path = os.path.join(args.out_dir, f'sweep_{metric}.png')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
