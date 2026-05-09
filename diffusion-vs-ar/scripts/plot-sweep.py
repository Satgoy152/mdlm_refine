#!/usr/bin/env python3
"""Plot accuracy vs. diffusion steps from sweep_results.csv, one figure per task."""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="eval_results/sweep/sweep_results.csv",
        help="Path to sweep_results.csv produced by sweep-eval.sh.",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_results/sweep/plots",
        help="Directory to write per-task plots into.",
    )
    parser.add_argument(
        "--metric",
        default="predict_acc",
        choices=["predict_acc", "predict_loss"],
        help="Which metric to plot on the y-axis.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["diffusion_steps"] = pd.to_numeric(df["diffusion_steps"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["diffusion_steps", args.metric])

    if df.empty:
        print(f"No usable rows in {csv_path} (after dropping NA).")
        return

    # Distinguish refine n=1 vs n=8, proseco b=1/f=3 vs b=2/f=1, etc.
    df["config"] = df.apply(
        lambda r: f"{r['model']} / {r['sampling_method']}"
        + (f" [{r['extra']}]" if isinstance(r["extra"], str) and r["extra"] else ""),
        axis=1,
    )

    # A list of distinct markers to cycle through
    marker_list = ["o", "s", "^", "D", "v", "p", "*", "X", "<", ">", "h"]

    for task, sub in df.groupby("task"):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create an iterator that loops over the markers infinitely
        markers = itertools.cycle(marker_list)
        
        for cfg, cfg_sub in sub.groupby("config"):
            cfg_sub = cfg_sub.sort_values("diffusion_steps")
            ax.plot(
                cfg_sub["diffusion_steps"],
                cfg_sub[args.metric],
                marker=next(markers),
                label=cfg,
            )
        ax.set_xlabel("diffusion steps")
        ax.set_ylabel(args.metric)
        ax.set_title(f"{task}: {args.metric} vs. diffusion steps")
        ax.set_xscale("log")
        ax.set_xticks(sorted(sub["diffusion_steps"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        out_path = out_dir / f"{task}_{args.metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()