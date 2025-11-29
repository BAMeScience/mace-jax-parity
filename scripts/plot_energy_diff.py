#!/usr/bin/env python3
"""
Quick helper to visualise energy differences recorded by compare_mace_torch_jax.py.

Usage
-----
    python scripts/plot_energy_diff.py \
        --cpu-csv results/compare_cpu.csv \
        --gpu-csv results/compare_gpu.csv \
        --out results/energy_diff.png

Both CSVs are expected to follow the schema written by compare_mace_torch_jax.py:
    file,graph_index,batch_id,delta_e,rel_delta,torch_energy,jax_energy
The script creates two histograms (CPU + GPU) of the relative |ΔE|/scale values for quick
visual comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from csv import DictReader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CPU/GPU energy difference histograms.")
    parser.add_argument(
        "--cpu-csv",
        type=Path,
        required=True,
        help="CSV with CPU comparison results (compare_mace_torch_jax.py --device cpu).",
    )
    parser.add_argument(
        "--gpu-csv",
        type=Path,
        required=True,
        help="CSV with GPU comparison results (compare_mace_torch_jax.py --device cuda).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("energy_diff.png"),
        help="Output image path (default: energy_diff.png).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for the histogram (default: 100).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Annotate the plot with the dtype used for comparison (default: float32).",
    )
    return parser.parse_args()


def _load_rel_diff(csv_path: Path) -> np.ndarray:
    values = []
    with csv_path.open() as handle:
        reader = DictReader(handle)
        for row in reader:
            try:
                rel = float(row.get("rel_delta", 0.0))
            except (TypeError, ValueError):
                continue
            values.append(abs(rel))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = _parse_args()
    cpu_delta = _load_rel_diff(args.cpu_csv)
    gpu_delta = _load_rel_diff(args.gpu_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    hist_specs = [
        (cpu_delta, "CPU comparison", axes[0], "steelblue"),
        (gpu_delta, "GPU comparison", axes[1], "darkorange"),
    ]

    eps = 1e-18
    for delta_data, title, ax, color in hist_specs:
        positive = delta_data[delta_data > 0]
        if positive.size == 0:
            positive = np.array([eps])
        min_edge = max(positive.min(), eps)
        max_edge = max(positive.max(), min_edge * 10)
        bins = np.logspace(np.log10(min_edge), np.log10(max_edge), args.bins)
        ax.hist(positive, bins=bins, color=color, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("|ΔE|/scale (unitless)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(alpha=0.2, linestyle="--")

    fig.suptitle(f"Relative energy differences (dtype={args.dtype})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
