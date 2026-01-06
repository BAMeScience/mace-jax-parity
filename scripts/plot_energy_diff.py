#!/usr/bin/env python3
"""
Quick helper to visualise energy differences recorded by compare_mace_torch_jax.py.

Usage
-----
    python scripts/plot_energy_diff.py \
        --cpu-csv results/compare_cpu.csv \
        --gpu-csv results/compare_gpu.csv \
        --out results/energy_diff.png

    python scripts/plot_energy_diff.py \
        --cpu-f32 results/compare_cpu_f32.csv \
        --cpu-f64 results/compare_cpu_f64.csv \
        --gpu-f32 results/compare_gpu_f32.csv \
        --gpu-f64 results/compare_gpu_f64.csv \
        --gpu-nocueq-f32 results/compare_gpu_f32_nocueq.csv \
        --gpu-nocueq-f64 results/compare_gpu_f64_nocueq.csv \
        --out results/energy_diff_grid.png

Both CSVs are expected to follow the schema written by compare_mace_torch_jax.py:
    file,graph_index,batch_id,delta_e,rel_delta,torch_energy,jax_energy
The script can create a two-panel CPU/GPU plot or a 2x3 grid (float32/float64 by
CPU/GPU/GPU nocueq) of the relative |ΔE|/scale values for quick visual comparison.
"""

from __future__ import annotations

import argparse
from csv import DictReader
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CPU/GPU energy difference histograms."
    )
    parser.add_argument(
        "--cpu-csv",
        type=Path,
        default=None,
        help="CSV with CPU comparison results (compare_mace_torch_jax.py --device cpu).",
    )
    parser.add_argument(
        "--gpu-csv",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional foundation model identifier to annotate the figure title.",
    )
    parser.add_argument(
        "--cpu-f32",
        type=Path,
        default=None,
        help="CSV with CPU float32 comparison results.",
    )
    parser.add_argument(
        "--cpu-f64",
        type=Path,
        default=None,
        help="CSV with CPU float64 comparison results.",
    )
    parser.add_argument(
        "--gpu-f32",
        type=Path,
        default=None,
        help="CSV with GPU float32 comparison results.",
    )
    parser.add_argument(
        "--gpu-f64",
        type=Path,
        default=None,
        help="CSV with GPU float64 comparison results.",
    )
    parser.add_argument(
        "--gpu-nocueq-f32",
        type=Path,
        default=None,
        help="CSV with GPU float32 comparison results (no cueq).",
    )
    parser.add_argument(
        "--gpu-nocueq-f64",
        type=Path,
        default=None,
        help="CSV with GPU float64 comparison results (no cueq).",
    )
    return parser.parse_args()


def _load_rel_diff(csv_path: Path | None) -> np.ndarray:
    if csv_path is None or not csv_path.exists():
        return np.asarray([], dtype=np.float64)
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


def _plot_hist(ax: plt.Axes, delta_data: np.ndarray, bins: int, color: str) -> None:
    positive = delta_data[delta_data > 0]
    if positive.size > 0:
        eps = 1e-18
        min_edge = max(positive.min(), eps)
        max_edge = max(positive.max(), min_edge * 10)
        edges = np.logspace(np.log10(min_edge), np.log10(max_edge), bins)
        ax.hist(positive, bins=edges, color=color, alpha=0.8)
        ax.set_xscale("log")
    ax.grid(alpha=0.2, linestyle="--")


def _has_grid_inputs(args: argparse.Namespace) -> bool:
    return any(
        path is not None
        for path in (
            args.cpu_f32,
            args.cpu_f64,
            args.gpu_f32,
            args.gpu_f64,
            args.gpu_nocueq_f32,
            args.gpu_nocueq_f64,
        )
    )


def main() -> None:
    args = _parse_args()
    if _has_grid_inputs(args):
        grid_specs = [
            ("CPU", args.cpu_f32, args.cpu_f64, "steelblue"),
            ("GPU nocueq", args.gpu_nocueq_f32, args.gpu_nocueq_f64, "seagreen"),
            ("GPU", args.gpu_f32, args.gpu_f64, "darkorange"),
        ]
        grid_data = [
            (
                label,
                _load_rel_diff(path_f32),
                _load_rel_diff(path_f64),
                color,
            )
            for label, path_f32, path_f64, color in grid_specs
        ]
        if all(
            data_f32.size == 0 and data_f64.size == 0
            for _, data_f32, data_f64, _ in grid_data
        ):
            raise ValueError("No CSV data provided for plotting.")

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
        row_titles = ["float32", "float64"]
        col_titles = [label for label, _, _, _ in grid_data]
        for col_idx, (label, data_f32, data_f64, color) in enumerate(grid_data):
            for row_idx, data in enumerate((data_f32, data_f64)):
                ax = axes[row_idx][col_idx]
                _plot_hist(ax, data, args.bins, color)
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(f"{row_titles[row_idx]}\nCount")
                else:
                    ax.set_ylabel("Count")
                ax.set_xlabel("|ΔE|/scale (unitless)")

        model_note = f"{args.model_name}, " if args.model_name else ""
        fig.suptitle(
            f"Relative energy differences ({model_note}grid)",
            fontsize=14,
        )
    else:
        cpu_delta = _load_rel_diff(args.cpu_csv)
        gpu_delta = _load_rel_diff(args.gpu_csv)
        if cpu_delta.size == 0 and gpu_delta.size == 0:
            raise ValueError("No CSV data provided for plotting.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        hist_specs = [
            (cpu_delta, "CPU comparison", axes[0], "steelblue"),
            (gpu_delta, "GPU comparison", axes[1], "darkorange"),
        ]
        for delta_data, title, ax, color in hist_specs:
            _plot_hist(ax, delta_data, args.bins, color)
            ax.set_xlabel("|ΔE|/scale (unitless)")
            ax.set_ylabel("Count")
            ax.set_title(title)

        model_note = f"{args.model_name}, " if args.model_name else ""
        fig.suptitle(
            f"Relative energy differences ({model_note}dtype={args.dtype})",
            fontsize=14,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
