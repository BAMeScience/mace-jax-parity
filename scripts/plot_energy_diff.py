#!/usr/bin/env python3
"""
Quick helper to visualise energy differences recorded by compare_mace_torch_jax.py.

Usage
-----
    python scripts/plot_energy_diff.py --csv energy_diffs.csv --out energy_diff.png

The input CSV is expected to have the header:
    file,graph_index,batch_id,delta_e,torch_energy,jax_energy
as written by the compare script. Two plots are produced in one figure:
  1) Histogram of |ΔE| values.
  2) Scatter of Torch vs JAX energies with a y=x reference line.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot energy differences from CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to energy_diffs.csv created by compare_mace_torch_jax.py.",
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
        help="Number of bins for the ΔE histogram (default: 100).",
    )
    return parser.parse_args()


def _load(csv_path: Path):
    delta = []
    torch_e = []
    jax_e = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            delta.append(float(row["delta_e"]))
            torch_e.append(float(row["torch_energy"]))
            jax_e.append(float(row["jax_energy"]))
    return (
        np.asarray(delta, dtype=np.float64),
        np.asarray(torch_e, dtype=np.float64),
        np.asarray(jax_e, dtype=np.float64),
    )


def main() -> None:
    args = _parse_args()
    delta, torch_e, jax_e = _load(args.csv)

    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of absolute differences.
    ax_hist.hist(np.abs(delta), bins=args.bins, color="steelblue", alpha=0.8)
    ax_hist.set_xlabel("|ΔE| [eV]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Energy difference distribution")

    # Scatter Torch vs JAX with y=x line.
    min_e = min(torch_e.min(), jax_e.min())
    max_e = max(torch_e.max(), jax_e.max())
    ax_scatter.plot([min_e, max_e], [min_e, max_e], "k--", linewidth=1, label="y=x")
    ax_scatter.scatter(torch_e, jax_e, s=6, alpha=0.6, color="darkorange")
    ax_scatter.set_xlabel("Torch energy [eV]")
    ax_scatter.set_ylabel("JAX energy [eV]")
    ax_scatter.set_title("Torch vs JAX energies")
    ax_scatter.legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
