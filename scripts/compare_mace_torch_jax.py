#!/usr/bin/env python3
"""
Compare energy predictions from a Torch MACE model and its JAX counterpart on
the HDF5 datasets stored under ``resources/data/mptraj/mptraj``. The script
iterates over every batch, forwards it through both models, and reports any
energy discrepancy above the configured tolerance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

try:  # pragma: no cover - torch<2.6 compatibility
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover
    add_safe_globals = None

if callable(add_safe_globals):  # pragma: no cover - guard for torch<2.6
    safe_globals = [slice]
    try:
        from mace.modules.models import ScaleShiftMACE
    except ImportError:
        ScaleShiftMACE = None  # type: ignore[assignment]
    if ScaleShiftMACE is not None:
        safe_globals.append(ScaleShiftMACE)
    add_safe_globals(safe_globals)

from equitrain import get_args_parser_predict
from equitrain.backends.jax_predict import predict as jax_predict
from equitrain.backends.torch_model import get_model as get_torch_model
from equitrain.backends.torch_predict import _predict as torch_predict_impl
from equitrain.backends.torch_utils import set_dtype as torch_set_dtype
from equitrain.data.backend_torch.loaders import get_dataloader as get_torch_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mp-traj batches through Torch and JAX MACE models and compare energies."
    )
    parser.add_argument(
        "--torch-model",
        type=Path,
        required=True,
        help="Path to the serialized Torch MACE model (.pt/.model).",
    )
    parser.add_argument(
        "--jax-model",
        type=Path,
        required=True,
        help="Directory containing the MACE-JAX bundle (config.json + params.msgpack).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("resources/data/mptraj/mptraj"),
        help="Directory with mp-traj HDF5 files (default: resources/data/mptraj/mptraj).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="train",
        help='Which split to evaluate (default: train). Use "all" to process every *.h5 file.',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size fed into the equitrain predict routines (default: 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers for both Torch and JAX (default: 0).",
    )
    parser.add_argument(
        "--max-nodes-per-batch",
        type=int,
        default=200000,
        help="JAX/Torch greedy packer node cap per macro-batch.",
    )
    parser.add_argument(
        "--max-edges-per-batch",
        type=int,
        default=480000,
        help="JAX/Torch greedy packer edge cap per macro-batch.",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=1e-5,
        help="Relative tolerance (unitless) before reporting a discrepancy (default: 1e-5).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Floating point precision for both Torch and JAX models (default: float64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model execution (Torch and JAX). Examples: cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--diff-csv",
        type=Path,
        default=Path("energy_diffs.csv"),
        help="Write per-graph energy differences to the given CSV path (default: energy_diffs.csv).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-graph warnings when discrepancies exceed the tolerance.",
    )
    return parser.parse_args()


def _make_predict_args(base_args, backend: str, model_path: Path, predict_file: Path):
    predict_args = get_args_parser_predict().parse_args([])
    predict_args.backend = backend
    predict_args.model = str(model_path)
    predict_args.predict_file = str(predict_file)
    predict_args.batch_size = base_args.batch_size
    predict_args.dtype = base_args.dtype
    predict_args.energy_weight = 1.0
    predict_args.forces_weight = 0.0
    predict_args.stress_weight = 0.0
    if backend == "torch":
        predict_args.shuffle = False
    predict_args.model_wrapper = getattr(base_args, "model_wrapper", None) or "mace"
    if not hasattr(predict_args, "pin_memory"):
        predict_args.pin_memory = False
    predict_args.num_workers = int(getattr(base_args, "num_workers", 0))
    if not hasattr(predict_args, "batch_max_nodes"):
        predict_args.batch_max_nodes = None
    if not hasattr(predict_args, "batch_max_edges"):
        predict_args.batch_max_edges = None
    if not hasattr(predict_args, "batch_drop"):
        predict_args.batch_drop = False
    if not hasattr(predict_args, "niggli_reduce"):
        predict_args.niggli_reduce = False
    predict_args.batch_max_nodes = getattr(base_args, "max_nodes_per_batch", None)
    predict_args.batch_max_edges = getattr(base_args, "max_edges_per_batch", None)
    return predict_args


def _predict_torch(args, predict_file: Path):
    predict_args = _make_predict_args(
        args, backend="torch", model_path=args.torch_model, predict_file=predict_file
    )
    predict_args.tqdm = True
    device = torch.device(args.device)
    if device.type == "cpu":
        energy, _, _ = torch_predict_impl(predict_args, device=device)
        return energy.detach().cpu().numpy()
    return _predict_torch_gpu(predict_args, predict_file, device)


def _predict_jax(args, predict_file: Path):
    predict_args = _make_predict_args(
        args, backend="jax", model_path=args.jax_model, predict_file=predict_file
    )
    predict_args.jax_platform = _device_to_jax_platform(args.device)
    predict_args.tqdm = True
    predict_args.tqdm_desc = f"Predict jax ({predict_file.name})"
    energy, _, _ = jax_predict(predict_args)
    return np.asarray(energy)


def _device_to_jax_platform(device_spec: str) -> str:
    spec = (device_spec or "cpu").strip().lower()
    if spec.startswith("cpu"):
        return "cpu"
    if spec.startswith("cuda") or spec.startswith("gpu"):
        return "gpu"
    if spec.startswith("tpu"):
        return "tpu"
    raise ValueError(f"Unsupported device specification for JAX: {device_spec}")


def _predict_torch_gpu(predict_args, predict_file: Path, device: torch.device):
    torch_set_dtype(predict_args.dtype)
    model = get_torch_model(predict_args)
    model = model.to(device=device, dtype=getattr(torch, predict_args.dtype))
    model.eval()

    atomic_numbers = getattr(model.atomic_numbers, "zs", model.atomic_numbers)
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.detach().cpu().tolist()
    elif hasattr(atomic_numbers, "zs"):
        atomic_numbers = list(atomic_numbers.zs)

    r_max = model.r_max
    if isinstance(r_max, torch.Tensor):
        r_max = float(r_max.detach().cpu().item())
    else:
        r_max = float(r_max)

    loader = get_torch_loader(
        predict_args,
        predict_file,
        atomic_numbers,
        r_max,
        accelerator=None,
    )
    if loader is None:
        return np.zeros((0,), dtype=np.float32)

    energies = []
    iterator = loader
    if getattr(predict_args, "tqdm", False):
        iterator = tqdm(
            loader,
            desc=f"Predict torch ({predict_file.name})",
            unit="batch",
        )

    with torch.no_grad():
        for data_list in iterator:
            for data in data_list:
                if hasattr(data, "to"):
                    data = data.to(device)
                y_pred = model(data)
                energy = y_pred["energy"]
                if not isinstance(energy, torch.Tensor):
                    energy = torch.as_tensor(energy)
                energies.append(energy.detach().cpu())

    if not energies:
        return np.zeros((0,), dtype=np.float32)
    stacked = torch.cat(energies, dim=0)
    return stacked.numpy()


def main() -> None:
    args = _parse_args()

    if getattr(torch, args.dtype, None) is None:
        raise ValueError(f"Unsupported Torch dtype: {args.dtype}")
    h5_files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        h5_files = [p for p in h5_files if p.stem == args.split]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    total_graphs = 0
    max_rel_diff = 0.0
    max_abs_diff = 0.0
    flagged = False
    diff_count = 0
    diff_file = None
    diff_writer = None
    if args.diff_csv is not None:
        diff_path = args.diff_csv
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_file = diff_path.open("w", newline="")
        diff_writer = csv.writer(diff_file)
        diff_writer.writerow(
            [
                "file",
                "graph_index",
                "batch_id",
                "delta_e",
                "rel_delta",
                "torch_energy",
                "jax_energy",
            ]
        )

    try:
        for h5_path in h5_files:
            with torch.no_grad():
                energy_torch = _predict_torch(args, h5_path)
            energy_jax = _predict_jax(args, h5_path)

            if energy_torch.shape != energy_jax.shape:
                raise RuntimeError(
                    f"Energy shape mismatch for {h5_path.name}: "
                    f"Torch {energy_torch.shape} vs JAX {energy_jax.shape}"
                )

            total_graphs += energy_torch.shape[0]
            diff = np.abs(energy_torch - energy_jax)
            scale = np.maximum(
                np.maximum(np.abs(energy_torch), np.abs(energy_jax)), 1e-12
            )
            rel_diff = diff / scale
            batch_max_rel = float(rel_diff.max(initial=0.0))
            batch_max_abs = float(diff.max(initial=0.0))
            max_rel_diff = max(max_rel_diff, batch_max_rel)
            max_abs_diff = max(max_abs_diff, batch_max_abs)

            for idx, (delta, rel_delta, scale_val, e_t, e_j) in enumerate(
                zip(diff, rel_diff, scale, energy_torch, energy_jax)
            ):
                if rel_delta > args.energy_tol:
                    flagged = True
                    if args.verbose:
                        tqdm.write(
                            f"[WARN] {h5_path.name} graph #{idx}: "
                            f"|ΔE|/scale={rel_delta:.3e} exceeds tol {args.energy_tol:.1e} "
                            f"(abs={delta:.3e} eV, scale={scale_val:.3e}, torch={e_t:.6f}, jax={e_j:.6f})"
                        )
                if diff_writer is not None:
                    diff_writer.writerow(
                        [
                            h5_path.name,
                            int(idx),
                            "",
                            float(delta),
                            float(rel_delta),
                            float(e_t),
                            float(e_j),
                        ]
                    )
                    diff_file.flush()
                    diff_count += 1
    finally:
        if diff_file is not None:
            diff_file.close()

    print(
        f"Compared {total_graphs} graphs across {len(h5_files)} files. "
        f"Max relative |ΔE|/scale={max_rel_diff:.3e}, max absolute |ΔE|={max_abs_diff:.3e} eV"
    )
    if not flagged:
        print(
            f"✅ No discrepancies with relative error above {args.energy_tol:.1e} detected between Torch and JAX predictions."
        )
    else:
        print(
            f"⚠️ Relative energy discrepancies exceeded {args.energy_tol:.1e}. "
            "See warnings above for affected graphs."
        )
    if diff_writer is not None and args.diff_csv is not None:
        print(f"Wrote {diff_count} discrepancy rows to {args.diff_csv}")


if __name__ == "__main__":
    main()
