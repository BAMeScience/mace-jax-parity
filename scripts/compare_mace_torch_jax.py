#!/usr/bin/env python3
"""
Compare energy predictions from a Torch MACE model and its JAX counterpart on
the HDF5 datasets stored under ``resources/data/mptraj/mptraj``. The script
iterates over every batch, forwards it through both models, and reports any
energy discrepancy above the configured tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

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

from equitrain.backends.jax_utils import load_model_bundle
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.format_hdf5 import HDF5GraphDataset


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
        help="Batch size for the PyG DataLoader (default: 4).",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=1e-5,
        help="Absolute tolerance (in eV) before reporting a discrepancy (default: 1e-5).",
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
        help="Device for the Torch model/data (default: cpu).",
    )
    return parser.parse_args()


def _get_batch_value(batch: Batch, key: str):
    try:
        return batch[key]
    except (AttributeError, KeyError, TypeError):
        return getattr(batch, key)


def _set_batch_value(batch: Batch, key: str, value):
    try:
        batch[key] = value
    except (AttributeError, KeyError, TypeError):
        setattr(batch, key, value)


def _batch_to_jax(batch: Batch):
    data_dict = {}
    for key in batch.keys():
        value = _get_batch_value(batch, key)
        if isinstance(value, torch.Tensor):
            data_dict[key] = jax.device_put(value.detach().cpu().numpy())
        else:
            data_dict[key] = value
    return data_dict


def _cast_batch(batch, dtype):
    for key in batch.keys():
        value = _get_batch_value(batch, key)
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            _set_batch_value(batch, key, value.to(dtype))
    return batch


def _extract_atomic_numbers(torch_model, jax_bundle):
    if hasattr(torch_model, "atomic_numbers"):
        numbers = getattr(torch_model, "atomic_numbers")
        if hasattr(numbers, "zs"):  # AtomicNumberTable
            return [int(z) for z in numbers.zs]
        if isinstance(numbers, torch.Tensor):
            return [int(z) for z in numbers.detach().cpu().tolist()]
        if isinstance(numbers, (list, tuple)):
            return [int(z) for z in numbers]
    config_numbers = jax_bundle.config.get("atomic_numbers")
    if config_numbers is not None:
        return [int(z) for z in config_numbers]
    raise RuntimeError("Unable to infer atomic numbers from the provided models.")


def _extract_r_max(torch_model, jax_bundle):
    value = getattr(torch_model, "r_max", None)
    if value is not None:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    config_value = jax_bundle.config.get("r_max")
    if config_value is not None:
        return float(config_value)
    raise RuntimeError("Unable to infer r_max from the provided models.")


def _extract_energy(pred):
    if isinstance(pred, dict):
        if "energy" not in pred:
            raise RuntimeError("Torch model output dict lacks `energy` key.")
        return pred["energy"]
    if isinstance(pred, (list, tuple)):
        return pred[0]
    return pred


def _graph_indices(batch, size):
    if hasattr(batch, "idx"):
        idx = getattr(batch, "idx")
        if isinstance(idx, torch.Tensor):
            return [int(i) for i in idx.detach().cpu().tolist()]
    return list(range(size))


def _build_loader(
    h5_path: Path, z_table: AtomicNumberTable, r_max: float, batch_size: int
):
    dataset = HDF5GraphDataset(
        h5_path,
        r_max=r_max,
        atomic_numbers=z_table,
    )
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def _forward_torch(model, batch):
    try:
        return model(batch, compute_force=False, compute_stress=False)
    except TypeError:
        return model(batch)


def main() -> None:
    args = _parse_args()

    torch_dtype = getattr(torch, args.dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported Torch dtype: {args.dtype}")
    device = torch.device(args.device)

    torch_model = torch.load(
        args.torch_model,
        map_location=device,
        weights_only=False,
    )
    torch_model = torch_model.to(device=device)
    torch_model = torch_model.to(dtype=torch_dtype)
    if hasattr(torch_model, "atomic_energies_fn"):
        energies = getattr(torch_model.atomic_energies_fn, "atomic_energies", None)
        if isinstance(energies, torch.Tensor):
            torch_model.atomic_energies_fn.atomic_energies = energies.to(torch_dtype)
    torch_model = torch_model.eval()

    bundle = load_model_bundle(str(args.jax_model), dtype=args.dtype)

    atomic_numbers = _extract_atomic_numbers(torch_model, bundle)
    r_max = _extract_r_max(torch_model, bundle)

    z_table = AtomicNumberTable(atomic_numbers)

    h5_files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        h5_files = [p for p in h5_files if p.stem == args.split]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    total_graphs = 0
    max_diff = 0.0
    flagged = False

    with torch.no_grad():
        for h5_path in h5_files:
            loader = _build_loader(h5_path, z_table, r_max, args.batch_size)
            for batch_id, batch in enumerate(
                tqdm(loader, desc=f'compare {h5_path.name}', leave=False)
            ):
                batch = batch.to(device)
                batch = _cast_batch(batch, torch_dtype)

                torch_pred = _extract_energy(_forward_torch(torch_model, batch))
                energy_torch = torch_pred.detach().cpu().numpy()

                batch_jax = _batch_to_jax(batch)
                jax_pred = bundle.module.apply(
                    bundle.params,
                    batch_jax,
                    compute_force=False,
                    compute_stress=False,
                )
                energy_jax = np.asarray(jax_pred["energy"])

                if energy_torch.shape != energy_jax.shape:
                    raise RuntimeError(
                        f"Energy shape mismatch: Torch {energy_torch.shape} vs JAX {energy_jax.shape}"
                    )

                total_graphs += energy_torch.shape[0]
                diff = np.abs(energy_torch - energy_jax)
                batch_max = float(diff.max(initial=0.0))
                max_diff = max(max_diff, batch_max)

                if batch_max > args.energy_tol:
                    flagged = True
                    indices = _graph_indices(batch, len(diff))
                    for idx, delta, e_t, e_j in zip(
                        indices, diff, energy_torch, energy_jax
                    ):
                        if delta > args.energy_tol:
                            tqdm.write(
                                f"[WARN] {h5_path.name} graph #{idx} batch {batch_id}: "
                                f"|ΔE|={delta:.3e} eV exceeds tol {args.energy_tol:.1e} "
                                f"(torch={e_t:.6f}, jax={e_j:.6f})"
                            )

    print(
        f"Compared {total_graphs} graphs across {len(h5_files)} files. Max |ΔE|={max_diff:.3e} eV"
    )
    if not flagged:
        print(
            f"✅ No discrepancies larger than {args.energy_tol:.1e} eV detected between Torch and JAX predictions."
        )
    else:
        print(
            f"⚠️ Energy discrepancies exceeded {args.energy_tol:.1e} eV. "
            "See warnings above for affected graphs."
        )


if __name__ == "__main__":
    main()
