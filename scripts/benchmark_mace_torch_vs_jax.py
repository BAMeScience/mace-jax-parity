#!/usr/bin/env python3
"""
Benchmark the inference throughput of a Torch MACE model and its JAX
counterpart on the mp-traj HDF5 datasets. The script runs both models over the
same batches, records timings, and reports graphs/sec together with the JAX
compile overhead. The dtype is enforced to be identical for both models, and
you can point the Torch and JAX runs at GPU devices for fair comparisons.

Example
-------
    python scripts/benchmark_mace_torch_vs_jax.py \\
        --torch-model models/mace_foundation.pt \\
        --jax-model models/mace_jax_bundle \\
        --torch-device cuda \\
        --jax-platform gpu \\
        --dtype float32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

# Torch 2.6 tightened torch.load defaults; the checkpoints still store ``slice``.
try:  # pragma: no cover - defensive guard for torch<2.6
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover
    add_safe_globals = None

if callable(add_safe_globals):  # pragma: no cover - guard for torch<2.6
    add_safe_globals([slice])

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.format_hdf5 import HDF5GraphDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Torch and JAX MACE models over mp-traj batches and report throughput."
        )
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
        default=Path("data/mptraj"),
        help="Directory with mp-traj HDF5 files (default: data/mptraj).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="valid",
        help='Which split to benchmark (default: valid). Use "all" to process every *.h5 file.',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the PyG DataLoader (default: 8).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Floating point precision to enforce for both models (default: float32).",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        help="Device for the Torch model/data (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--jax-platform",
        type=str,
        default="cpu",
        help="JAX platform to target (cpu/gpu).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally cap the number of batches processed for quick smoke tests.",
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


def _cast_batch(batch: Batch, dtype):
    for key in batch.keys():
        value = _get_batch_value(batch, key)
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            _set_batch_value(batch, key, value.to(dtype))
    return batch


def _batch_to_jax(batch: Batch, jax, jnp_dtype):
    data_dict = {}
    for key in batch.keys():
        value = _get_batch_value(batch, key)
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if array.dtype.kind == "f":
                array = array.astype(jnp_dtype)
            data_dict[key] = jax.device_put(array)
        else:
            data_dict[key] = value
    return data_dict


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


def _setup_jax(platform: str, enable_x64: bool):
    import jax

    jax.config.update("jax_platform_name", platform)
    jax.config.update("jax_enable_x64", enable_x64)
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


def _load_jax_bundle(bundle_path: Path, dtype: str, platform: str):
    jax, jnp = _setup_jax(platform, enable_x64=dtype == "float64")
    from equitrain.backends.jax_utils import load_model_bundle

    bundle = load_model_bundle(str(bundle_path), dtype=dtype)
    return jax, jnp, bundle


def _benchmark_torch(
    model, h5_files, z_table, r_max, batch_size, dtype, device, max_batches
):
    total_graphs = 0
    total_time = 0.0
    batches_seen = 0

    sync = (
        torch.cuda.synchronize
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )

    with torch.no_grad():
        for h5_path in h5_files:
            loader = _build_loader(h5_path, z_table, r_max, batch_size)
            for batch in tqdm(loader, desc=f"Torch {h5_path.name}", leave=False):
                if max_batches is not None and batches_seen >= max_batches:
                    return total_graphs, total_time, batches_seen

                batch = batch.to(device)
                batch = _cast_batch(batch, dtype)

                if sync:
                    sync()
                start = time.perf_counter()
                pred = _forward_torch(model, batch)
                energy = _extract_energy(pred)
                energy = energy.detach()
                if sync:
                    sync()
                step_time = time.perf_counter() - start

                total_graphs += energy.shape[0]
                total_time += step_time
                batches_seen += 1

    return total_graphs, total_time, batches_seen


def _benchmark_jax(
    jax, jnp, bundle, h5_files, z_table, r_max, batch_size, dtype, max_batches
):
    jnp_dtype = jnp.dtype(dtype)
    jit_apply = jax.jit(
        bundle.module.apply,
        static_argnames=("compute_force", "compute_stress"),
    )

    total_graphs = 0
    total_time = 0.0
    batches_seen = 0
    compile_time = None
    first_batch_graphs = None

    for h5_path in h5_files:
        loader = _build_loader(h5_path, z_table, r_max, batch_size)
        for batch in tqdm(loader, desc=f"JAX {h5_path.name}", leave=False):
            if max_batches is not None and batches_seen >= max_batches:
                return (
                    total_graphs,
                    total_time,
                    batches_seen,
                    compile_time,
                    first_batch_graphs,
                )

            batch_jax = _batch_to_jax(batch, jax, jnp_dtype)

            start = time.perf_counter()
            pred = jit_apply(
                bundle.params,
                batch_jax,
                compute_force=False,
                compute_stress=False,
            )
            energy = pred["energy"]
            if hasattr(energy, "block_until_ready"):
                energy.block_until_ready()
            step_time = time.perf_counter() - start

            if compile_time is None:
                compile_time = step_time
                first_batch_graphs = energy.shape[0]

            total_graphs += energy.shape[0]
            total_time += step_time
            batches_seen += 1

    return total_graphs, total_time, batches_seen, compile_time, first_batch_graphs


def main() -> None:
    args = _parse_args()

    torch_dtype = getattr(torch, args.dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    device = torch.device(args.torch_device)

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

    jax, jnp, bundle = _load_jax_bundle(
        args.jax_model, dtype=args.dtype, platform=args.jax_platform
    )

    atomic_numbers = _extract_atomic_numbers(torch_model, bundle)
    r_max = _extract_r_max(torch_model, bundle)

    z_table = AtomicNumberTable(atomic_numbers)

    h5_files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        h5_files = [p for p in h5_files if p.stem == args.split]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    torch_graphs, torch_time, torch_batches = _benchmark_torch(
        torch_model,
        h5_files,
        z_table,
        r_max,
        args.batch_size,
        torch_dtype,
        device,
        args.max_batches,
    )

    jax_graphs, jax_time, jax_batches, jax_compile, first_batch_graphs = _benchmark_jax(
        jax,
        jnp,
        bundle,
        h5_files,
        z_table,
        r_max,
        args.batch_size,
        args.dtype,
        args.max_batches,
    )

    def _throughput(graphs, elapsed):
        return graphs / elapsed if elapsed and graphs else 0.0

    steadystate_time = None
    steadystate_graphs = None
    if jax_compile is not None and jax_batches > 1:
        steadystate_time = jax_time - jax_compile
        steadystate_graphs = jax_graphs - (first_batch_graphs or 0)

    print(
        f"Torch [{args.dtype}] on {device}: "
        f"{torch_graphs} graphs across {torch_batches} batches "
        f"in {torch_time:.3f}s => {_throughput(torch_graphs, torch_time):.2f} graphs/s"
    )

    print(
        f"JAX [{args.dtype}] on {args.jax_platform}: "
        f"{jax_graphs} graphs across {jax_batches} batches "
        f"in {jax_time:.3f}s => {_throughput(jax_graphs, jax_time):.2f} graphs/s"
    )

    if jax_compile is not None:
        print(f"JAX compile+first-step time: {jax_compile:.3f}s")
    if steadystate_time is not None and steadystate_graphs is not None:
        print(
            f"JAX steady-state: {steadystate_graphs} graphs "
            f"in {steadystate_time:.3f}s => "
            f"{_throughput(steadystate_graphs, steadystate_time):.2f} graphs/s"
        )


if __name__ == "__main__":
    main()
