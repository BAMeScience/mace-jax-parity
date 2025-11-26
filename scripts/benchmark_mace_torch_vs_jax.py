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
        --device gpu \\
        --dtype float32
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import torch

# Torch 2.6 tightened torch.load defaults; the checkpoints still store ``slice``.
from torch.serialization import add_safe_globals
from tqdm import tqdm

from equitrain.backends.jax_utils import (
    load_model_bundle,
    prepare_sharded_batch,
    replicate_to_local_devices,
)
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader as get_dataloader_jax
from equitrain.data.backend_jax.atoms_to_graphs import graph_to_data
from equitrain.data.backend_torch import get_dataloader as get_dataloader_torch

add_safe_globals([slice])


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
        default=18,
        help="Batch size for the PyG DataLoader (default: 8).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Floating point precision to enforce for both models (default: float32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for both Torch and JAX (cpu/gpu/cuda/cuda:0). "
            "Values starting with 'gpu' are treated as CUDA."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally cap the number of batches processed for quick smoke tests.",
    )
    parser.add_argument(
        "--max-edges-per-batch",
        type=int,
        default=10000,
        help="Hard cap on total edges per JAX batch (greedy packing).",
    )
    parser.add_argument(
        "--max-nodes-per-batch",
        type=int,
        default=None,
        help="Optional hard cap on total nodes per JAX batch (greedy packing).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=0,
        help="If >0, spawn a CPU worker to prebuild padded JAX batches and queue them "
        "for the main process. Value sets queue maxsize.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available JAX GPUs via pmap (requires batch divisible by device count).",
    )
    return parser.parse_args()


def _resolve_devices(spec: str) -> tuple[torch.device, str]:
    """Interpret the CLI device flag for Torch and infer the JAX platform."""
    text = spec.strip()
    lower = text.lower()
    if lower.startswith("gpu"):
        torch_spec = "cuda" + text[3:]
    else:
        torch_spec = text
    device = torch.device(torch_spec)
    platform = "gpu" if device.type == "cuda" else "cpu"
    return device, platform


def _extract_atomic_numbers(torch_model, jax_bundle):
    """
    Resolve the list of atomic numbers to build the AtomicNumberTable used for
    graph construction and shared by both Torch and JAX runs. We try the Torch
    checkpoint first (it may store an AtomicNumberTable, tensor, or plain list),
    then fall back to the JAX bundle config. Failing both, we error because the
    models would be misconfigured.
    """
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


def _graph_to_jax_data(graph, num_species: int, jnp_dtype):
    data = graph_to_data(graph, num_species=num_species)
    casted = {}
    for k, v in data.items():
        if hasattr(v, "dtype") and v.dtype.kind == "f":
            casted[k] = v.astype(jnp_dtype)
        else:
            casted[k] = v
    data = jax.device_put(casted)
    return data


def _forward_torch(model, batch):
    return model(batch, compute_force=False, compute_stress=False)


def _setup_jax(platform: str, enable_x64: bool):
    jax.config.update("jax_platform_name", platform)
    jax.config.update("jax_enable_x64", enable_x64)
    # Favor speed (allow TF32/fast matmul on GPU).
    try:
        jax.config.update("jax_default_matmul_precision", "fastest")
    except Exception:
        pass
    return None


def _load_jax_bundle(bundle_path: Path, dtype: str, platform: str):
    cache_dir = Path(bundle_path) / ".cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    print(f"JAX compilation cache configured at {cache_dir}")

    _setup_jax(platform, enable_x64=dtype == "float64")

    bundle = load_model_bundle(str(bundle_path), dtype=dtype)
    return bundle


def _benchmark_torch(
    model,
    h5_files,
    z_table,
    r_max,
    batch_size,
    dtype,
    device,
    max_batches,
):
    wall_start = time.perf_counter()
    total_graphs = 0
    batches_seen = 0

    sync = (
        torch.cuda.synchronize
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )

    with torch.no_grad():
        base_args = SimpleNamespace(
            pin_memory=False,
            workers=0,
            batch_size=batch_size,
            shuffle=False,
            batch_max_nodes=None,
            batch_max_edges=None,
            batch_drop=False,
            niggli_reduce=False,
        )
        for h5_path in h5_files:
            loader = get_dataloader_torch(
                base_args,
                h5_path,
                atomic_numbers=z_table,
                r_max=r_max,
                accelerator=None,
            )
            if loader is None:
                raise RuntimeError(f"Failed to build Torch loader for {h5_path}")

            pbar = tqdm(total=len(loader), desc=f"Torch {h5_path.name}", leave=True)

            for item in loader:
                batches = item if isinstance(item, list) else [item]
                if len(batches) > 1:
                    pbar.total += len(batches) - 1
                    pbar.refresh()
                for batch in batches:
                    if max_batches is not None and batches_seen >= max_batches:
                        pbar.close()
                        return total_graphs, batches_seen, wall_start

                    batch = batch.to(device=device)

                    if sync:
                        sync()
                    pred = _forward_torch(model, batch)
                    energy = pred["energy"].detach()
                    if sync:
                        sync()

                    total_graphs += energy.shape[0]
                    batches_seen += 1
                    pbar.update(1)
            pbar.close()

    wall_time = time.perf_counter() - wall_start
    return total_graphs, batches_seen, wall_time


def _benchmark_jax(
    bundle,
    loader,
    num_species,
    dtype,
    max_batches,
    *,
    multi_gpu: bool = False,
):
    wall_start = time.perf_counter()
    jnp_dtype = jnp.dtype(dtype)
    jit_apply = jax.jit(
        bundle.module.apply,
        static_argnames=("compute_force", "compute_stress"),
    )
    # Optional pmap across GPUs.
    num_devices = jax.local_device_count() if multi_gpu else 1
    use_pmap = multi_gpu and num_devices > 1
    pmap_apply = None
    params_for_apply = bundle.params
    if use_pmap:
        pmap_apply = jax.pmap(
            bundle.module.apply,
            axis_name="devices",
            static_argnames=("compute_force", "compute_stress"),
        )
        params_for_apply = replicate_to_local_devices(bundle.params)

    total_graphs = 0
    batches_seen = 0
    compile_time = None
    shape_hits: dict[tuple[int, int, int], int] = {}

    if loader is None:
        return total_graphs, batches_seen, compile_time, wall_start

    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    pack_info = getattr(loader, "pack_info", lambda: {})()
    dropped_graphs = pack_info.get("dropped", 0)
    pad_graphs = pack_info.get("pad_graphs", "?")
    pad_nodes = pack_info.get("pad_nodes", "?")
    pad_edges = pack_info.get("pad_edges", "?")
    print(
        f"JAX loader: batches={total_batches}, pad_graphs={pad_graphs}, "
        f"pad_nodes={pad_nodes}, pad_edges={pad_edges}"
    )
    if dropped_graphs:
        print(
            f"WARNING: Dropped {dropped_graphs} graphs exceeding "
            "the configured batch limits."
        )

    graphs_loader = loader
    load_id = "loader_stream"

    print(f"Running JAX load {load_id}: batches={total_batches}")

    def _iter_batches(loader):
        for item in loader:
            if isinstance(item, list):
                for sub in item:
                    yield sub
            else:
                yield item

    batch_iter = _iter_batches(graphs_loader)
    first_batch = next(batch_iter, None)

    # Warmup compile on first batch (excluded from totals).
    if first_batch is not None:
        warm_batch_jax = None
        if use_pmap:
            try:
                warm_batch_jax = prepare_sharded_batch(first_batch, num_devices)
            except ValueError:
                warm_batch_jax = None
        if use_pmap and warm_batch_jax is None:
            tqdm.write(
                "[JAX] Disabling multi-GPU: batch graphs not divisible by device count."
            )
            use_pmap = False
            params_for_apply = bundle.params
        if not use_pmap:
            warm_batch_jax = _graph_to_jax_data(
                first_batch, num_species=num_species, jnp_dtype=jnp_dtype
            )
        mask = np.asarray(jraph.get_graph_padding_mask(first_batch))
        real_graphs = int(np.sum(mask))
        warm_start = time.perf_counter()
        if use_pmap and pmap_apply is not None:
            pred = pmap_apply(
                params_for_apply,
                warm_batch_jax,
                compute_force=False,
                compute_stress=False,
            )
        else:
            pred = jit_apply(
                params_for_apply,
                warm_batch_jax,
                compute_force=False,
                compute_stress=False,
            )
        energy = pred["energy"]
        if hasattr(energy, "block_until_ready"):
            energy.block_until_ready()
        compile_time = time.perf_counter() - warm_start
        tqdm.write(f"[JAX] Compile finished in {compile_time:.3f}s for first shape")

    chained_iter = (
        itertools.chain([first_batch], batch_iter)
        if first_batch is not None
        else batch_iter
    )

    pbar = tqdm(
        chained_iter,
        desc="JAX graphs",
        leave=True,
        total=total_batches,
        position=0,
    )

    for graph in pbar:
        if max_batches is not None and batches_seen >= max_batches:
            break

        mask = np.asarray(jraph.get_graph_padding_mask(graph))
        real_graphs = int(np.sum(mask))
        num_graphs = int(graph.n_node.shape[0])
        num_nodes = int(graph.nodes.positions.shape[0])
        num_edges = int(graph.edges.shifts.shape[0])
        signature = (num_graphs, num_nodes, num_edges)
        shape_hits[signature] = shape_hits.get(signature, 0) + 1
        if shape_hits[signature] == 1 and len(shape_hits) <= 5:
            pbar.write(
                f"[JAX] Padded batch shape graphs={num_graphs} atoms={num_nodes} edges={num_edges} (will trigger XLA compile)"
            )

        batch_jax = None
        if use_pmap:
            try:
                batch_jax = prepare_sharded_batch(graph, num_devices)
            except ValueError:
                batch_jax = None
            if batch_jax is None:
                pbar.write(
                    "[JAX] Disabling multi-GPU mid-run: batch not divisible by device count."
                )
                use_pmap = False
                params_for_apply = bundle.params
        if not use_pmap:
            batch_jax = _graph_to_jax_data(
                graph, num_species=num_species, jnp_dtype=jnp_dtype
            )

        if use_pmap and pmap_apply is not None and batch_jax is not None:
            pred = pmap_apply(
                params_for_apply,
                batch_jax,
                compute_force=False,
                compute_stress=False,
            )
        else:
            pred = jit_apply(
                params_for_apply,
                batch_jax,
                compute_force=False,
                compute_stress=False,
            )
        energy = pred["energy"]
        if hasattr(energy, "block_until_ready"):
            energy.block_until_ready()

        total_graphs += real_graphs
        batches_seen += 1

    wall_time = time.perf_counter() - wall_start
    return total_graphs, batches_seen, compile_time, wall_time


def main() -> None:
    args = _parse_args()

    torch_dtype = getattr(torch, args.dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    device, jax_platform = _resolve_devices(args.device)

    def _load_torch_model() -> torch.nn.Module:
        model = torch.load(
            args.torch_model,
            map_location=device,
            weights_only=False,
        )
        model = model.to(device=device, dtype=torch_dtype).eval()
        if hasattr(model, "atomic_energies_fn"):
            energies = getattr(model.atomic_energies_fn, "atomic_energies", None)
            if isinstance(energies, torch.Tensor):
                model.atomic_energies_fn.atomic_energies = energies.to(torch_dtype)
        return model

    def _list_h5_files() -> list[Path]:
        files = sorted(args.data_dir.glob("*.h5"))
        if args.split != "all":
            files = [p for p in files if p.stem == args.split]
        if not files:
            raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")
        return files

    torch_model = _load_torch_model()
    bundle = _load_jax_bundle(args.jax_model, dtype=args.dtype, platform=jax_platform)

    atomic_numbers = _extract_atomic_numbers(torch_model, bundle)
    r_max = _extract_r_max(torch_model, bundle)
    z_table = AtomicNumberTable(atomic_numbers)
    h5_files = _list_h5_files()

    prep_start = time.perf_counter()
    jax_loader = get_dataloader_jax(
        data_file=h5_files,
        atomic_numbers=z_table,
        r_max=r_max,
        batch_size=None,
        shuffle=False,
        max_nodes=args.max_nodes_per_batch,
        max_edges=args.max_edges_per_batch,
        seed=None,
        niggli_reduce=False,
        max_batches=args.max_batches,
        prefetch_batches=args.prefetch_batches,
    )
    print(
        f"Built JAX loader in {time.perf_counter() - prep_start:.2f}s "
        "(streaming from HDF5)"
    )

    (
        jax_graphs,
        jax_batches,
        jax_compile,
        jax_wall_time,
    ) = _benchmark_jax(
        bundle,
        jax_loader,
        len(atomic_numbers),
        args.dtype,
        args.max_batches,
        multi_gpu=args.multi_gpu,
    )

    (
        torch_graphs,
        torch_batches,
        torch_wall_time,
    ) = _benchmark_torch(
        torch_model,
        h5_files,
        z_table,
        r_max,
        args.batch_size,
        torch_dtype,
        device,
        args.max_batches,
    )

    def _throughput(graphs, elapsed):
        return graphs / elapsed if elapsed and graphs else 0.0

    print(
        f"Torch [{args.dtype}] on {device}: "
        f"{torch_graphs} graphs across {torch_batches} batches "
        f"in {torch_wall_time:.3f}s (wall, includes all prep/compute) => "
        f"{_throughput(torch_graphs, torch_wall_time):.2f} graphs/s"
    )

    print(
        f"JAX [{args.dtype}] on {jax_platform}: "
        f"{jax_graphs} graphs across {jax_batches} batches "
        f"in {jax_wall_time:.3f}s (wall, includes compile) => "
        f"{_throughput(jax_graphs, jax_wall_time):.2f} graphs/s"
    )
    if jax_compile is not None:
        print(f"JAX compile+first-step time: {jax_compile:.3f}s")


if __name__ == "__main__":
    main()
