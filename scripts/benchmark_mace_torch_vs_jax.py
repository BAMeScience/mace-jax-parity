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
import multiprocessing as mp
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import torch

# Torch 2.6 tightened torch.load defaults; the checkpoints still store ``slice``.
from torch.serialization import add_safe_globals
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from equitrain.backends.jax_utils import load_model_bundle
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import atoms_to_graphs
from equitrain.data.backend_jax.atoms_to_graphs import graph_to_data
from equitrain.data.format_hdf5 import HDF5GraphDataset

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
        "--max-nodes",
        type=int,
        default=None,
        help="Optional cap on nodes per padded JAX batch (drops oversize graphs when used).",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional cap on edges per padded JAX batch (drops oversize graphs when used).",
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
    h5_path: Path,
    z_table: AtomicNumberTable,
    r_max: float,
    batch_size: int,
    *,
    drop_last: bool = False,
):
    dataset = HDF5GraphDataset(
        h5_path,
        r_max=r_max,
        atomic_numbers=z_table,
    )

    return PyGDataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )


def _prepare_jax_graphs(
    h5_files: list[Path],
    z_table: AtomicNumberTable,
    r_max: float,
    batch_size: int,
    max_batches: int | None,
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
):
    """
    Load HDF5 structures into jraph.GraphsTuple objects and build a padded loader.
    """
    graphs = []
    target_graphs = None if max_batches is None else max_batches * batch_size
    dropped_graphs = 0

    for h5_path in h5_files:
        new_graphs = atoms_to_graphs(h5_path, r_max=r_max, z_table=z_table)
        graphs.extend(new_graphs)
        if target_graphs is not None and len(graphs) >= target_graphs:
            graphs = graphs[:target_graphs]
            break
    observed_nodes = max((int(g.n_node.sum()) for g in graphs), default=0)
    observed_edges = max((int(g.n_edge.sum()) for g in graphs), default=0)

    if max_nodes is not None or max_edges is not None:
        filtered = []
        for g in graphs:
            nodes = int(g.n_node.sum())
            edges = int(g.n_edge.sum())
            if (max_nodes is not None and nodes > max_nodes) or (
                max_edges is not None and edges > max_edges
            ):
                dropped_graphs += 1
                continue
            filtered.append(g)
        graphs = filtered

    return graphs, observed_nodes, observed_edges, dropped_graphs


def _pack_by_edges(
    graphs: list,
    *,
    max_edges_per_batch: int,
    max_nodes_per_batch: int | None,
    batch_size_limit: int | None,
):
    """Greedy pack graphs into batches limited by edge/node counts."""
    batches: list[list] = []
    current: list = []
    edge_sum = 0
    node_sum = 0
    for g in graphs:
        g_edges = int(g.n_edge.sum())
        g_nodes = int(g.n_node.sum())
        would_edges = edge_sum + g_edges
        would_nodes = node_sum + g_nodes
        if current and (
            would_edges > max_edges_per_batch
            or (max_nodes_per_batch is not None and would_nodes > max_nodes_per_batch)
            or (batch_size_limit is not None and len(current) >= batch_size_limit)
        ):
            batches.append(current)
            current = []
            edge_sum = 0
            node_sum = 0
        current.append(g)
        edge_sum += g_edges
        node_sum += g_nodes
    if current:
        batches.append(current)
    # Build fixed padding based on worst-case batch sums.
    max_graphs = max((len(b) for b in batches), default=0)
    max_nodes = max((sum(int(g.n_node.sum()) for g in b) for b in batches), default=0)
    max_edges = max((sum(int(g.n_edge.sum()) for g in b) for b in batches), default=0)
    pad_graphs = max_graphs + 1 if max_graphs else 1
    pad_nodes = max_nodes + 1 if max_nodes else 1
    pad_edges = max_edges + 1 if max_edges else 1
    padded_batches = []
    for b in batches:
        batched = jraph.batch_np(b)
        padded = jraph.pad_with_graphs(
            batched,
            n_node=pad_nodes,
            n_edge=pad_edges,
            n_graph=pad_graphs,
        )
        padded_batches.append(padded)
    return padded_batches


def _producer_enqueue_batches(batches: list, queue: mp.Queue):
    """CPU worker to enqueue prebuilt batches."""
    try:
        for b in batches:
            queue.put(b, block=True)
    finally:
        queue.put(None)


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
        for h5_path in h5_files:
            loader = _build_loader(
                h5_path,
                z_table,
                r_max,
                batch_size,
                drop_last=False,
            )
            for batch in tqdm(loader, desc=f"Torch {h5_path.name}", leave=True):
                if max_batches is not None and batches_seen >= max_batches:
                    return total_graphs, batches_seen, wall_start

                batch = batch.to(device)
                batch = _cast_batch(batch, dtype)

                if sync:
                    sync()
                pred = _forward_torch(model, batch)
                energy = _extract_energy(pred)
                energy = energy.detach()
                if sync:
                    sync()

                total_graphs += energy.shape[0]
                batches_seen += 1

    wall_time = time.perf_counter() - wall_start
    return total_graphs, batches_seen, wall_time


def _benchmark_jax(
    jax,
    jnp,
    bundle,
    graphs,
    num_species,
    dtype,
    max_batches,
    *,
    max_edges_per_batch: int,
    max_nodes_per_batch: int | None,
    prefetch_batches: int = 0,
):
    def _maybe_prefetch(source, total_batches: int | None):
        """Optionally prefetch batches in a background process."""
        if prefetch_batches and prefetch_batches > 0:
            ctx = mp.get_context("spawn")
            q: mp.Queue = ctx.Queue(maxsize=prefetch_batches)
            proc = ctx.Process(
                target=_producer_enqueue_batches,
                args=(source, q),
                daemon=True,
            )
            proc.start()

            def _queue_iter():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    yield item

            return _queue_iter(), total_batches, "packed_prefetch"
        return source, total_batches, "packed"

    wall_start = time.perf_counter()
    jnp_dtype = jnp.dtype(dtype)
    jit_apply = jax.jit(
        bundle.module.apply,
        static_argnames=("compute_force", "compute_stress"),
    )

    total_graphs = 0
    batches_seen = 0
    compile_time = None
    shape_hits: dict[tuple[int, int, int], int] = {}

    if graphs is None:
        return total_graphs, batches_seen, compile_time, wall_start

    packed_batches = _pack_by_edges(
        graphs,
        max_edges_per_batch=max_edges_per_batch,
        max_nodes_per_batch=max_nodes_per_batch,
        batch_size_limit=None,
    )
    print(
        f"Greedy edge-packed batches: {len(packed_batches)} batches, "
        f"max_edges_per_batch={max_edges_per_batch}, "
        f"max_nodes_per_batch={max_nodes_per_batch}"
    )

    total_batches = len(packed_batches)

    graphs_loader, total_batches, load_id = _maybe_prefetch(
        packed_batches, total_batches
    )

    print(
        f"Running JAX load {load_id}: batches={total_batches}, "
        f"max_edges_per_batch={max_edges_per_batch}, "
        f"max_nodes_per_batch={max_nodes_per_batch}"
    )

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
        mask = np.asarray(jraph.get_graph_padding_mask(first_batch))
        real_graphs = int(np.sum(mask))
        warm_batch_jax = _graph_to_jax_data(
            first_batch, num_species=num_species, jnp_dtype=jnp_dtype
        )
        warm_start = time.perf_counter()
        pred = jit_apply(
            bundle.params,
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

        batch_jax = _graph_to_jax_data(
            graph, num_species=num_species, jnp_dtype=jnp_dtype
        )

        pred = jit_apply(
            bundle.params,
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

    bundle = _load_jax_bundle(args.jax_model, dtype=args.dtype, platform=jax_platform)

    atomic_numbers = _extract_atomic_numbers(torch_model, bundle)
    r_max = _extract_r_max(torch_model, bundle)

    z_table = AtomicNumberTable(atomic_numbers)

    h5_files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        h5_files = [p for p in h5_files if p.stem == args.split]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    prep_start = time.perf_counter()
    (
        jax_graphs_list,
        observed_nodes,
        observed_edges,
        dropped_graphs,
    ) = _prepare_jax_graphs(
        h5_files,
        z_table,
        r_max,
        args.batch_size,
        args.max_batches,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
    )
    if dropped_graphs:
        print(
            f"WARNING: Dropped {dropped_graphs} graphs exceeding "
            f"max_nodes={args.max_nodes} or max_edges={args.max_edges}."
        )
    print(
        f"Built JAX graphs/loader in {time.perf_counter() - prep_start:.2f}s "
        f"({len(jax_graphs_list)} graphs loaded, "
        f"observed_max_nodes={observed_nodes}, observed_max_edges={observed_edges})"
    )

    (
        jax_graphs,
        jax_batches,
        jax_compile,
        jax_wall_time,
    ) = _benchmark_jax(
        jax,
        jnp,
        bundle,
        jax_graphs_list,
        len(atomic_numbers),
        args.dtype,
        args.max_batches,
        prefetch_batches=args.prefetch_batches,
        max_edges_per_batch=args.max_edges_per_batch,
        max_nodes_per_batch=args.max_nodes_per_batch,
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
