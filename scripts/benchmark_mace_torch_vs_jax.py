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
import os
import pickle
import time
from pathlib import Path
from typing import Iterable

import jraph
import numpy as np
import torch
import multiprocessing as mp
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
        "--drop-oversize",
        action="store_true",
        help="Drop graphs that exceed --max-nodes/--max-edges instead of padding to their size.",
    )
    parser.add_argument(
        "--graphs-cache",
        type=Path,
        default=None,
        help="Optional path to cache prebuilt JAX graphs (pickle). Speeds up repeated runs.",
    )
    parser.add_argument(
        "--stream-jax",
        action="store_true",
        help="Stream HDF5 -> JAX graphs on the fly (avoids holding all graphs in memory). "
        "Requires --max-nodes/--max-edges to bound padding.",
    )
    parser.add_argument(
        "--write-graph-cache",
        action="store_true",
        help="When loading torch/PyG batches, store generated graphs into the HDF5 graph_cache dataset.",
    )
    parser.add_argument(
        "--max-edges-per-batch",
        type=int,
        default=None,
        help="Optional hard cap on total edges per JAX batch (greedy packing).",
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
    cache_graphs: bool = False,
):
    if cache_graphs:
        # Disable HDF5 file locking to avoid contention when appending cache.
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    cache_path = None
    if cache_graphs:
        h5p = Path(h5_path)
        cache_path = h5p.with_name(f"{h5p.stem}_cache{h5p.suffix}")

    def _open_dataset() -> HDF5GraphDataset:
        return HDF5GraphDataset(
            h5_path,
            r_max=r_max,
            atomic_numbers=z_table,
            cache_graphs=cache_graphs,
            cache_path=cache_path,
        )

    try:
        dataset = _open_dataset()
    except BlockingIOError as exc:
        # Force-disable locking and retry once.
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        print(
            f"WARNING: retrying HDF5 open for {h5_path} without file locking "
            f"due to: {exc}"
        )
        dataset = _open_dataset()

    if cache_graphs:
        _populate_graph_cache(dataset)
    return PyGDataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )


def _populate_graph_cache(dataset: HDF5GraphDataset):
    """Populate missing graph_cache entries with a tqdm progress bar."""
    cache_ds = getattr(dataset, "_get_cache_dataset", lambda: None)()
    if cache_ds is None:
        return
    missing_indices = []
    for idx in range(len(dataset)):
        try:
            entry = cache_ds[idx]
            if entry is None or len(entry) == 0:
                missing_indices.append(idx)
        except Exception:
            missing_indices.append(idx)
    if not missing_indices:
        return

    print(
        f"Caching {len(missing_indices)} graphs into graph_cache for {dataset._filename}"
    )
    for idx in tqdm(missing_indices, desc="Caching HDF5 graphs", leave=False):
        _ = dataset[idx]


def _prepare_jax_graphs(
    h5_files: list[Path],
    z_table: AtomicNumberTable,
    r_max: float,
    batch_size: int,
    max_batches: int | None,
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    drop_oversize: bool = False,
    cache_path: Path | None = None,
):
    """
    Load HDF5 structures into jraph.GraphsTuple objects and build a padded loader.
    """
    from equitrain.data.backend_jax import atoms_to_graphs, build_loader

    if cache_path is not None and cache_path.exists():
        try:
            cached = pickle.loads(cache_path.read_bytes())
            graphs = cached.get("graphs", [])
            cached_obs_nodes = cached.get("observed_nodes", 0)
            cached_obs_edges = cached.get("observed_edges", 0)
            dropped_graphs = cached.get("dropped_graphs", 0)
            print(
                f"Loaded {len(graphs)} cached graphs from {cache_path} "
                f"(max_nodes={cached_obs_nodes}, max_edges={cached_obs_edges}, "
                f"dropped={dropped_graphs})."
            )
            loader = build_loader(
                graphs,
                batch_size=batch_size,
                shuffle=False,
                max_nodes=max_nodes,
                max_edges=max_edges,
                drop=drop_oversize,
                force_padding=True,
            )
            return graphs, loader, cached_obs_nodes, cached_obs_edges, dropped_graphs
        except Exception as exc:
            print(
                f"Warning: failed to load cache {cache_path}: {exc}. Rebuilding graphs."
            )

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

    if drop_oversize and (max_nodes is not None or max_edges is not None):
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

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = dict(
                graphs=graphs,
                observed_nodes=observed_nodes,
                observed_edges=observed_edges,
                dropped_graphs=dropped_graphs,
            )
            cache_path.write_bytes(pickle.dumps(cache_payload))
            print(f"Saved {len(graphs)} graphs to cache at {cache_path}.")
        except Exception as exc:
            print(f"Warning: failed to save cache {cache_path}: {exc}.")

    return graphs, observed_nodes, observed_edges, dropped_graphs


def _build_streaming_jax_loader(
    h5_files: list[Path],
    z_table: AtomicNumberTable,
    r_max: float,
    batch_size: int,
    *,
    max_nodes: int,
    max_edges: int,
    max_batches: int | None,
    drop_oversize: bool,
):
    """
    Stream graphs from HDF5 -> JAX batches without holding all graphs in memory.
    """
    import jraph

    from equitrain.data.backend_jax.atoms_to_graphs import graph_from_configuration
    from equitrain.data.configuration import Configuration as EqConfiguration
    from equitrain.data.format_hdf5.dataset import HDF5Dataset

    if max_nodes is None or max_edges is None:
        raise ValueError("Streaming JAX loader requires --max-nodes and --max-edges.")

    class _StreamingLoader:
        def __init__(self):
            self._n_node = max_nodes
            self._n_edge = max_edges
            self._n_graph = batch_size

        def __iter__(self):
            dropped = 0
            emitted_batches = 0
            current = []
            nodes = 0
            edges = 0

            def _flush():
                nonlocal current, nodes, edges, emitted_batches
                if not current:
                    return None
                batched = current[0] if len(current) == 1 else jraph.batch_np(current)
                padded = jraph.pad_with_graphs(
                    batched,
                    n_node=max_nodes,
                    n_edge=max_edges,
                    n_graph=batch_size,
                )
                current = []
                nodes = 0
                edges = 0
                emitted_batches += 1
                return padded

            for h5_path in h5_files:
                ds = HDF5Dataset(h5_path, mode="r")
                try:
                    for idx in range(len(ds)):
                        atoms = ds[idx]
                        conf = EqConfiguration.from_atoms(atoms)
                        graph = graph_from_configuration(
                            conf, cutoff=r_max, z_table=z_table
                        )
                        g_nodes = int(graph.n_node.sum())
                        g_edges = int(graph.n_edge.sum())
                        if drop_oversize and (
                            g_nodes > max_nodes or g_edges > max_edges
                        ):
                            dropped += 1
                            continue
                        if current and (
                            len(current) >= batch_size
                            or nodes + g_nodes > max_nodes
                            or edges + g_edges > max_edges
                        ):
                            batch = _flush()
                            if batch is not None:
                                yield batch
                                if (
                                    max_batches is not None
                                    and emitted_batches >= max_batches
                                ):
                                    print(
                                        f"Streaming JAX loader dropped {dropped} graphs (oversize)."
                                    )
                                    return
                        current.append(graph)
                        nodes += g_nodes
                        edges += g_edges
                    # End h5 file
                finally:
                    ds.close()
            final_batch = _flush()
            if final_batch is not None:
                yield final_batch
            if dropped:
                print(
                    f"Streaming JAX loader dropped {dropped} graphs exceeding "
                    f"max_nodes={max_nodes} or max_edges={max_edges}."
                )

    return _StreamingLoader()


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


def _graph_to_jax_data(graph, num_species: int, jnp, jax, jnp_dtype):
    from mace_jax.tools.gin_model import _graph_to_data

    data = _graph_to_data(graph, num_species=num_species)
    casted = {}
    for k, v in data.items():
        if hasattr(v, "dtype") and v.dtype.kind == "f":
            casted[k] = v.astype(jnp_dtype)
        else:
            casted[k] = v
    data = jax.device_put(casted)
    return data


def _forward_torch(model, batch):
    try:
        return model(batch, compute_force=False, compute_stress=False)
    except TypeError:
        return model(batch)


def _setup_jax(platform: str, enable_x64: bool):
    import jax

    jax.config.update("jax_platform_name", platform)
    jax.config.update("jax_enable_x64", enable_x64)
    # Enable persistent compilation cache if the env var is set.
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        try:
            import jax.experimental.compilation_cache as cc

            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cc.initialize_cache(cache_dir)
            print(f"JAX compilation cache initialized at {cache_dir}")
        except Exception as exc:
            print(f"Warning: failed to initialize JAX compilation cache: {exc}")
    # Favor speed (allow TF32/fast matmul on GPU).
    try:
        jax.config.update("jax_default_matmul_precision", "fastest")
    except Exception:
        pass
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


def _load_jax_bundle(bundle_path: Path, dtype: str, platform: str):
    # If no compilation cache is configured, create one alongside the bundle.
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        cache_dir = (
            Path(bundle_path)
            .with_suffix("")
            .with_name(f"{Path(bundle_path).name}_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    jax, jnp = _setup_jax(platform, enable_x64=dtype == "float64")
    from equitrain.backends.jax_utils import load_model_bundle

    bundle = load_model_bundle(str(bundle_path), dtype=dtype)
    return jax, jnp, bundle


def _benchmark_torch(
    model,
    h5_files,
    z_table,
    r_max,
    batch_size,
    dtype,
    device,
    max_batches,
    cache_graphs: bool = False,
):
    wall_start = time.perf_counter()
    total_graphs = 0
    total_time = 0.0
    prep_time = 0.0
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
                cache_graphs=cache_graphs,
            )
            for batch in tqdm(loader, desc=f"Torch {h5_path.name}", leave=True):
                if max_batches is not None and batches_seen >= max_batches:
                    return total_graphs, total_time, batches_seen, prep_time

                prep_start = time.perf_counter()
                batch = batch.to(device)
                batch = _cast_batch(batch, dtype)
                prep_time += time.perf_counter() - prep_start

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

    wall_time = time.perf_counter() - wall_start
    return total_graphs, total_time, batches_seen, prep_time, wall_time


def _benchmark_jax(
    jax,
    jnp,
    bundle,
    graphs_loader,
    num_species,
    dtype,
    max_batches,
    *,
    warmup_compile: bool = True,
    total_override: int | None = None,
):
    wall_start = time.perf_counter()
    jnp_dtype = jnp.dtype(dtype)
    jit_apply = jax.jit(
        bundle.module.apply,
        static_argnames=("compute_force", "compute_stress"),
    )

    total_graphs = 0
    total_time = 0.0
    convert_time = 0.0
    apply_time = 0.0
    batches_seen = 0
    compile_time = None
    first_batch_graphs = None
    shape_hits: dict[tuple[int, int, int], int] = {}

    if graphs_loader is None:
        return (
            total_graphs,
            total_time,
            batches_seen,
            compile_time,
            first_batch_graphs,
            convert_time,
            apply_time,
            shape_hits,
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
    if warmup_compile and first_batch is not None:
        mask = np.asarray(jraph.get_graph_padding_mask(first_batch))
        real_graphs = int(np.sum(mask))
        convert_start = time.perf_counter()
        warm_batch_jax = _graph_to_jax_data(
            first_batch, num_species=num_species, jnp=jnp, jax=jax, jnp_dtype=jnp_dtype
        )
        convert_time += time.perf_counter() - convert_start
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
        first_batch_graphs = real_graphs
        tqdm.write(f"[JAX] Compile finished in {compile_time:.3f}s for first shape")

    total_batches = total_override if total_override is not None else len(graphs_loader)

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

        convert_start = time.perf_counter()
        batch_jax = _graph_to_jax_data(
            graph, num_species=num_species, jnp=jnp, jax=jax, jnp_dtype=jnp_dtype
        )
        convert_time += time.perf_counter() - convert_start

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

        total_graphs += real_graphs
        total_time += step_time
        apply_time += step_time
        batches_seen += 1

    wall_time = time.perf_counter() - wall_start
    return (
        total_graphs,
        total_time,
        batches_seen,
        compile_time,
        first_batch_graphs,
        convert_time,
        apply_time,
        shape_hits,
        wall_time,
    )


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

    jax, jnp, bundle = _load_jax_bundle(
        args.jax_model, dtype=args.dtype, platform=jax_platform
    )

    atomic_numbers = _extract_atomic_numbers(torch_model, bundle)
    r_max = _extract_r_max(torch_model, bundle)

    z_table = AtomicNumberTable(atomic_numbers)

    h5_files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        h5_files = [p for p in h5_files if p.stem == args.split]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")

    prep_start = time.perf_counter()
    if args.stream_jax:
        jax_loader = _build_streaming_jax_loader(
            h5_files,
            z_table,
            r_max,
            args.batch_size,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_batches=args.max_batches,
            drop_oversize=args.drop_oversize,
        )
        print(
            f"JAX streaming loader: graphs={jax_loader._n_graph}, "
            f"nodes={jax_loader._n_node}, edges={jax_loader._n_edge}"
        )
        jax_graphs_list = []  # not materialized
    else:
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
            drop_oversize=args.drop_oversize,
            cache_path=args.graphs_cache,
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

    jax_loads = []
    if args.max_edges_per_batch:
        packed_batches = _pack_by_edges(
            jax_graphs_list,
            max_edges_per_batch=args.max_edges_per_batch,
            max_nodes_per_batch=args.max_nodes_per_batch,
            batch_size_limit=None,
        )
        print(
            f"Greedy edge-packed batches: {len(packed_batches)} batches, "
            f"max_edges_per_batch={args.max_edges_per_batch}, "
            f"max_nodes_per_batch={args.max_nodes_per_batch}"
        )
        if args.prefetch_batches and args.prefetch_batches > 0:
            ctx = mp.get_context("spawn")
            q: mp.Queue = ctx.Queue(maxsize=args.prefetch_batches)
            proc = ctx.Process(
                target=_producer_enqueue_batches,
                args=(packed_batches, q),
                daemon=True,
            )
            proc.start()

            def _queue_iter():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    yield item

            jax_loads = [("packed_prefetch", _queue_iter(), 0, len(packed_batches))]
        else:
            jax_loads = [("packed", packed_batches, 0, len(packed_batches))]
    else:
        from equitrain.data.backend_jax import build_loader

        node_counts = sorted(
            (int(g.n_node.sum()) for g in jax_graphs_list), reverse=True
        )
        edge_counts = sorted(
            (int(g.n_edge.sum()) for g in jax_graphs_list), reverse=True
        )
        worst_nodes = sum(node_counts[: args.batch_size]) if node_counts else 0
        worst_edges = sum(edge_counts[: args.batch_size]) if edge_counts else 0
        loader = build_loader(
            jax_graphs_list,
            batch_size=args.batch_size,
            shuffle=False,
            max_nodes=observed_nodes,
            max_edges=observed_edges,
            drop=False,
            force_padding=True,
            pad_total_nodes=worst_nodes + 1,
            pad_total_edges=worst_edges + 1,
        )
        total_batches = None
        try:
            total_batches = len(loader)
        except Exception:
            total_batches = None
        jax_loads = [("all_graphs", loader, 0, total_batches)]

    jax_wall_start = time.perf_counter()
    jax_wall_time = 0.0
    jax_graphs = jax_time = jax_batches = jax_compile = jax_convert_time = (
        jax_apply_time
    ) = 0
    first_batch_graphs = None
    jax_shape_hits: dict[tuple[int, int, int], int] = {}
    for load_id, loader, dropped, total_batches in jax_loads:
        if loader is None:
            continue
        graph_source = "list" if isinstance(loader, list) else "loader"
        graph_count = (
            len(loader) if isinstance(loader, list) else getattr(loader, "_n_graph", 0)
        )
        print(
            f"Running JAX load {load_id}: graphs={graph_count}, dropped={dropped} (source={graph_source})"
        )
        (
            b_graphs,
            b_time,
            b_batches,
            b_compile,
            b_first_graphs,
            b_convert_time,
            b_apply_time,
            b_shape_hits,
            b_wall_time,
        ) = _benchmark_jax(
            jax,
            jnp,
            bundle,
            loader,
            len(atomic_numbers),
            args.dtype,
            args.max_batches,
            warmup_compile=True,
            total_override=total_batches,
        )
        jax_graphs += b_graphs
        jax_time += b_time
        jax_batches += b_batches
        jax_convert_time += b_convert_time
        jax_apply_time += b_apply_time
        jax_shape_hits.update(b_shape_hits)
        if first_batch_graphs is None:
            first_batch_graphs = b_first_graphs
        if jax_compile is None and b_compile is not None:
            jax_compile = b_compile
        jax_wall_time += b_wall_time
    if jax_wall_time == 0:
        jax_wall_time = time.perf_counter() - jax_wall_start

    (
        torch_graphs,
        torch_time,
        torch_batches,
        torch_prep,
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
        cache_graphs=args.write_graph_cache,
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
    if steadystate_time is not None and steadystate_graphs is not None:
        print(
            f"JAX steady-state: {steadystate_graphs} graphs "
            f"in {steadystate_time:.3f}s => "
            f"{_throughput(steadystate_graphs, steadystate_time):.2f} graphs/s"
        )


if __name__ == "__main__":
    main()
