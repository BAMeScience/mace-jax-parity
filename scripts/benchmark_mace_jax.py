#!/usr/bin/env python3
"""
Benchmark the inference throughput of a JAX MACE model on the mp-traj HDF5
datasets. The script streams fixed-shape batches, records timings, and reports
graphs/sec together with the XLA compile overhead. Torch is only used to load
the reference model so we can extract shared metadata such as atomic numbers.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import torch
from jax import tree_util as jtu
from torch.serialization import add_safe_globals
from tqdm import tqdm

from equitrain.backends.jax_utils import (
    load_model_bundle,
    replicate_to_local_devices,
)
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader as get_dataloader_jax
from equitrain.data.backend_jax.atoms_to_graphs import graph_to_data

add_safe_globals([slice])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a JAX MACE bundle on mp-traj data."
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
        "--dtype",
        type=str,
        default="float32",
        help="Floating point precision to enforce (default: float32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for the Torch metadata load (cpu/gpu/cuda/cuda:0). "
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
        help="If >0, build padded JAX batches ahead of time (queue size = value).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of background threads to construct JAX graphs.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs via pmap (requires batch divisible by device count).",
    )
    return parser.parse_args()


def _resolve_devices(spec: str) -> torch.device:
    text = spec.strip()
    lower = text.lower()
    if lower.startswith("gpu"):
        torch_spec = "cuda" + text[3:]
    else:
        torch_spec = text
    return torch.device(torch_spec)


def _extract_atomic_numbers(torch_model, jax_bundle):
    if hasattr(torch_model, "atomic_numbers"):
        numbers = getattr(torch_model, "atomic_numbers")
        if hasattr(numbers, "zs"):
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
    return jax.device_put(casted)


def _count_real_graphs(graph: jraph.GraphsTuple) -> int:
    mask = np.asarray(jraph.get_graph_padding_mask(graph))
    return int(np.sum(mask))


def _prepare_multi_device_batch(
    graphs: list[jraph.GraphsTuple],
    num_species: int,
    jnp_dtype,
):
    device_batches = [
        _graph_to_jax_data(graph, num_species=num_species, jnp_dtype=jnp_dtype)
        for graph in graphs
    ]

    def _stack_or_none(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack(values)

    return jtu.tree_map(_stack_or_none, *device_batches)


def _setup_jax(enable_x64: bool):
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    try:
        jax.config.update("jax_default_matmul_precision", "fastest")
    except Exception:
        pass


def _benchmark_jax(
    bundle,
    loader,
    num_species,
    dtype,
    max_batches,
    *,
    multi_gpu: bool,
):
    wall_start = time.perf_counter()
    jnp_dtype = jnp.dtype(dtype)

    def _apply_fn(params, batch):
        return bundle.module.apply(
            params,
            batch,
            compute_force=False,
            compute_stress=False,
        )

    jit_apply = jax.jit(_apply_fn)
    num_devices = jax.local_device_count() if multi_gpu else 1
    use_pmap = multi_gpu and num_devices > 1
    pmap_apply = None
    params_for_apply = bundle.params
    if use_pmap:
        pmap_apply = jax.pmap(_apply_fn, axis_name="devices")
        params_for_apply = replicate_to_local_devices(bundle.params)

    total_graphs = 0
    batches_seen = 0
    compile_time = None
    shape_hits: dict[tuple[int, int, int, int], int] = {}

    if loader is None:
        return total_graphs, batches_seen, compile_time, wall_start

    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    progress_total = (
        total_batches // num_devices if total_batches and use_pmap else total_batches
    )
    pack_info = getattr(loader, "pack_info", lambda: {})()
    dropped_graphs = pack_info.get("dropped", 0)
    print(
        f"JAX loader: batches={total_batches}, pad_graphs={pack_info.get('pad_graphs', '?')}, "
        f"pad_nodes={pack_info.get('pad_nodes', '?')}, pad_edges={pack_info.get('pad_edges', '?')}"
    )
    if dropped_graphs:
        print(
            f"WARNING: dropped {dropped_graphs} graphs exceeding the configured limits."
        )

    def _iter_micro_batches():
        for item in loader:
            if isinstance(item, list):
                for sub in item:
                    if sub is not None:
                        yield sub
            elif item is not None:
                yield item

    micro_iter = _iter_micro_batches()
    group_size = num_devices if use_pmap else 1

    def _take_chunk(iterator, size: int):
        return list(itertools.islice(iterator, size))

    first_chunk = _take_chunk(micro_iter, group_size)
    if use_pmap and len(first_chunk) < group_size:
        tqdm.write(
            "[JAX] Disabling multi-GPU: insufficient micro-batches for the device count."
        )
        micro_iter = itertools.chain(first_chunk, micro_iter)
        use_pmap = False
        num_devices = 1
        group_size = 1
        progress_total = total_batches
        params_for_apply = bundle.params
        first_chunk = _take_chunk(micro_iter, group_size)

    first_chunk = [g for g in first_chunk if g is not None]
    if not first_chunk:
        return total_graphs, batches_seen, compile_time, wall_start

    def _batched_iter(iterator, size: int):
        while True:
            chunk = _take_chunk(iterator, size)
            if len(chunk) < size:
                if chunk:
                    tqdm.write(
                        "[JAX] Dropping incomplete multi-device chunk "
                        f"({len(chunk)}/{size})."
                    )
                break
            yield chunk

    remaining_iter = _batched_iter(micro_iter, group_size)
    chunk_iter = (
        itertools.chain([first_chunk], remaining_iter)
        if first_chunk
        else remaining_iter
    )

    def _prepare_chunk(graphs_chunk: list[jraph.GraphsTuple]):
        if use_pmap:
            return _prepare_multi_device_batch(
                graphs_chunk, num_species=num_species, jnp_dtype=jnp_dtype
            )
        return _graph_to_jax_data(
            graphs_chunk[0], num_species=num_species, jnp_dtype=jnp_dtype
        )

    warm_start = time.perf_counter()
    warm_batch = _prepare_chunk(first_chunk)
    if use_pmap and pmap_apply is not None:
        pred = pmap_apply(params_for_apply, warm_batch)
    else:
        pred = jit_apply(params_for_apply, warm_batch)
    energy = pred["energy"]
    if hasattr(energy, "block_until_ready"):
        energy.block_until_ready()
    compile_time = time.perf_counter() - warm_start
    tqdm.write(f"[JAX] Compile finished in {compile_time:.3f}s for first shape")

    pbar = tqdm(
        chunk_iter,
        desc="JAX graphs",
        leave=True,
        total=progress_total,
        position=0,
    )

    for chunk in pbar:
        if max_batches is not None and batches_seen >= max_batches:
            break

        graphs = chunk if isinstance(chunk, list) else [chunk]
        real_graphs = sum(_count_real_graphs(graph) for graph in graphs)
        rep_graph = graphs[0]
        signature = (
            int(rep_graph.n_node.shape[0]),
            int(rep_graph.nodes.positions.shape[0]),
            int(rep_graph.edges.shifts.shape[0]),
            len(graphs),
        )
        shape_hits[signature] = shape_hits.get(signature, 0) + 1
        if shape_hits[signature] == 1 and len(shape_hits) <= 5:
            pbar.write(
                "[JAX] Padded batch shape "
                f"graphs/device={signature[0]} atoms={signature[1]} edges={signature[2]} "
                f"devices={signature[3]} (will trigger XLA compile)"
            )

        batch_jax = _prepare_chunk(graphs)
        if use_pmap and pmap_apply is not None:
            pred = pmap_apply(params_for_apply, batch_jax)
        else:
            pred = jit_apply(params_for_apply, batch_jax)
        energy = pred["energy"]
        if hasattr(energy, "block_until_ready"):
            energy.block_until_ready()

        total_graphs += real_graphs
        batches_seen += 1

    wall_time = time.perf_counter() - wall_start
    return total_graphs, batches_seen, compile_time, wall_time


def _load_torch_model(args, device: torch.device, torch_dtype):
    model = torch.load(
        args.torch_model,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device=device, dtype=torch_dtype).eval()
    return model


def _list_h5_files(args) -> list[Path]:
    files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        files = [p for p in files if p.stem == args.split]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")
    return files


def main() -> None:
    args = _parse_args()

    torch_dtype = getattr(torch, args.dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    torch_device = _resolve_devices(args.device)

    torch_model = _load_torch_model(args, torch_device, torch_dtype)
    bundle = load_model_bundle(str(args.jax_model), dtype=args.dtype)
    _setup_jax(enable_x64=args.dtype == "float64")

    atomic_numbers = _extract_atomic_numbers(torch_model, bundle)
    r_max = _extract_r_max(torch_model, bundle)
    z_table = AtomicNumberTable(atomic_numbers)
    h5_files = _list_h5_files(args)

    prep_start = time.perf_counter()
    device_multiplier = jax.local_device_count() if args.multi_gpu else 1
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
        num_workers=args.num_workers,
        graph_multiple=device_multiplier,
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

    def _throughput(graphs, elapsed):
        return graphs / elapsed if elapsed and graphs else 0.0

    print(
        f"JAX [{args.dtype}]: {jax_graphs} graphs across {jax_batches} batches "
        f"in {jax_wall_time:.3f}s (wall, includes compile) => "
        f"{_throughput(jax_graphs, jax_wall_time):.2f} graphs/s"
    )
    if jax_compile is not None:
        print(f"JAX compile+first-step time: {jax_compile:.3f}s")


if __name__ == "__main__":
    main()
