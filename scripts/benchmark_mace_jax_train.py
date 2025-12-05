#!/usr/bin/env python3
"""Benchmark a single training epoch for a JAX MACE bundle."""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import jax
import jraph
import numpy as np
from jax import tree_util as jtu
from tqdm import tqdm

import torch
try:
    from torch.serialization import add_safe_globals as _torch_add_safe_globals
except ImportError:
    _torch_add_safe_globals = None

if callable(_torch_add_safe_globals):
    _torch_add_safe_globals([slice])

from equitrain.backends.jax_backend import TrainState, _build_train_functions
from equitrain.backends.jax_loss import JaxLossCollection, update_collection_from_aux
from equitrain.backends.jax_loss_fn import LossSettings, build_loss_fn
from equitrain.backends.jax_loss_metrics import LossMetrics
from equitrain.backends.jax_optimizer import create_optimizer
from equitrain.backends.jax_utils import (
    batched_iterator,
    iter_micro_batches,
    load_model_bundle,
    prepare_sharded_batch as _prepare_sharded_batch,
    prepare_single_batch as _prepare_single_batch,
    replicate_to_local_devices,
    supports_multiprocessing_workers,
    take_chunk,
    unreplicate_from_local_devices,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader as get_dataloader_jax
from equitrain.data.backend_jax import make_apply_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one epoch of JAX MACE training on mp-traj data."
    )
    parser.add_argument(
        "--jax-model",
        type=Path,
        required=True,
        help="Directory containing config.json and params.msgpack for the JAX bundle.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mptraj"),
        help="Directory with mp-traj *.h5 files (default: data/mptraj).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="train",
        help="Which split to stream (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Graphs per device (before greedy padding).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Floating point precision for the bundle (default: float32).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optax optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=("adamw", "adam", "sgd", "nesterov", "momentum", "rmsprop"),
        help="Optimizer to use (default: adamw).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizers that support it (default: 0).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum coefficient for SGD-like optimizers (default: 0.9).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSProp alpha smoothing factor (default: 0.99).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Optional gradient clipping value (applied element-wise).",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=1.0,
        help="Loss weight for energies (default: 1.0).",
    )
    parser.add_argument(
        "--forces-weight",
        type=float,
        default=0.0,
        help="Loss weight for forces (default: 0.0).",
    )
    parser.add_argument(
        "--stress-weight",
        type=float,
        default=0.0,
        help="Loss weight for stress (default: 0.0).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="huber",
        choices=("mae", "smooth-l1", "mse", "huber"),
        help="Base loss type (default: huber).",
    )
    parser.add_argument(
        "--loss-type-energy",
        type=str,
        default=None,
        choices=("mae", "smooth-l1", "mse", "huber"),
        help="Override loss type for energies (default: inherit --loss-type).",
    )
    parser.add_argument(
        "--loss-type-forces",
        type=str,
        default=None,
        choices=("mae", "smooth-l1", "mse", "huber"),
        help="Override loss type for forces (default: inherit --loss-type).",
    )
    parser.add_argument(
        "--loss-type-stress",
        type=str,
        default=None,
        choices=("mae", "smooth-l1", "mse", "huber"),
        help="Override loss type for stress (default: inherit --loss-type).",
    )
    parser.add_argument(
        "--smooth-l1-beta",
        type=float,
        default=1.0,
        help="Beta parameter for smooth-L1 losses (default: 1.0).",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.01,
        help="Delta parameter for huber losses (default: 0.01).",
    )
    parser.add_argument(
        "--loss-clipping",
        type=float,
        default=None,
        help="Clip individual loss terms to this value (default: disabled).",
    )
    parser.add_argument(
        "--loss-energy-per-atom",
        dest="loss_energy_per_atom",
        action="store_true",
        default=True,
        help="Compute energy loss per atom (default: enabled).",
    )
    parser.add_argument(
        "--loss-total-energy",
        dest="loss_energy_per_atom",
        action="store_false",
        help="Treat energies as totals (disable per-atom normalization).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling (default: 0).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle the mp-traj dataset (default: enabled).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable dataset shuffling.",
    )
    parser.add_argument(
        "--niggli-reduce",
        action="store_true",
        help="Apply Niggli reduction when building graphs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Limit the number of optimizer steps (default: full epoch).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop streaming after this many greedy-packed batches.",
    )
    parser.add_argument(
        "--max-edges-per-batch",
        type=int,
        default=480000,
        help="Cap the number of edges per padded batch (default: 480000).",
    )
    parser.add_argument(
        "--max-nodes-per-batch",
        type=int,
        default=None,
        help="Optional cap on nodes per padded batch.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=None,
        help="Override loader prefetch depth (default: match worker count).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of graph construction workers (default: 4).",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all local JAX devices via pmap.",
    )
    parser.set_defaults(tqdm=True)
    parser.add_argument(
        "--tqdm",
        dest="tqdm",
        action="store_true",
        help="Display a tqdm progress bar for the epoch (default: enabled).",
    )
    parser.add_argument(
        "--no-tqdm",
        dest="tqdm",
        action="store_false",
        help="Disable the tqdm progress bar during training.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print step-level stats every N updates (default: disabled).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Append benchmark results to this CSV file.",
    )
    return parser.parse_args()


def _setup_jax(dtype: str):
    if dtype.lower() == "float64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)
    try:
        jax.config.update("jax_default_matmul_precision", "fastest")
    except Exception:
        pass


def _extract_atomic_numbers(bundle) -> list[int]:
    numbers = bundle.config.get("atomic_numbers")
    if not numbers:
        raise RuntimeError("Model configuration did not include atomic_numbers.")
    return [int(z) for z in numbers]


def _extract_r_max(bundle) -> float:
    r_max = bundle.config.get("r_max")
    if r_max is None:
        raise RuntimeError("Model configuration missing r_max.")
    return float(r_max)


def _list_h5_files(args) -> list[Path]:
    files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        files = [p for p in files if p.stem == args.split]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")
    return files


def _count_real_graphs(graph: jraph.GraphsTuple) -> int:
    mask = np.asarray(jraph.get_graph_padding_mask(graph))
    return int(np.sum(mask))


def _build_loss_settings(args) -> LossSettings:
    loss_type_energy = args.loss_type_energy or args.loss_type
    loss_type_forces = args.loss_type_forces or args.loss_type
    loss_type_stress = args.loss_type_stress or args.loss_type
    return LossSettings(
        energy_weight=float(args.energy_weight),
        forces_weight=float(args.forces_weight),
        stress_weight=float(args.stress_weight),
        loss_type=args.loss_type,
        loss_type_energy=loss_type_energy,
        loss_type_forces=loss_type_forces,
        loss_type_stress=loss_type_stress,
        loss_weight_type=None,
        loss_weight_type_energy=None,
        loss_weight_type_forces=None,
        loss_weight_type_stress=None,
        loss_energy_per_atom=bool(args.loss_energy_per_atom),
        smooth_l1_beta=float(args.smooth_l1_beta),
        huber_delta=float(args.huber_delta),
        loss_clipping=args.loss_clipping,
    )


def _block_first_leaf(tree) -> None:
    leaves = jtu.tree_leaves(tree, is_leaf=lambda leaf: leaf is None)
    for leaf in leaves:
        if leaf is None:
            continue
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return
        try:
            jax.device_get(leaf)
            return
        except Exception:
            continue


def _unwrap_state(state):
    while isinstance(state, tuple) and len(state) == 1:
        state = state[0]
    return state


def _prepare_chunk(graphs: list[jraph.GraphsTuple], use_pmap: bool, num_devices: int):
    if use_pmap:
        return _prepare_sharded_batch(graphs, num_devices)
    return _prepare_single_batch(graphs[0])


def _build_chunk_iterator(loader, use_pmap: bool, num_devices: int):
    micro_iter = iter_micro_batches(loader)
    effective_multi = use_pmap and num_devices > 1
    group_size = num_devices if effective_multi else 1
    first_chunk = take_chunk(micro_iter, group_size)
    if effective_multi and len(first_chunk) < group_size:
        tqdm.write(
            "[train] Falling back to single-device execution: insufficient micro-batches."
        )
        micro_iter = itertools.chain(first_chunk, micro_iter)
        effective_multi = False
        group_size = 1
        first_chunk = take_chunk(micro_iter, group_size)
    first_chunk = [g for g in first_chunk if g is not None]
    if not first_chunk:
        return None, None, False, 1

    if effective_multi:
        def _warn(count: int, expected: int):
            tqdm.write(
                f"[train] Dropping incomplete multi-device chunk ({count}/{expected})."
            )

        remainder = batched_iterator(
            micro_iter,
            group_size,
            remainder_action=_warn,
        )
        chunk_iter = itertools.chain([first_chunk], remainder)
    else:
        def _single_iter():
            for micro in micro_iter:
                if micro is not None:
                    yield [micro]

        chunk_iter = itertools.chain([first_chunk], _single_iter())

    return chunk_iter, first_chunk, effective_multi, (num_devices if effective_multi else 1)


def _run_warmup_step(
    state,
    chunk,
    *,
    use_pmap: bool,
    num_devices: int,
    grad_step_fn,
    apply_updates_fn,
) -> float:
    state = _unwrap_state(state)
    batch = _prepare_chunk(chunk, use_pmap, num_devices)
    start = time.perf_counter()
    loss, aux, grads = grad_step_fn(state.params, batch)
    _ = aux  # keep reference so compilation cascades
    new_state = apply_updates_fn(state, grads, 0.0)
    new_state = _unwrap_state(new_state)
    _block_first_leaf(loss)
    _block_first_leaf(new_state)
    return time.perf_counter() - start


def _train_epoch(
    state: TrainState,
    loader,
    *,
    grad_step_fn,
    apply_updates_fn,
    max_steps: int | None,
    use_pmap: bool,
    num_devices: int,
    enable_tqdm: bool,
    log_interval: int,
):
    iterator_info = _build_chunk_iterator(loader, use_pmap, num_devices)
    if iterator_info[0] is None:
        return state, JaxLossCollection(), 0, 0, None, 0.0

    chunk_iter, first_chunk, effective_multi, device_count = iterator_info
    compile_time = _run_warmup_step(
        state,
        first_chunk,
        use_pmap=effective_multi,
        num_devices=device_count,
        grad_step_fn=grad_step_fn,
        apply_updates_fn=apply_updates_fn,
    )

    collection = JaxLossCollection()
    total_graphs = 0
    total_steps = 0
    total_batches = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    progress_total = (
        total_batches // device_count if total_batches and effective_multi else total_batches
    )

    pbar = None
    if enable_tqdm:
        pbar = tqdm(total=progress_total, desc="Training", leave=True)

    run_start = time.perf_counter()
    for chunk in chunk_iter:
        if max_steps is not None and total_steps >= max_steps:
            break
        graphs = [g for g in chunk if g is not None]
        if not graphs:
            continue
        state = _unwrap_state(state)
        batch = _prepare_chunk(graphs, effective_multi, device_count)
        loss, aux, grads = grad_step_fn(state.params, batch)
        if effective_multi:
            aux_host = unreplicate_from_local_devices(aux)
        else:
            aux_host = jax.device_get(aux)
        update_collection_from_aux(collection, aux_host)
        state = apply_updates_fn(state, grads, 0.0)
        state = _unwrap_state(state)
        _block_first_leaf(loss)
        _block_first_leaf(state)

        real_graphs = sum(_count_real_graphs(graph) for graph in graphs)
        total_graphs += real_graphs
        total_steps += 1

        if pbar is not None:
            pbar.update(1)
            if total_steps and collection.components['total'].count:
                avg_loss = collection.components['total'].value
                pbar.set_description(f"Training (loss={avg_loss:.4g})")

        if log_interval and total_steps % log_interval == 0:
            if collection.components['total'].count:
                loss_value = collection.components['total'].value
                print(
                    f"Step {total_steps:>5d}: loss={loss_value:.6f}, graphs={total_graphs}"
                )

    if pbar is not None:
        pbar.close()

    run_time = time.perf_counter() - run_start
    return state, collection, total_graphs, total_steps, compile_time, run_time


def _write_results_csv(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "backend",
        "mode",
        "split",
        "dtype",
        "optimizer",
        "learning_rate",
        "graphs",
        "steps",
        "run_time_s",
        "throughput_graphs_per_s",
        "compile_time_s",
        "mean_loss",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerows([row])


def main() -> None:
    args = _parse_args()

    if args.batch_size is None or args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    _setup_jax(args.dtype)
    bundle = load_model_bundle(str(args.jax_model), dtype=args.dtype)
    atomic_numbers = _extract_atomic_numbers(bundle)
    r_max = _extract_r_max(bundle)
    z_table = AtomicNumberTable(atomic_numbers)
    loss_settings = _build_loss_settings(args)

    wrapper = JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=loss_settings.forces_weight > 0.0,
        compute_stress=loss_settings.stress_weight > 0.0,
    )
    apply_fn = make_apply_fn(wrapper, num_species=len(atomic_numbers))
    loss_fn = build_loss_fn(apply_fn, loss_settings)

    optimizer = create_optimizer(
        optimizer_name=args.optimizer,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        momentum=float(args.momentum),
        alpha=float(args.alpha),
        mask=None,
    )
    opt_state = optimizer.init(bundle.params)
    state = TrainState(params=bundle.params, opt_state=opt_state, ema_params=None)

    requested_multi = bool(args.multi_gpu)
    device_count = jax.local_device_count()
    use_multi = requested_multi and device_count > 1
    if use_multi:
        state = replicate_to_local_devices(state)

    grad_step_fn, apply_updates_fn = _build_train_functions(
        loss_fn,
        optimizer,
        grad_clip_value=args.grad_clip,
        use_ema=False,
        multi_device=use_multi,
    )

    h5_files = _list_h5_files(args)
    device_multiplier = device_count if use_multi else 1
    base_workers = max(int(args.num_workers or 0), 0)
    if base_workers > 0 and supports_multiprocessing_workers():
        effective_workers = base_workers * max(device_multiplier, 1)
    else:
        effective_workers = 0
    if args.prefetch_batches is None:
        prefetch_batches = effective_workers
    else:
        prefetch_batches = max(int(args.prefetch_batches or 0), 0)

    seed_value = None if args.seed is None else int(args.seed)
    train_loader = get_dataloader_jax(
        data_file=h5_files,
        atomic_numbers=z_table,
        r_max=r_max,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        max_nodes=args.max_nodes_per_batch,
        max_edges=args.max_edges_per_batch,
        drop=False,
        seed=seed_value,
        niggli_reduce=bool(args.niggli_reduce),
        max_batches=args.max_batches,
        prefetch_batches=prefetch_batches,
        num_workers=effective_workers,
        graph_multiple=max(device_multiplier, 1),
    )

    metric_settings = dict(
        include_energy=loss_settings.energy_weight > 0.0,
        include_forces=loss_settings.forces_weight > 0.0,
        include_stress=loss_settings.stress_weight > 0.0,
        loss_label=loss_settings.loss_type,
    )

    (
        state,
        loss_collection,
        graphs,
        steps,
        compile_time,
        run_time,
    ) = _train_epoch(
        state,
        train_loader,
        grad_step_fn=grad_step_fn,
        apply_updates_fn=apply_updates_fn,
        max_steps=args.max_steps,
        use_pmap=use_multi,
        num_devices=device_multiplier,
        enable_tqdm=bool(args.tqdm),
        log_interval=int(args.log_interval or 0),
    )

    throughput = graphs / run_time if run_time and graphs else 0.0
    metrics = LossMetrics(**metric_settings)
    metrics.update(loss_collection)
    mean_loss = (
        float(metrics.main.meters['total'].avg)
        if metrics.main.meters['total'].count
        else float('nan')
    )

    used_label = f"{jax.default_backend()}_pmap" if use_multi else jax.default_backend()
    print(
        f"Trained {steps} steps ({graphs} graphs) in {run_time:.3f}s => "
        f"{throughput:.2f} graphs/s"
    )
    if compile_time is not None:
        print(f"First-step compile time: {compile_time:.3f}s")
    print(f"Mean loss: {mean_loss:.6f}")

    _write_results_csv(
        args.csv_output,
        {
            "backend": "jax",
            "mode": "train",
            "split": args.split,
            "dtype": args.dtype,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "graphs": graphs,
            "steps": steps,
            "run_time_s": run_time,
            "throughput_graphs_per_s": throughput,
            "compile_time_s": compile_time if compile_time is not None else "",
            "mean_loss": mean_loss,
        },
    )


if __name__ == "__main__":
    main()
