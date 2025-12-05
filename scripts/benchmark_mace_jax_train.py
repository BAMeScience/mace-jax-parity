#!/usr/bin/env python3
"""Benchmark a single training epoch for a JAX MACE bundle."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import jax
from tqdm import tqdm

try:
    from torch.serialization import add_safe_globals as _torch_add_safe_globals
except ImportError:  # pragma: no cover
    _torch_add_safe_globals = None

if callable(_torch_add_safe_globals):  # pragma: no cover
    _torch_add_safe_globals([slice])

from equitrain.argparser import get_args_parser_train
from equitrain.backends.jax_backend import (
    TrainState,
    _build_train_functions,
    _replicate_state,
    _run_train_epoch,
)
from equitrain.backends.jax_loss_fn import LossSettings, build_loss_fn
from equitrain.backends.jax_optimizer import create_optimizer
from equitrain.backends.jax_utils import (
    load_model_bundle,
    supports_multiprocessing_workers,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader as get_dataloader_jax
from equitrain.data.backend_jax import make_apply_fn


class _CountingLoader:
    """Thin wrapper to record how many macro-batches were consumed."""

    def __init__(self, loader):
        self._loader = loader
        self.batches = 0
        self._len_cache = None

    def __iter__(self):
        for item in self._loader:
            self.batches += 1
            yield item

    def __len__(self):
        if self._len_cache is None:
            self._len_cache = len(self._loader)
        return self._len_cache

    def pack_info(self):
        return self._loader.pack_info()

    def __getattr__(self, name):
        return getattr(self._loader, name)


def _augment_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(tqdm=True)
    parser.add_argument(
        "--no-tqdm",
        dest="tqdm",
        action="store_false",
        help="Disable the tqdm progress bar during training.",
    )
    parser.add_argument(
        "--jax-model",
        type=Path,
        default=None,
        help="Alias for --model when pointing to a JAX bundle directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mptraj"),
        help="Directory with mp-traj *.h5 files (used when --train-file is unset).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "all"),
        default="train",
        help="Which mp-traj split to stream when --train-file is unset.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Append the benchmark summary to this CSV file if set.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop streaming after this many greedy-packed batches.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all local JAX devices via pmap.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print step-level stats every N updates (overrides --print-freq).",
    )
    parser.add_argument(
        "--learning-rate",
        dest="lr",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --lr.",
    )
    parser.add_argument(
        "--optimizer",
        dest="opt",
        type=str,
        default=argparse.SUPPRESS,
        help="Alias for --opt.",
    )
    parser.add_argument(
        "--grad-clip",
        dest="gradient_clipping",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias for --gradient-clipping.",
    )
    parser.add_argument(
        "--max-edges-per-batch",
        dest="batch_max_edges",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias for --batch-max-edges.",
    )
    parser.add_argument(
        "--max-nodes-per-batch",
        dest="batch_max_nodes",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias for --batch-max-nodes.",
    )
    parser.add_argument(
        "--max-steps",
        dest="train_max_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias for --train-max-steps.",
    )
    return parser


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "jax_model", None) and not getattr(args, "model", None):
        args.model = str(args.jax_model)
    if args.model is None:
        raise ValueError("--model (or --jax-model) must be specified.")
    if isinstance(args.model, Path):
        args.model = str(args.model)
    args.data_dir = Path(getattr(args, "data_dir", Path("data/mptraj")))
    return args


def _setup_jax(dtype: str):
    if dtype.lower() == "float64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)
    try:
        jax.config.update("jax_default_matmul_precision", "fastest")
    except Exception:  # pragma: no cover
        pass


def _list_h5_files(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob("*.h5"))
    if split != "all":
        files = [p for p in files if p.stem == split]
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found under {data_dir} for split '{split}'"
        )
    return files


def _resolve_float(value, default: float) -> float:
    return float(default if value is None else value)


def _build_loss_settings(args) -> LossSettings:
    loss_type_energy = getattr(args, "loss_type_energy", None) or args.loss_type
    loss_type_forces = getattr(args, "loss_type_forces", None) or args.loss_type
    loss_type_stress = getattr(args, "loss_type_stress", None) or args.loss_type
    return LossSettings(
        energy_weight=_resolve_float(getattr(args, "energy_weight", None), 1.0),
        forces_weight=_resolve_float(getattr(args, "forces_weight", None), 1.0),
        stress_weight=_resolve_float(getattr(args, "stress_weight", None), 1.0),
        loss_type=args.loss_type,
        loss_type_energy=loss_type_energy,
        loss_type_forces=loss_type_forces,
        loss_type_stress=loss_type_stress,
        loss_weight_type=None,
        loss_weight_type_energy=None,
        loss_weight_type_forces=None,
        loss_weight_type_stress=None,
        loss_energy_per_atom=bool(getattr(args, "loss_energy_per_atom", False)),
        smooth_l1_beta=float(args.smooth_l1_beta),
        huber_delta=float(args.huber_delta),
        loss_clipping=getattr(args, "loss_clipping", None),
    )


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
        "batches",
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
        writer.writerow(row)


def main() -> None:
    parser = _augment_train_parser(get_args_parser_train())
    args = _finalize_args(parser.parse_args())
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    _setup_jax(args.dtype)
    bundle = load_model_bundle(str(args.model), dtype=args.dtype)
    atomic_numbers = bundle.config.get("atomic_numbers")
    if not atomic_numbers:
        raise RuntimeError("Model configuration is missing `atomic_numbers`.")
    z_table = AtomicNumberTable(list(atomic_numbers))
    r_max = float(bundle.config.get("r_max", 0.0))
    if r_max <= 0.0:
        raise RuntimeError("Model configuration must define a positive `r_max`.")

    train_files = _list_h5_files(args.data_dir, args.split)
    device_multiplier = jax.local_device_count()
    multi_device = args.multi_gpu and device_multiplier > 1
    base_workers = max(int(args.num_workers or 0), 0)
    if base_workers > 0 and supports_multiprocessing_workers():
        effective_workers = base_workers * (device_multiplier if multi_device else 1)
    else:
        effective_workers = 0
    if args.prefetch_batches is None:
        prefetch_batches = effective_workers
    else:
        prefetch_batches = max(int(args.prefetch_batches or 0), 0)

    raw_loader = get_dataloader_jax(
        data_file=[str(p) for p in train_files],
        atomic_numbers=z_table,
        r_max=r_max,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=False,
        seed=int(args.seed),
        niggli_reduce=bool(getattr(args, "niggli_reduce", False)),
        max_batches=args.max_batches,
        prefetch_batches=prefetch_batches,
        num_workers=effective_workers,
        graph_multiple=device_multiplier if multi_device else 1,
    )
    train_loader = _CountingLoader(raw_loader)

    wrapper = JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=args.forces_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
    )
    apply_fn = make_apply_fn(wrapper, num_species=len(z_table))
    loss_settings = _build_loss_settings(args)
    loss_fn = build_loss_fn(apply_fn, loss_settings)

    optimizer = create_optimizer(
        optimizer_name=args.opt,
        learning_rate=float(args.lr),
        weight_decay=_resolve_float(getattr(args, "weight_decay", None), 0.0),
        momentum=_resolve_float(getattr(args, "momentum", None), 0.0),
        alpha=_resolve_float(getattr(args, "alpha", None), 0.99),
        mask=None,
    )
    opt_state = optimizer.init(bundle.params)
    state = TrainState(params=bundle.params, opt_state=opt_state, ema_params=None)
    if multi_device:
        state = _replicate_state(state)

    grad_step_fn, apply_updates_fn = _build_train_functions(
        loss_fn,
        optimizer,
        grad_clip_value=args.gradient_clipping,
        use_ema=False,
        multi_device=multi_device,
    )
    if multi_device:
        raw_apply_updates_fn = apply_updates_fn

        def apply_updates_fn(state, grads, ema_factor):
            result = raw_apply_updates_fn(state, grads, ema_factor)
            if isinstance(result, tuple) and len(result) == 1:
                return result[0]
            return result

    metric_settings = dict(
        include_energy=loss_settings.energy_weight > 0.0,
        include_forces=loss_settings.forces_weight > 0.0,
        include_stress=loss_settings.stress_weight > 0.0,
        loss_label=loss_settings.loss_type,
    )
    epoch_args = SimpleNamespace(
        tqdm=bool(args.tqdm and tqdm is not None),
        print_freq=int(args.log_interval or 0),
        verbose=2 if args.log_interval else 1,
    )

    epoch_start = time.perf_counter()
    state, train_metrics_collection, _ = _run_train_epoch(
        state,
        train_loader,
        grad_step_fn,
        apply_updates_fn,
        max_steps=args.train_max_steps,
        multi_device=multi_device,
        use_ema=False,
        ema_decay=None,
        ema_count_start=0,
        logger=None,
        args=epoch_args,
        epoch=1,
        metric_settings=metric_settings,
        learning_rate=float(args.lr),
        mask=None,
    )
    run_time = time.perf_counter() - epoch_start

    graphs = float(train_metrics_collection.components["total"].count)
    batches = train_loader.batches
    throughput = graphs / run_time if run_time and graphs else 0.0
    mean_loss = (
        float(train_metrics_collection.components["total"].value)
        if train_metrics_collection.components["total"].count
        else float("nan")
    )

    pack_info = train_loader.pack_info()
    dropped = pack_info.get("dropped", 0)
    if dropped:
        print(f"[train] Dropped {dropped} graphs due to edge/node limits.")

    print(
        f"Train benchmark: {graphs:.0f} graphs across {batches} batches in {run_time:.3f}s => {throughput:.2f} graphs/s"
    )

    _write_results_csv(
        args.csv_output,
        {
            "backend": "jax",
            "mode": "train",
            "split": args.split,
            "dtype": args.dtype,
            "optimizer": args.opt,
            "learning_rate": args.lr,
            "graphs": graphs,
            "batches": batches,
            "run_time_s": run_time,
            "throughput_graphs_per_s": throughput,
            "compile_time_s": "",
            "mean_loss": mean_loss,
        },
    )


if __name__ == "__main__":
    main()
