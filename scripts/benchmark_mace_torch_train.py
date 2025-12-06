#!/usr/bin/env python3
"""Benchmark a single training epoch for a Torch MACE model."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm  # noqa: F401 (used inside equitrain's train_one_epoch)

try:
    from torch.serialization import add_safe_globals as _torch_add_safe_globals
except ImportError:  # pragma: no cover
    _torch_add_safe_globals = None

if callable(_torch_add_safe_globals):  # pragma: no cover
    _torch_add_safe_globals([slice])

from equitrain.argparser import get_args_parser_train, get_loss_monitor
from equitrain.backends.torch_backend import train_one_epoch
from equitrain.backends.torch_optimizer import create_optimizer
from equitrain.backends.torch_wrappers.mace import MaceWrapper as TorchMaceWrapper
from equitrain.backends.torch_utils import set_dtype as set_torch_default_dtype
from equitrain.backends.torch_utils import set_seeds
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.loaders import get_dataloader as get_dataloader_torch


class _BenchmarkLogger:
    """Minimal logger that mimics the Equitrain interface."""

    def __init__(self, enabled: bool):
        self._enabled = enabled

    def log(self, level: int, message: str, force: bool = False) -> None:
        if self._enabled or force:
            print(message)


class _CountingLoader:
    """Thin wrapper that counts how many macro-batches were yielded."""

    def __init__(self, loader):
        self._loader = loader
        self.batches = 0

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __iter__(self):
        for item in self._loader:
            self.batches += 1
            yield item


def _resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported dtype '{name}'. Expected one of {tuple(mapping)}."
        )


def _augment_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(tqdm=True, backend="torch")
    parser.add_argument(
        "--torch-model",
        type=Path,
        required=True,
        help="Path to the serialized Torch MACE model (.pt/.model).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mptraj"),
        help="Directory with mp-traj *.h5 files (default: data/mptraj).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default="train",
        help="Which mp-traj split to use for training (default: train).",
    )
    parser.add_argument(
        "--valid-split",
        choices=("train", "valid"),
        default="valid",
        help="Split used for validation loader bookkeeping (default: valid).",
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
        help="Stop training after this many macro-batches (alias for --train-max-steps).",
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
        help="Print step-level stats every N updates (overrides --print-freq).",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Reserved for CLI compatibility (Accelerate manages multi-GPU execution).",
    )
    return parser


def _select_split_file(data_dir: Path, split: str) -> Path:
    candidate = data_dir / f"{split}.h5"
    if not candidate.exists():
        raise FileNotFoundError(f"No HDF5 file found for split '{split}': {candidate}")
    return candidate


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_dir = Path(args.data_dir)
    args.torch_model = Path(args.torch_model)
    args.model = str(args.torch_model)
    if not getattr(args, "train_file", None):
        args.train_file = str(_select_split_file(args.data_dir, args.split))
    if not getattr(args, "valid_file", None) and args.valid_split:
        try:
            args.valid_file = str(_select_split_file(args.data_dir, args.valid_split))
        except FileNotFoundError:
            args.valid_file = None
    if not getattr(args, "output_dir", None):
        args.output_dir = "results/torch-train-benchmark"
    if args.max_batches is not None:
        if getattr(args, "train_max_steps", None) is None:
            args.train_max_steps = args.max_batches
        else:
            args.train_max_steps = min(int(args.train_max_steps), int(args.max_batches))
    # mp-traj benchmarks do not provide consistent forces/stress labels; disable by default.
    args.forces_weight = 0.0
    args.stress_weight = 0.0
    return args


def _resolve_atomic_numbers(model) -> AtomicNumberTable:
    if hasattr(model, "atomic_numbers"):
        numbers = getattr(model, "atomic_numbers")
        if isinstance(numbers, AtomicNumberTable):
            return numbers
        if isinstance(numbers, torch.Tensor):
            return AtomicNumberTable([int(z) for z in numbers.detach().cpu().tolist()])
        if hasattr(numbers, "zs"):
            return AtomicNumberTable(list(numbers.zs))
        if isinstance(numbers, (list, tuple)):
            return AtomicNumberTable(numbers)
    if hasattr(model, "config"):
        config = getattr(model, "config")
        if isinstance(config, dict) and "atomic_numbers" in config:
            return AtomicNumberTable(config["atomic_numbers"])
    raise RuntimeError("Unable to infer atomic numbers from the Torch model.")


def _resolve_r_max(model) -> float:
    if hasattr(model, "r_max"):
        return float(getattr(model, "r_max"))
    if hasattr(model, "config"):
        config = getattr(model, "config")
        if isinstance(config, dict) and "r_max" in config:
            return float(config["r_max"])
    raise RuntimeError("Torch model is missing an r_max attribute.")


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

    set_seeds(args.seed)
    set_torch_default_dtype(args.dtype)
    torch_dtype = _resolve_torch_dtype(args.dtype)

    args.loss_monitor = get_loss_monitor(args)
    if getattr(args, "loss_type_energy", None) is None:
        args.loss_type_energy = args.loss_type.lower()
    if getattr(args, "loss_type_forces", None) is None:
        args.loss_type_forces = args.loss_type.lower()
    if getattr(args, "loss_type_stress", None) is None:
        args.loss_type_stress = args.loss_type.lower()
    if (
        args.loss_type_energy != args.loss_type.lower()
        or args.loss_type_forces != args.loss_type.lower()
        or args.loss_type_stress != args.loss_type.lower()
    ):
        args.loss_type = "mixed"

    if args.log_interval:
        args.print_freq = int(args.log_interval)
        args.verbose = 2
    else:
        args.print_freq = 0
        args.verbose = 1

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(
            args.energy_weight == 0.0
            or getattr(args, "freeze_params", None)
            or getattr(args, "unfreeze_params", None)
            or getattr(args, "find_unused_parameters", False)
        )
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        log_with=None,
    )
    logger = _BenchmarkLogger(accelerator.is_main_process)

    model = torch.load(
        args.torch_model,
        map_location="cpu",
        weights_only=False,
    )
    model = model.to(dtype=torch_dtype)
    model = TorchMaceWrapper(args, model)

    atomic_numbers = _resolve_atomic_numbers(model)
    r_max = _resolve_r_max(model)

    train_loader = get_dataloader_torch(
        args,
        args.train_file,
        atomic_numbers,
        r_max,
        accelerator=None,
    )
    optimizer = create_optimizer(args, model)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    counting_loader = _CountingLoader(train_loader)

    errors = None

    try:
        epoch_start = time.perf_counter()
        train_metrics, errors = train_one_epoch(
            args=args,
            model=model,
            model_ema=None,
            accelerator=accelerator,
            dataloader=counting_loader,
            optimizer=optimizer,
            errors=errors,
            epoch=1,
            logger=logger,
        )
        run_time = time.perf_counter() - epoch_start

        accelerator.wait_for_everyone()

        graphs = float(train_metrics.main["total"].count)

        local_batches = torch.tensor(
            float(counting_loader.batches),
            device=accelerator.device,
            dtype=torch.float64,
        )
        total_batches = accelerator.reduce(local_batches, reduction="sum").item()

        run_time_tensor = torch.tensor(
            float(run_time), device=accelerator.device, dtype=torch.float64
        )
        total_time = accelerator.reduce(run_time_tensor, reduction="max").item()
        throughput = graphs / total_time if graphs and total_time else 0.0
        mean_loss = float(train_metrics.main["total"].avg)

        if accelerator.is_main_process:
            print(
                f"Torch train benchmark: {total_graphs:.0f} graphs across {int(total_batches)} "
                f"batches in {total_time:.3f}s => {throughput:.2f} graphs/s"
            )
            _write_results_csv(
                args.csv_output,
                {
                    "backend": "torch",
                    "mode": "train",
                    "split": args.split,
                    "dtype": args.dtype,
                    "optimizer": args.opt,
                    "learning_rate": args.lr,
                    "graphs": graphs,
                    "batches": total_batches,
                    "run_time_s": total_time,
                    "throughput_graphs_per_s": throughput,
                    "compile_time_s": "",
                    "mean_loss": mean_loss,
                },
            )
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
