#!/usr/bin/env python3
"""
Benchmark the inference throughput of a Torch MACE model on mp-traj data.
Supports multi-GPU execution via Hugging Face Accelerate (launch this script
with `accelerate launch` to spawn one process per GPU).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from tqdm import tqdm

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch import get_dataloader as get_dataloader_torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Torch MACE model on mp-traj data."
    )
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
        help="Batch size for the PyG DataLoader (default: 18).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Floating point precision to use (default: float32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device hint when running single-process benchmarks (cpu/cuda/cuda:0).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally cap the number of batches processed for quick smoke tests.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Deprecated flag kept for backwards compatibility (no-op).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="If set, append the final benchmark summary to this CSV file.",
    )
    return parser.parse_args()


def _resolve_device(spec: str) -> torch.device:
    text = spec.strip()
    lower = text.lower()
    if lower.startswith("gpu"):
        torch_spec = "cuda" + text[3:]
    else:
        torch_spec = text
    return torch.device(torch_spec)


def _load_torch_model(
    model_path: Path,
    dtype,
    device: torch.device,
) -> torch.nn.Module:
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    return model.to(device=device, dtype=dtype).eval()


def _list_h5_files(args) -> list[Path]:
    files = sorted(args.data_dir.glob("*.h5"))
    if args.split != "all":
        files = [p for p in files if p.stem == args.split]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.data_dir}")
    return files


def _benchmark_torch(
    accelerator: Accelerator,
    model,
    loaders: list[tuple[Path, object]],
    max_batches,
):
    model = accelerator.prepare(model)

    local_graphs = 0
    local_batches = 0

    with torch.no_grad():
        run_start = time.perf_counter()
        for h5_path, loader in loaders:
            prepared_loader = accelerator.prepare(loader)
            local_total = (
                len(prepared_loader) if hasattr(prepared_loader, "__len__") else None
            )
            pbar = (
                tqdm(total=local_total, desc=f"Torch {h5_path.name}", leave=True)
                if accelerator.is_main_process
                else None
            )

            for item in prepared_loader:
                batches = item if isinstance(item, list) else [item]
                for batch in batches:
                    if max_batches is not None and local_batches >= max_batches:
                        break
                    batch = batch.to(device=accelerator.device)
                    pred = model(
                        batch,
                        compute_force=False,
                        compute_stress=False,
                    )
                    energy = pred["energy"]
                    local_graphs += int(energy.shape[0])
                    local_batches += 1
                    if pbar is not None:
                        pbar.update(1)
                if max_batches is not None and local_batches >= max_batches:
                    break

            if pbar is not None:
                pbar.close()
            if max_batches is not None and local_batches >= max_batches:
                break

        run_time = time.perf_counter() - run_start

    graphs_tensor = torch.tensor(
        local_graphs, device=accelerator.device, dtype=torch.float64
    )
    batches_tensor = torch.tensor(
        local_batches, device=accelerator.device, dtype=torch.float64
    )

    total_graphs = accelerator.reduce(graphs_tensor, reduction="sum").item()
    total_batches = accelerator.reduce(batches_tensor, reduction="sum").item()

    return total_graphs, total_batches, run_time


def _write_results_csv(path: Path, row: dict):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    headers = [
        "backend",
        "split",
        "dtype",
        "device",
        "graphs",
        "batches",
        "run_time_s",
        "throughput_graphs_per_s",
        "compile_time_s",
    ]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = _parse_args()
    accelerator = Accelerator(split_batches=False)

    torch_dtype = getattr(torch, args.dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    requested_device = _resolve_device(args.device)
    if (
        accelerator.num_processes > 1
        and args.device.lower().startswith("cpu")
        and accelerator.is_main_process
    ):
        print(
            "[Torch] Warning: multi-process run requested but --device=cpu. "
            "Each process will use its assigned accelerator device instead."
        )

    load_device = (
        torch.device("cpu") if accelerator.num_processes > 1 else requested_device
    )
    model = _load_torch_model(args.torch_model, torch_dtype, load_device)

    if accelerator.num_processes == 1 and requested_device != load_device:
        model = model.to(device=requested_device)

    if not hasattr(model, "atomic_numbers") or not hasattr(model, "r_max"):
        raise RuntimeError(
            "Torch model must expose 'atomic_numbers' and 'r_max' attributes."
        )
    z_table = AtomicNumberTable([int(z) for z in getattr(model, "atomic_numbers")])
    r_max = float(getattr(model, "r_max"))

    h5_files = _list_h5_files(args)
    base_args = SimpleNamespace(
        pin_memory=False,
        num_workers=max(int(args.num_workers or 0), 0),
        batch_size=args.batch_size,
        shuffle=False,
        batch_max_nodes=None,
        batch_max_edges=None,
        batch_drop=False,
        niggli_reduce=False,
    )
    loaders = []
    for h5_path in h5_files:
        loader = get_dataloader_torch(
            base_args,
            h5_path,
            atomic_numbers=z_table,
            r_max=r_max,
            accelerator=accelerator,
        )
        if loader is None:
            continue
        loaders.append((h5_path, loader))
    if not loaders:
        raise RuntimeError("No Torch dataloaders could be built for the provided data")

    try:
        graphs, batches, run_time = _benchmark_torch(
            accelerator,
            model,
            loaders,
            args.max_batches,
        )

        if accelerator.is_main_process:
            throughput = graphs / run_time if graphs and run_time else 0.0
            print(
                f"Torch [{args.dtype}]: {graphs} graphs across {int(batches)} batches "
                f"in {run_time:.3f}s => "
                f"{throughput:.2f} graphs/s"
            )
            if args.csv_output is not None:
                _write_results_csv(
                    args.csv_output,
                    {
                        "backend": "torch",
                        "split": args.split,
                        "dtype": args.dtype,
                        "device": args.device,
                        "graphs": int(graphs),
                        "batches": int(batches),
                        "run_time_s": run_time,
                        "throughput_graphs_per_s": throughput,
                        "compile_time_s": "",
                    },
                )
            if args.multi_gpu and accelerator.num_processes == 1:
                print(
                    "[Torch] '--multi-gpu' was set but only one process/device is active. "
                    "Launch via 'accelerate launch' to use multiple GPUs."
                )
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
