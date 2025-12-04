#!/usr/bin/env python3
"""
Inspect a serialized Torch MACE model and report whether cuEquivariance kernels
were enabled when the checkpoint was saved.

Usage:
    python scripts/check_cueq_enabled.py --model models/mace_foundation.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check if a Torch MACE checkpoint has cuEquivariance enabled."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the Torch model (.pt/.model) to inspect.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading parameters (default: cpu).",
    )
    return parser.parse_args()


def _extract_cue_config(model) -> object | None:
    """Recursively search for a cuEquivariance config on the model/wrapper."""
    for attr in ("cueq_config", "cue_config"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            return cfg
    module = getattr(model, "module", None)
    if module is not None and module is not model:
        return _extract_cue_config(module)
    return None


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model = torch.load(
        args.model,
        map_location=device,
        weights_only=False,
    )
    cue_cfg = _extract_cue_config(model)
    if cue_cfg is None:
        print(
            f"[INFO] No cuEquivariance configuration found on {args.model}. "
            "Model likely uses the standard e3nn kernels."
        )
        return
    enabled = getattr(cue_cfg, "enabled", False)
    print(f"[INFO] cuEquivariance config detected: {cue_cfg}")
    print(f"[RESULT] cuEquivariance enabled: {bool(enabled)}")


if __name__ == "__main__":
    main()
