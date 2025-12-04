#!/usr/bin/env python3
"""
Inspect a MACE-JAX bundle (config.json + params.msgpack) and report whether
cuEquivariance acceleration was enabled when the model was exported.

Usage:
    python scripts/check_jax_cueq.py --bundle models/mace_jax_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check if a MACE-JAX bundle was exported with cuEquivariance enabled."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to the bundle directory or config.json file.",
    )
    return parser.parse_args()


def _resolve_config_path(bundle_path: Path) -> Path:
    if bundle_path.is_dir():
        return bundle_path / "config.json"
    if bundle_path.suffix == ".json":
        return bundle_path
    raise FileNotFoundError(
        f"Expected a directory containing config.json or a JSON file, got: {bundle_path}"
    )


def main() -> None:
    args = _parse_args()
    config_path = _resolve_config_path(args.bundle)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {config_path}")

    config = json.loads(config_path.read_text())
    cue_cfg = config.get("cueq_config") or config.get("cue_config")
    if cue_cfg in (None, "None"):
        print(f"[INFO] No cuEquivariance configuration stored in {config_path}.")
        print("[RESULT] cuEquivariance enabled: False")
        return

    enabled = False
    if isinstance(cue_cfg, dict):
        enabled = bool(cue_cfg.get("enabled", False))
    elif isinstance(cue_cfg, bool):
        enabled = cue_cfg

    print(f"[INFO] cuEquivariance config entry: {cue_cfg}")
    print(f"[RESULT] cuEquivariance enabled: {enabled}")


if __name__ == "__main__":
    main()
