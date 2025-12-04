#!/usr/bin/env python3
"""
Convert a serialized Torch MACE model into a bundle that can be consumed by
``mace-jax``.  The script mirrors the logic in mace_jax.cli.mace_torch2jax and
produces a parameter/state msgpack pair together with the sanitized model
configuration.

Example
-------
    python scripts/convert_mace_model_to_jax.py \\
        --torch-model models/mace_mp_medium.pt \\
        --output-dir models/mace_mp_medium_jax
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:  # pragma: no cover - defensive guard for torch<2.6
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover
    add_safe_globals = None

if callable(add_safe_globals):  # pragma: no cover
    add_safe_globals([slice])

try:  # pragma: no cover - optional dependency during conversions
    from e3nn_jax import Irreps as JaxIrreps
except ImportError:  # pragma: no cover
    JaxIrreps = None

try:  # pragma: no cover - fallback to torch e3nn if available
    from e3nn.o3 import Irreps as TorchIrreps
except ImportError:  # pragma: no cover
    TorchIrreps = None

from flax import serialization
from mace_jax.cli import mace_torch2jax

from mace.tools.scripts_utils import extract_config_mace_model


def _sanitize(obj):
    """
    Convert nested config structures to JSON-friendly representations.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover - best effort conversion
            pass
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)


_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "f64": torch.float64,
    "double": torch.float64,
}


def _resolve_dtype(spec: str | None) -> torch.dtype | None:
    if spec is None:
        return None
    key = spec.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype '{spec}'. Expected one of: {', '.join(sorted(set(_DTYPE_ALIASES)))}"
        )
    return _DTYPE_ALIASES[key]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Torch MACE model checkpoint to a MACE-JAX bundle."
    )
    parser.add_argument(
        "--torch-model",
        type=Path,
        required=True,
        help="Path to the serialized torch.nn.Module produced by Equitrain or the helper script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/mace_jax_bundle"),
        help="Directory where the params/state/config artifacts will be written.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional Torch dtype (e.g. float32, float64) to cast the model before conversion.",
    )
    return parser.parse_args()


def _format_irreps(value):
    if value is None or isinstance(value, str):
        return value

    constructors = [ctor for ctor in (JaxIrreps, TorchIrreps) if ctor is not None]
    for ctor in constructors:
        try:
            return str(ctor(value))
        except Exception:  # pragma: no cover - fall through to generic str
            continue

    return str(value)


def _as_cue_config_dict(cue_cfg) -> dict | None:
    if cue_cfg is None:
        return None
    if isinstance(cue_cfg, dict):
        config = dict(cue_cfg)
    else:
        config = {}
        for attr in (
            "enabled",
            "layout",
            "layout_str",
            "group",
            "optimize_all",
            "optimize_linear",
            "optimize_channelwise",
            "optimize_symmetric",
            "optimize_fctp",
            "conv_fusion",
        ):
            if hasattr(cue_cfg, attr):
                config[attr] = getattr(cue_cfg, attr)

    if not config:
        return None

    layout = config.get("layout_str") or config.get("layout") or "mul_ir"
    layout_name = layout if isinstance(layout, str) else getattr(layout, "name", "mul_ir")
    group_val = config.get("group", "O3")
    if hasattr(group_val, "__name__"):
        group_name = group_val.__name__.split(".")[-1]
    elif isinstance(group_val, str):
        group_name = group_val
    else:
        group_name = str(group_val)
    if group_name.endswith("O3_e3nn") or group_name == "O3_e3nn":
        group_name = "O3"

    return {
        "enabled": bool(config.get("enabled", False)),
        "layout": layout_name,
        "group": group_name,
        "optimize_all": bool(config.get("optimize_all", False)),
        "optimize_linear": bool(config.get("optimize_linear", False)),
        "optimize_channelwise": bool(config.get("optimize_channelwise", False)),
        "optimize_symmetric": bool(config.get("optimize_symmetric", False)),
        "optimize_fctp": bool(config.get("optimize_fctp", False)),
        "conv_fusion": bool(config.get("conv_fusion", False)),
    }


def main() -> None:
    args = _parse_args()
    target_dtype = _resolve_dtype(args.dtype)

    torch_model = torch.load(args.torch_model, map_location="cpu")
    if target_dtype is not None:
        torch_model = torch_model.to(dtype=target_dtype)
    torch_model = torch_model.to("cpu").eval()

    config = extract_config_mace_model(torch_model)
    cue_cfg = config.get("cueq_config") or getattr(torch_model, "cueq_config", None)
    config["cueq_config"] = _as_cue_config_dict(cue_cfg)
    config["torch_model_class"] = torch_model.__class__.__name__

    # Ensure irreps are serialized in the textual form expected by mace-jax.
    for key, value in list(config.items()):
        if "irreps" in key.lower():
            config[key] = _format_irreps(value)

    jax_model, jax_params, _template = mace_torch2jax.convert_model(torch_model, config)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "params.msgpack").write_bytes(serialization.to_bytes(jax_params))

    (output_dir / "config.json").write_text(json.dumps(_sanitize(config), indent=2))

    print(f"Exported MACE-JAX artifacts to {output_dir}")


if __name__ == "__main__":
    main()
