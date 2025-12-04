#!/usr/bin/env python3
"""
Utility script to materialize a Torch MACE model from one of the published
foundation checkpoints.  The resulting ``.pt`` file contains a regular
``torch.nn.Module`` that can be fed to Equitrain or converted to MACE-JAX.

Example
-------
    python scripts/create_mace_foundation_model.py \\
        --family mp \\
        --model medium-mpa-0 \\
        --output models/mace_mp_medium.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from mace.calculators import foundations_models

# Torch 2.6 tightened torch.load defaults; the checkpoints still store ``slice``.
try:  # pragma: no cover - defensive guard for torch<2.6
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover
    add_safe_globals = None

if callable(add_safe_globals):  # pragma: no cover - guard for torch<2.6
    add_safe_globals([slice])

FOUNDATION_LOADERS: dict[str, Callable] = {
    "mp": foundations_models.mace_mp,
    "off": foundations_models.mace_off,
    "anicc": foundations_models.mace_anicc,
    "omol": foundations_models.mace_omol,
}

_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "f64": torch.float64,
    "double": torch.float64,
}


def _load_torch_model(
    family: str,
    model: str | None,
    device: str,
    default_dtype: str,
    *,
    enable_cueq: bool,
    only_cueq: bool,
) -> torch.nn.Module:
    """
    Instantiate a Torch foundation model for the requested family.
    """

    loader = FOUNDATION_LOADERS[family]

    loader_kwargs = dict(
        device=device,
        default_dtype=default_dtype,
        return_raw_model=True,
    )
    if enable_cueq:
        loader_kwargs["enable_cueq"] = True
    if only_cueq:
        loader_kwargs["only_cueq"] = True

    if family == "anicc":
        # ANIcc helper accepts ``model_path`` instead of ``model``.
        return loader(
            model_path=model,
            **loader_kwargs,
        )
    else:
        return loader(
            model=model,
            **loader_kwargs,
        )


def _wrap_with_cueq(
    model: torch.nn.Module,
    *,
    enable_cueq: bool,
    only_cueq: bool,
    device: str,
) -> torch.nn.Module:
    """Convert the raw Torch model to its cue-equivariant counterpart when requested."""

    if not (enable_cueq or only_cueq):
        return model

    try:
        from mace.cli.convert_e3nn_cueq import run as convert_e3nn_to_cueq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cuequivariance is required to enable cuEq acceleration. "
            "Install the optional cue packages before passing --enable-cueq/--only-cueq."
        ) from exc

    device_str = device or "cpu"
    converted = convert_e3nn_to_cueq(model, device=device_str, return_model=True)
    return converted


def _resolve_dtype(spec: str | None) -> torch.dtype | None:
    if spec is None:
        return None
    key = spec.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype '{spec}'. Expected one of: {', '.join(sorted(_DTYPE_ALIASES))}"
        )
    return _DTYPE_ALIASES[key]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or load a MACE foundation checkpoint and serialize the "
            "raw torch.nn.Module to disk."
        )
    )
    parser.add_argument(
        "--family",
        choices=tuple(FOUNDATION_LOADERS),
        default="mp",
        help="Foundation family to load (default: mp).",
    )
    parser.add_argument(
        "--model",
        default="medium-mpa-0",
        help=(
            "Model identifier or local checkpoint path passed to the MACE "
            "foundation loader. Family specific defaults apply."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while materializing the Torch model (default: cpu).",
    )
    parser.add_argument(
        "--default-dtype",
        default="float64",
        help="Default floating point precision requested from the loader.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mace_foundation.pt"),
        help="Destination file for the serialized torch.nn.Module.",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        default=None,
        help="Optional dtype (float32/float64) to cast the serialized Torch model to.",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        help="Ask the foundation loader to enable cuEquivariance acceleration.",
    )
    parser.add_argument(
        "--only-cueq",
        action="store_true",
        help="Restrict the model to cueEquivariance kernels only (where supported).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    target_dtype = _resolve_dtype(args.output_dtype)

    model = _load_torch_model(
        family=args.family,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        enable_cueq=bool(args.enable_cueq),
        only_cueq=bool(args.only_cueq),
    )
    model = _wrap_with_cueq(
        model,
        enable_cueq=bool(args.enable_cueq),
        only_cueq=bool(args.only_cueq),
        device=args.device,
    )
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
    model = model.to("cpu").eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)

    print(f"Serialized {args.family} foundation model to {args.output}")


if __name__ == "__main__":
    main()
