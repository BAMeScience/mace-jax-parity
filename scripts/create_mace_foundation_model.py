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

    if family == "mp":
        return loader(
            model=model,
            **loader_kwargs,
        )
    if family == "off":
        return loader(
            model=model,
            **loader_kwargs,
        )
    if family == "omol":
        return loader(
            model=model,
            **loader_kwargs,
        )

    # ANIcc helper accepts ``model_path`` instead of ``model``.
    return loader(
        model_path=model,
        **loader_kwargs,
    )


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

    model = _load_torch_model(
        family=args.family,
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        enable_cueq=bool(args.enable_cueq),
        only_cueq=bool(args.only_cueq),
    )
    model = model.to("cpu")
    model = model.float().eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)

    print(f"Serialized {args.family} foundation model to {args.output}")


if __name__ == "__main__":
    main()
