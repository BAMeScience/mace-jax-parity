# mace-jax-parity

Utilities to load or download a Torch MACE foundation model, convert it to a JAX bundle, and compare/benchmark predictions on the mp-traj dataset. Scripts live in `scripts/`, data in `data/`, and generated models in `models/`.

## What’s here
- `scripts/create_mace_foundation_model.py` — download/load a MACE foundation checkpoint and save a Torch `nn.Module`.
- `scripts/convert_mace_model_to_jax.py` — turn the Torch checkpoint into a MACE-JAX bundle (`config.json` + `params.msgpack`).
- `scripts/compare_mace_torch_jax.py` — run mp-traj batches through both models and flag energy differences above a tolerance (tqdm progress, unified `--dtype`, default split `train`).
- `scripts/benchmark_mace_torch_vs_jax.py` — measure throughput for Torch vs. JAX (tqdm progress, unified dtype, JAX jit, reports compile and steady-state throughput).
- `data/mptraj/{train.h5,valid.h5}` — sample mp-traj splits.
- `models/` — artifacts produced by the helper scripts (`mace_foundation.pt`, `mace_jax_bundle/`).
- `makefile` — convenience targets to build the models and run the comparison.

## Quickstart
Assumes the local virtualenv at `/home/pbenner/Env/mace-jax/.venv` is available.

```bash
cd mace-jax-parity
source ../.venv/bin/activate

# Build Torch foundation model and JAX bundle, then compare on train split (dtype float32)
make

# Direct script usage (examples):
python scripts/compare_mace_torch_jax.py \
  --torch-model models/mace_foundation.pt \
  --jax-model models/mace_jax_bundle \
  --data-dir data/mptraj \
  --split train \
  --dtype float32 \
  --batch-size 4

python scripts/benchmark_mace_torch_vs_jax.py \
  --torch-model models/mace_foundation.pt \
  --jax-model models/mace_jax_bundle \
  --data-dir data/mptraj \
  --split valid \
  --dtype float32 \
  --batch-size 8 \
  --torch-device cuda \
  --jax-platform gpu
```

## Notes
- Dtype is unified per script via `--dtype` to avoid mixed-precision comparisons.
- GPU is supported: set `--torch-device cuda` and `--jax-platform gpu` (ensure CUDA-enabled `jaxlib` and drivers are available).
- Progress is shown with tqdm; warnings during comparison use `tqdm.write` to keep the progress bar clean.
