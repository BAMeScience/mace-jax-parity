# mace-jax-parity

Utilities to load or download a Torch MACE foundation model, convert it to a JAX bundle, compare their energies on mp-traj, and benchmark throughput.

```
scripts/
├─ compare_mace_torch_jax.py      # parity check with progress bars and CSV logging
├─ benchmark_mace_torch.py        # Torch inference benchmark (Accelerate, CSV output)
├─ benchmark_mace_jax.py          # JAX benchmark (compile time + throughput)
├─ convert_mace_model_to_jax.py   # torch → JAX bundle converter
├─ create_mace_foundation_model.py
├─ plot_energy_diff.py            # CPU/GPU relative ΔE histograms (log scale)
```

Other important directories:

- `data/mptraj/` — contains `train.h5` / `valid.h5` subsets.
- `models/` — generated artifacts (`mace_foundation.pt`, `mace_jax_bundle/`, caches).
- `results/` — CSVs and plots emitted by the Makefile targets.
- `makefile` — orchestration for model creation, parity checks (float32/float64), benchmarks, and plotting.

## Quickstart
Assumes the virtualenv at `/home/pbenner/Env/mace-jax/.venv` is activated.

```bash
cd mace-jax-parity
source ../.venv/bin/activate

# Build Torch foundation model and JAX bundle.
make models/mace_jax_bundle

# Run comparisons (float32 + float64) and produce plots.
make compare
make plot-comparison

# Run benchmarks and write CSV summaries.
make benchmark

# Direct script usage examples:
python scripts/compare_mace_torch_jax.py \
  --torch-model models/mace_foundation.pt \
  --jax-model models/mace_jax_bundle \
  --data-dir data/mptraj \
  --split valid \
  --dtype float64 \
  --device cuda \
  --diff-csv results/custom_compare.csv \
  --tqdm

python scripts/benchmark_mace_jax.py \
  --torch-model models/mace_foundation.pt \
  --jax-model models/mace_jax_bundle \
  --data-dir data/mptraj \
  --split valid \
  --dtype float32 \
  --device cuda \
  --multi-gpu \
  --max-edges-per-batch 480000 \
  --csv-output results/benchmark_jax.csv
```

## Notes
- `make compare` now runs both float32 and float64 suites (`results/compare_*_f32.csv` / `_f64.csv`). Plotting creates separate figures per dtype.
- `compare_mace_torch_jax.py` forces both Torch and JAX onto the same device (`--device`) and shows tqdm for each backend. CSV output includes absolute and relative ΔE for further analysis.
- Benchmarks write machine-readable rows (backend, dtype, device, graphs/s, wall time, compile time). Torch uses Accelerate for multi-GPU, JAX reports XLA compile overhead.
- `plot_energy_diff.py` consumes the CSVs and produces log-scale histograms of relative |ΔE|/scale for CPU vs GPU comparisons; pass `--dtype` to annotate the plots.
