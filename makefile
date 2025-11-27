
all:
	@echo "Available targets:"
	@echo "  compare   - Compare MACE JAX model with MACE Torch model"
	@echo "  benchmark - Benchmark MACE JAX model against MACE Torch model"

compare: models/mace_jax_bundle
	python scripts/compare_mace_torch_jax.py --torch-model models/mace_foundation.pt --jax-model models/mace_jax_bundle/ --data-dir data/mptraj --dtype float32 --split valid

benchmark: models/mace_jax_bundle
	python scripts/benchmark_mace_torch_vs_jax.py --torch-model models/mace_foundation.pt --jax-model models/mace_jax_bundle --device cuda --max-edges-per-batch 60000 --max-nodes-per-batch 100000 --prefetch-batches 8 --multi-gpu

models/mace_jax_bundle: models/mace_foundation.pt
	rm -rf $@
	python scripts/convert_mace_model_to_jax.py --torch-model $< --output-dir $@

models/mace_foundation.pt:
	python scripts/create_mace_foundation_model.py --output $@
