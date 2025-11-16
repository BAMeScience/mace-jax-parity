
all: models/mace_jax_bundle
	python scripts/compare_mace_torch_jax.py --torch-model models/mace_foundation.pt --jax-model models/mace_jax_bundle/ --data-dir data/mptraj --torch-dtype float32

models/mace_jax_bundle: models/mace_foundation.pt
	rm -rf $@
	python scripts/convert_mace_model_to_jax.py --torch-model $< --output-dir $@

models/mace_foundation.pt:
	python scripts/create_mace_foundation_model.py --output $@
