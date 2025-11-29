
all:
	@echo "Available targets:"
	@echo "  compare   - Compare MACE JAX model with MACE Torch model"
	@echo "  benchmark - Benchmark MACE JAX model against MACE Torch model"
	@echo "  plot      - Plot CPU/GPU energy difference histograms"

################################################################################

compare: compare-cpu compare-gpu
plot: plot-energy

compare-cpu: models/mace_jax_bundle
	mkdir -p results
	python scripts/compare_mace_torch_jax.py --torch-model models/mace_foundation.pt --jax-model models/mace_jax_bundle/ --data-dir data/mptraj --dtype float32 --split valid --device cpu --diff-csv results/compare_cpu.csv

compare-gpu: models/mace_jax_bundle
	mkdir -p results
	python scripts/compare_mace_torch_jax.py --torch-model models/mace_foundation.pt --jax-model models/mace_jax_bundle/ --data-dir data/mptraj --dtype float32 --split valid --device cuda --diff-csv results/compare_gpu.csv

################################################################################

benchmark: benchmark-torch benchmark-jax

benchmark-torch: models/mace_foundation.pt
	mkdir -p results
	accelerate launch scripts/benchmark_mace_torch.py --torch-model models/mace_foundation.pt --data-dir data/mptraj --split valid --batch-size 18 --dtype float32 --device cuda --num-workers 8 --csv-output results/benchmark_torch.csv

benchmark-jax: models/mace_jax_bundle
	mkdir -p results
	python scripts/benchmark_mace_jax.py --torch-model models/mace_foundation.pt --jax-model models/mace_jax_bundle --data-dir data/mptraj --split valid --dtype float32 --device cuda --max-edges-per-batch 480000 --max-nodes-per-batch 200000 --num-workers 24 --multi-gpu --csv-output results/benchmark_jax.csv

plot-comparison:
	mkdir -p results
	python scripts/plot_energy_diff.py --cpu-csv results/compare_cpu.csv --gpu-csv results/compare_gpu.csv --out results/energy_diff.png

################################################################################

models/mace_jax_bundle: models/mace_foundation.pt
	rm -rf $@
	python scripts/convert_mace_model_to_jax.py --torch-model $< --output-dir $@

models/mace_foundation.pt:
	python scripts/create_mace_foundation_model.py --output $@
