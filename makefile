
TORCH_F32 := models/mace_foundation_f32.pt
TORCH_F64 := models/mace_foundation_f64.pt
JAX_F32 := models/mace_jax_bundle_f32
JAX_F64 := models/mace_jax_bundle_f64
FOUNDATION_FAMILY := mp
FOUNDATION_MODEL := medium-mpa-0
FOUNDATION_NAME := $(FOUNDATION_FAMILY)_$(FOUNDATION_MODEL)

all:
	@echo "Available targets:"
	@echo "  compare   - Compare MACE JAX model with MACE Torch model"
	@echo "  benchmark - Benchmark MACE JAX model against MACE Torch model"
	@echo "  plot      - Plot CPU/GPU energy difference histograms"

################################################################################

compare: compare32 compare64
compare32: compare32-cpu compare32-gpu
compare64: compare64-cpu compare64-gpu

compare32-cpu: $(JAX_F32)
	mkdir -p results
	python scripts/compare_mace_torch_jax.py --torch-model $(TORCH_F32) --jax-model $(JAX_F32) --data-dir data/mptraj --dtype float32 --split valid --device cpu --num-workers 5 --diff-csv results/compare_cpu_f32.csv

compare32-gpu: $(JAX_F32)
	mkdir -p results
	python scripts/compare_mace_torch_jax.py --torch-model $(TORCH_F32) --jax-model $(JAX_F32) --data-dir data/mptraj --dtype float32 --split valid --device cuda --num-workers 5 --diff-csv results/compare_gpu_f32.csv

compare64-cpu: $(JAX_F64)
	mkdir -p results
	python scripts/compare_mace_torch_jax.py --torch-model $(TORCH_F64) --jax-model $(JAX_F64) --data-dir data/mptraj --dtype float64 --split valid --device cpu --max-edges-per-batch 240000 --max-nodes-per-batch 100000 --num-workers 5 --diff-csv results/compare_cpu_f64.csv

compare64-gpu: $(JAX_F64)
	mkdir -p results
	python scripts/compare_mace_torch_jax.py --torch-model $(TORCH_F64) --jax-model $(JAX_F64) --data-dir data/mptraj --dtype float64 --split valid --device cuda --max-edges-per-batch 240000 --max-nodes-per-batch 100000 --num-workers 5 --diff-csv results/compare_gpu_f64.csv

################################################################################

benchmark: benchmark-torch-predict benchmark-jax-predict benchmark-jax-train benchmark-torch-train

benchmark-torch-predict: $(TORCH_F32)
	mkdir -p results
	accelerate launch scripts/benchmark_mace_torch_predict.py --torch-model $(TORCH_F32) --data-dir data/mptraj --split valid --batch-size 18 --dtype float32 --device cuda --num-workers 8 --csv-output results/benchmark_torch.csv

benchmark-jax-predict: $(JAX_F32)
	mkdir -p results
	python scripts/benchmark_mace_jax_predict.py --torch-model $(TORCH_F32) --jax-model $(JAX_F32) --data-dir data/mptraj --split valid --dtype float32 --device cuda --max-edges-per-batch 480000 --num-workers 24 --multi-gpu --csv-output results/benchmark_jax.csv

benchmark-jax-train: $(JAX_F32)
	mkdir -p results
	python scripts/benchmark_mace_jax_train.py --jax-model $(JAX_F32) --data-dir data/mptraj --split valid --dtype float32 --learning-rate 1e-3 --max-edges-per-batch 280000 --prefetch-batches 24 --multi-gpu --tqdm --csv-output results/benchmark_jax_train.csv

benchmark-torch-train: $(TORCH_F32)
	mkdir -p results
	accelerate launch scripts/benchmark_mace_torch_train.py --torch-model $(TORCH_F32) --data-dir data/mptraj --split valid --dtype float32 --batch-size 80 --learning-rate 1e-3 --num-workers 24 --tqdm --csv-output results/benchmark_torch_train.csv

plot-comparison:
	mkdir -p results
	python scripts/plot_energy_diff.py --cpu-csv results/compare_cpu_f32.csv --gpu-csv results/compare_gpu_f32.csv --out results/energy_diff_f32.png --dtype float32 --model-name $(FOUNDATION_NAME)
	python scripts/plot_energy_diff.py --cpu-csv results/compare_cpu_f64.csv --gpu-csv results/compare_gpu_f64.csv --out results/energy_diff_f64.png --dtype float64 --model-name $(FOUNDATION_NAME)

################################################################################

$(JAX_F32): $(TORCH_F32)
	rm -rf $@
	python scripts/convert_mace_model_to_jax.py --torch-model $< --output-dir $@ --dtype float32

$(JAX_F64): $(TORCH_F64)
	rm -rf $@
	python scripts/convert_mace_model_to_jax.py --torch-model $< --output-dir $@ --dtype float64

$(TORCH_F32):
	python scripts/create_mace_foundation_model.py --output $@ --default-dtype float32 --output-dtype float32 --family $(FOUNDATION_FAMILY) --model $(FOUNDATION_MODEL) --enable-cueq --only-cueq

$(TORCH_F64):
	python scripts/create_mace_foundation_model.py --output $@ --default-dtype float64 --output-dtype float64 --family $(FOUNDATION_FAMILY) --model $(FOUNDATION_MODEL) --enable-cueq --only-cueq
