# REViT — Roto-reflection Equivariant convolutional Vision Transformer
**PAPER:** [![arXiv](https://img.shields.io/badge/arXiv-2606.25318-b31b1b.svg)](https://arxiv.org/abs/2606.25318)

REViT is a research codebase for rotation/roto-reflection equivariant vision transformers. It contains:

- a budget-adaptive REViT with nested widths and one invariant decision per image,
- an equivariant transformer baseline (`Rot2DTransformer`) for small image datasets,
- a windowed hierarchical equivariant transformer (`Rot2DTransformerV2`) for ImageNet-scale training.

## Repository layout

### Core models

- `budget_revit.py`: shared lifting stem, nested steerable convolutions, global and windowed profiles, invariant controller, and fixed-profile export.
- `budget_cli.py`: shared adaptive flags, config serialization, and checkpoint loading.
- `revit_gcsa.py`: original/global-attention equivariant transformer (`Rot2DTransformer`).
- `revit_windowed_gcsa.py`: windowed hierarchical equivariant transformer (`Rot2DTransformerV2`).
- `group_space.py`: maps group strings (`C4`, `D4`, etc.) to ESCNN `gspaces`.

### Training scripts

- `train_revit.py`: global or adaptive training entrypoint for `rotmnist`, `cifar10`, and `pcam` datasets.
- `imagenet_train_revit.py`: ImageNet training for the original and adaptive windowed models (DDP, AMP, Mixup/CutMix, checkpoint resume).
- `imagenet_train_vit.py`: ImageNet training for a vanilla `torchvision` ViT-Small baseline.

### Dataset utilities

- `datasets/rot_mnist.py`: rotated MNIST dataset class and dataloaders.
- `datasets/cifar10.py`: CIFAR-10 dataloaders and augmentation.
- `datasets/pcam.py`: PCam dataloaders.

### Experiments and utilities

- `scripts/run_budget.py`: evaluate budgets and controls, benchmark end-to-end forwards, and export fixed profiles.
- `tests/test_budget_revit.py`: numerical symmetry, real operator shapes, training, and checkpoint tests.
- `utils.py`: standalone quarter-turn augmentation and early-stopping utilities.
- `scripts/cleanup_checkpoints.py`: keep only the newest N checkpoint files in a directory.
- `scripts/benchmark_sdpa.py`: compare bmm vs SDPA attention paths for latency/memory.
- `requirements.txt` / `requirements-lock.txt`: install deps and optional pinned lockfile.

## Supported symmetry groups

You can pass `--group-str` as one of:

`Z2`, `D2`, `C4`, `C8`, `C12`, `C16`, `D4`, `D8`, `D12`, `D16`. The adaptive backbone supports C4 and D4, plus the trivial-group `Z2` control.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Use a torch/torchvision build matching your CUDA or CPU setup:
# https://pytorch.org/get-started/locally/
pip install torch torchvision

# Remaining dependencies
pip install -r requirements.txt
```

## Budget-adaptive REViT

`--budget-profiles` enables the adaptive implementation. Use semicolons between profiles and commas between stages, with widths in **regular fields**. For C4 and D4, each field has four and eight scalar channels respectively. Every stage width must be divisible by its head count. Profiles must be componentwise nested and differ in at least one stage. A single profile trains a matched fixed-width control. Use a stable `--budget-stem-width` across independent controls.

At inference, `--budget 0` admits only the cheapest profile and `--budget 1` admits all available profiles. Intermediate budgets cap normalized post-stem MACs by default. `--profile-costs` accepts increasing measured costs in a common unit for hardware-specific calibration. The controller still chooses among the eligible profiles. During training, the budget is sampled at profile thresholds and every profile receives a supervised update.

### Global CIFAR-10 example

```bash
python train_revit.py --dataset cifar10 --validation-fraction 0.1 \
  --group-str C4 --budget-profiles '2;4;8' --heads 2 --depths 2 \
  --downsize 4 --conv-kernel-size 3 --epochs 100 --no-one-cycle \
  --warmup-epochs 5 --output-dir runs_adaptive/cifar10
```

Use `--window-size 7` for window attention. The small-data trainer accepts matching multi-stage `--heads` and `--depths` lists. It uses the supplied validation split for PCam. For CIFAR-10 and Rotated MNIST, pass `--validation-fraction` to reserve train examples for model selection ahead of the official test evaluation. The loaders cache these datasets under `--data-root` or their default `data/` paths. For PCam, place the `PCam/train`, `PCam/valid`, and `PCam/test` ImageFolder splits under `--data-root`.

### Windowed ImageNet example

```bash
python imagenet_train_revit.py --data-root /path/to/imagenet --group-str D4 \
  --budget-profiles '4,8,16,32;8,16,32,64;12,24,48,96' \
  --heads 1 2 4 8 --depths 1 2 3 1 --qkv-kernel-size 1 \
  --batch-size 32 --output-dir runs_adaptive/imagenet
```

Each training step evaluates all profiles, so size the per-device batch accordingly. The budget backbone uses deterministic blocks and separate normalization statistics per profile. Its geometry uses centered downsampling and symmetric window padding. Use the same adaptive architecture with a singleton profile for a geometry-matched fixed baseline. Set `--group-str Z2` for the trivial-group adaptive control and tune its widths to match measured latency.

### Python inference and fixed export

```python
from budget_cli import load_budget_model
import torch

model = load_budget_model("checkpoint.pt", "cpu").eval()
images = torch.rand(2, 3, 32, 32)  # Match the checkpoint's data preprocessing.
with torch.inference_mode():
    logits, routes = model(images, budget=torch.tensor([0.0, 1.0]), return_routes=True)
    fixed = model.export_profile(0)
    narrow_logits = fixed(images)
```

### Evaluate, benchmark, and export

```bash
python -m scripts.run_budget evaluate checkpoint.pt --dataset cifar10 \
  --budgets 0 0.5 1 --check-group
python -m scripts.run_budget benchmark --checkpoint checkpoint.pt --dataset cifar10 \
  --batch-sizes 1 16 --iterations 100 --prewarm-all-profiles
python -m scripts.run_budget export checkpoint.pt 0 fixed_small.pt
python -m scripts.run_budget export independently_trained_wide.pt 0 pruned.pt --widths 2
```

Use `export --widths` to prune a source profile to whole-field prefixes and initialize a static fine-tuning run. Train the wide source with `--budget-stem-width` at or below the planned narrow first-stage width, then keep that stem width during fine-tuning. Load the exported weights with `--pretrained-path` in the small-data trainer or `--pretrained --pretrained-weights-only --pretrained-path` in the ImageNet trainer. `evaluate` reports fixed and adaptive accuracy, route histograms, shuffled and oracle controls, and optional transformed-input checks. For PCam, `--pcam-split test` targets the held-out test folder and reports AUROC. `benchmark` reports synchronized forward timings, throughput and CUDA allocator peaks, with optional Chrome traces. `--compile-profiles` enables `torch.compile` for the post-stem path on a supported backend. On C4, `--pointwise-backend fourier` selects the unfused Fourier reference in FP32. Use `benchmark --smoke-backbone global` or `benchmark --smoke-backbone windowed` for a synthetic smoke run with random weights.

```bash
python -m pip install pytest
python -m pytest -q tests/test_budget_revit.py
```

## Legacy REViT usage

### 1) Small dataset training (`train_revit.py`)

`train_revit.py --help` lists the small-dataset flags. Running without flags uses the original Rotated MNIST preset.

Supported dataset values:

- `rotmnist`
- `cifar10`
- `pcam`

Run:

```bash
python train_revit.py
```

Outputs are saved to:

- `runs/` (TensorBoard logs),
- `checkpoints/` (best checkpoint dict with optimizer/scheduler),
- `es_models/` (best model state dict).

### 2) ImageNet training — REViT v2

Expected data layout:

```text
/path/to/imagenet/
  train/
    class_000/
    ...
  val/
    class_000/
    ...
```

Single GPU:

```bash
python imagenet_train_revit.py \
  --data-root /path/to/imagenet \
  --model-size small \
  --group-str D4 \
  --batch-size 128 \
  --epochs 300 \
  --warmup-epochs 20 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --drop-path-rate 0.2 \
  --qkv-kernel-size 3 \
  --window-size 7
```

Multi GPU (DDP):

```bash
torchrun --standalone --nproc_per_node=4 imagenet_train_revit.py \
  --ddp \
  --data-root /path/to/imagenet \
  --model-size small \
  --group-str D4 \
  --batch-size 128 \
  --epochs 300 \
  --warmup-epochs 20 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --drop-path-rate 0.2 \
  --qkv-kernel-size 3 \
  --window-size 7 \
  --num-workers 8
```

Resume from checkpoint:

```bash
python imagenet_train_revit.py \
  --data-root /path/to/imagenet \
  --pretrained \
  --pretrained-path /path/to/checkpoint.pt
```

Use `--pretrained-weights-only` to load only model weights and start optimizer/scheduler fresh.

### 3) ImageNet baseline — vanilla ViT-Small

```bash
torchrun --standalone --nproc_per_node=4 imagenet_train_vit.py \
  --ddp \
  --data-root /path/to/imagenet \
  --batch-size 128 \
  --epochs 300 \
  --warmup-epochs 20 \
  --lr 3e-4 \
  --weight-decay 0.05
```

## Model presets in `imagenet_train_revit.py`

- `tiny` (default): dims `12,24,48,96`; depths `1,2,3,1`; heads `1,2,4,8`
- `small`: dims `24,48,96,192`; depths `1,2,4,1`; heads `1,2,4,8`
- `base`: dims `32,64,128,256`; depths `2,2,4,2`; heads `1,2,4,8`
- `large`: dims `64,128,192,384`; depths `2,2,6,2`; heads `2,4,8,16`

You can override with `--dims`, `--depths`, and `--heads`.

## Monitoring

TensorBoard:

```bash
# small dataset runs
tensorboard --logdir runs

# ImageNet REViT runs
tensorboard --logdir imagenet_es_v2_outputs/runs

# ImageNet ViT baseline runs
tensorboard --logdir imagenet_vit_small_outputs/runs
```

## Model Zoo

Pretrained model weights are available [here](https://drive.google.com/drive/folders/1iulEWyquORbqyRt0sNTawzFB57GbrhQu?usp=sharing).

## Citation

```tex
@inproceedings{
zaheer2026revit,
title={{REV}iT: Roto-reflection Equivariant Convolutional Vision Transformer},
author={Sheir A. Zaheer and Alexander C. Holston and Chan Y. Park},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=n2RIMdIbv6}
}
```

## License

See `LICENSE`.
