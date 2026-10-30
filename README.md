# REViT — Roto-reflection Equivariant convolutional Vision Transformer
**PAPER:** [![arXiv](https://img.shields.io/badge/arXiv-2606.25318-b31b1b.svg)](https://arxiv.org/abs/2606.25318)

REViT is a research codebase for rotation/roto-reflection equivariant vision transformers. It contains:

- an equivariant transformer baseline (`Rot2DTransformer`) for small image datasets,
- a windowed hierarchical equivariant transformer (`Rot2DTransformerV2`) for ImageNet-scale training.

## Repository layout

### Core models

- `revit_gcsa.py`: original/global-attention equivariant transformer (`Rot2DTransformer`).
- `revit_windowed_gcsa.py`: windowed hierarchical equivariant transformer (`Rot2DTransformerV2`).
- `group_space.py`: maps group strings (`C4`, `D4`, etc.) to ESCNN `gspaces`.

### Training scripts

- `train_revit.py`: training entrypoint for `rotmnist`, `cifar10`, and `pcam` datasets.
- `imagenet_train_revit.py`: ImageNet training for `Rot2DTransformerV2` (DDP, AMP, Mixup/CutMix, checkpoint resume).
- `imagenet_train_vit.py`: ImageNet training for a vanilla `torchvision` ViT-Small baseline.

### Dataset utilities

- `datasets/rot_mnist.py`: rotated MNIST dataset class and dataloaders.
- `datasets/cifar10.py`: CIFAR-10 dataloaders and augmentation.
- `datasets/pcam.py`: PCam dataloaders.

### Misc

- `scripts/cleanup_checkpoints.py`: keep only the newest N checkpoint files in a directory.
- `scripts/benchmark_sdpa.py`: compare bmm vs SDPA attention paths for latency/memory.
- `requirements.txt` / `requirements-lock.txt`: install deps and optional pinned lockfile.

## Supported symmetry groups

You can pass `--group-str` as one of:

`Z2`, `D2`, `C4`, `C8`, `C12`, `C16`, `D4`, `D8`, `D12`, `D16`.

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

## Usage

### 1) Small dataset training (`train_revit.py`)

`train_revit.py` is configured via constants in its `if __name__ == "__main__":` block (there is no argparse CLI yet).

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

`--epochs` is the total target, including completed epochs when resuming. New
checkpoints store optimizer, scheduler, scaler, configuration and per-rank RNG
state. `checkpoints/best.pt` keeps the best model; `last.pt` is saved every
`--checkpoint-interval` epochs (default 10) and at the final epoch. Set the
interval to 0 for best-only saving. Resume with the original total epoch count
and model configuration. Old checkpoint dictionaries are supported; use
`imagenet_utils.load_model_state()` when loading raw exported model states.

Execution options include `--no-fused-qkv`, `--attention-backend bmm|sdpa`,
`--amp-dtype float16|bfloat16`, `--fused-adamw`, and `--log-interval 50`.
The defaults retain bmm and FP16; benchmark optional backends on the target
hardware. `--image-size` and `--stem-downsample` expose resolution experiments.
Stage dimensions count **regular fields**: actual channels equal the field
count times group order (D4 has order 8). Geometry and width changes can affect
accuracy and equivariance; see the analysis below.

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

## Scaling analysis and optimization verification

- [ImageNet architecture, parameter/FLOP scaling and equivariance limits](analysis/IMAGENET_ANALYSIS.txt)
- [Applied changes, CPU timings, tests and accuracy-validation limits](analysis/OPTIMIZATION_REPORT.txt)
- [Reproduction commands and environment notes](analysis/REPRODUCTION.txt)

Generate per-image estimates (2 FLOPs per multiply-add), optionally building
models for exact constrained parameter counts:

```bash
python scripts/profile_scaling.py --model-sizes tiny small --groups C4 D4 \
  --image-sizes 224 448 --kernel-sizes 1 3 --exact-parameters \
  --output analysis/scaling_results.json

python scripts/benchmark_optimizations.py --baseline-ref 2558962 \
  --model-size tiny --group D4 --qkv-kernel-size 3 --device cuda \
  --dtype float16 --batch-size 32 --output benchmark_results/optimization.json

pip install pytest
python -m pytest -q
```

FLOP estimates count dense convolutions, linears and attention matrix products;
they exclude kernel-basis expansion, softmax, backward and optimizer work.
CPU timing evidence is included under `analysis/`; CUDA speed and ImageNet
accuracy still require target-hardware runs. Tests cover aligned C4/D4 window
attention, numerical/gradient equivalence, checkpoint state and distributed
validation. The original full model has stride-phase and window-grid limits;
these execution changes preserve its function without claiming exact spatial
equivariance for every image size or nominal group.

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
