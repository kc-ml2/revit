# REViT — Roto-reflection Equivariant convolutional Vision Transformer

REViT is a research codebase for rotation/roto-reflection equivariant vision transformers. It contains:

- an equivariant transformer baseline (`Rot2DTransformer`) for small image datasets,
- a windowed hierarchical equivariant transformer (`Rot2DTransformerV2`) for ImageNet-scale training,

## Repository layout

### Core models

- `revit_gcsa.py`: original/global-attention equivariant transformer (`Rot2DTransformer`).
- `revit_windowed_gcsa.py`: windowed hierarchical equivariant transformer (`Rot2DTransformerV2`).
- `group_space.py`: maps group strings (`C4`, `D4`, etc.) to ESCNN `gspaces`.

### Training scripts

- `train_revit.py`: training entrypoint for `rotmnist`, `cifar10`, and `pcam` datasets.
- `imagenet_train_revit.py`: ImageNet training for `Rot2DTransformerV2` (DDP, AMP, checkpoint resume).
- `imagenet_train_vit.py`: ImageNet training for a vanilla `torchvision` ViT-Small baseline.

### Dataset utilities

- `datasets/rot_mnist.py`: rotated MNIST dataset class and dataloaders.
- `datasets/cifar10.py`: CIFAR-10 dataloaders and augmentation.
- `datasets/pcam.py`: PCam dataloaders.

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

- `tiny`: dims `24,48,96,192`; depths `1,1,3,1`; heads `1,2,4,8`
- `small`: dims `24,48,96,192`; depths `1,2,4,1`; heads `1,2,4,8`
- `medium`: dims `32,64,128,256`; depths `1,2,4,1`; heads `1,2,4,8`
- `base`: dims `64,128,256,512`; depths `2,2,6,2`; heads `2,4,8,16`

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

## Citation

```tex
@inproceedings{zaheer2026revit,
title={{REV}iT: Roto-reflection Equivariant Convolutional Vision Transformer},
author={Zaheer, Sheir A. and Holston, Alexander C. and Park, Chan Y.},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=n2RIMdIbv6}
}
```

## License

See `LICENSE`.