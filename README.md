# REViT — Roto-reflection Equivariant convolutional Vision Transformer
**PAPER:** [![arXiv](https://img.shields.io/badge/arXiv-2606.25318-b31b1b.svg)](https://arxiv.org/abs/2606.25318)

REViT is a research codebase for rotation/roto-reflection equivariant vision transformers. It contains:

- a symmetry-and-width supernet that trains one shared D4 model and exports standalone subgroup profiles,
- a budget-adaptive REViT with nested widths and one invariant decision per image,
- an equivariant transformer baseline (`Rot2DTransformer`) for small image datasets,
- a windowed hierarchical equivariant transformer (`Rot2DTransformerV2`) for ImageNet-scale training.

## Repository layout

### Core models

- `symmetry_supernet.py`: D4 coefficient sharing, subgroup crops, sampled profile training, and native checkpoint export.
- `budget_revit.py`: shared lifting stem, nested steerable convolutions, global and windowed profiles, invariant controller, and fixed-profile export.
- `budget_cli.py`: shared adaptive flags, config serialization, and checkpoint loading for both profile models.
- `revit_gcsa.py`: original/global-attention equivariant transformer (`Rot2DTransformer`).
- `revit_windowed_gcsa.py`: windowed hierarchical equivariant transformer (`Rot2DTransformerV2`).
- `group_space.py`: maps group strings (`C4`, `D4`, etc.) to ESCNN `gspaces`.

### Training scripts

- `train_revit.py`: global or adaptive training entrypoint for `rotmnist`, `cifar10`, and `pcam` datasets.
- `imagenet_train_revit.py`: ImageNet training for the symmetry supernet and the original and adaptive windowed models (DDP, AMP, Mixup/CutMix, checkpoint resume).
- `imagenet_train_vit.py`: ImageNet training for a vanilla `torchvision` ViT-Small baseline.

### Dataset utilities

- `datasets/rot_mnist.py`: rotated MNIST dataset class and dataloaders.
- `datasets/cifar10.py`: CIFAR-10 dataloaders and augmentation.
- `datasets/pcam.py`: PCam dataloaders.

### Experiments and utilities

- `scripts/run_supernet.py`: list trained profiles, export standalone checkpoints, and build cold-cropped controls.
- `scripts/run_budget.py`: evaluate budgets and controls, benchmark end-to-end forwards, and export fixed profiles.
- `tests/test_budget_revit.py`, `tests/test_symmetry_supernet.py`: numerical symmetry, operator shapes, training, and checkpoint tests.
- `utils.py`: standalone quarter-turn augmentation and early-stopping utilities.
- `scripts/cleanup_checkpoints.py`: keep only the newest N checkpoint files in a directory.
- `scripts/benchmark_sdpa.py`: compare bmm vs SDPA attention paths for latency/memory.
- `requirements.txt` / `requirements-lock.txt`: install deps and optional pinned lockfile.

## Supported symmetry groups

You can pass `--group-str` as one of:

The profile models support `D4` (8 actions), `C4` (4 rotations), `D2_4` (half-turn and two mirrors), `C2` (half-turn), `Cs` (one mirror), and `C1` (identity). Legacy `D2` is an alias for `Cs`, and legacy `Z2` is an alias for `C1`. The original REViT backbones also accept `C8`, `C12`, `C16`, `D8`, `D12`, and `D16`.

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

## Train once and export symmetry and width profiles

The symmetry supernet shares one bank of $D_4$ steerable coefficients. Each training step selects a `(group, stage widths)` profile. A smaller profile slices whole fields and retains one complete orbit of its subgroup inside each field, starting at the lift. Each profile keeps separate normalization state. At export, the tool converts the selected weights into the native steerable basis of its group. The resulting checkpoint runs as a standalone REViT and can be fine-tuned within its exported group.

### Tiny proof of concept on ImageNet-1K

`--model-size tiny` selects the REViT Tiny preset with widths `12,24,48,96`, depths `1,2,3,1`, and heads `1,2,4,8`. Training uses standard 224-pixel ImageNet crops and 1,000 classes by default. Pass the ImageNet-1K root containing `train/<class>/` and `val/<class>/`.

```bash
python imagenet_train_revit.py --symmetry-supernet --model-size tiny \
  --data-root /path/to/imagenet --output-dir runs_supernet/tiny \
  --epochs 1 --warmup-epochs 0 --batch-size 8
```

The defaults train `D4` and `C4` at three head-aligned widths each. The full-width $D_4$ profile runs every other optimizer step, and the remaining steps cycle through the other profiles. Each validation sweep reports every profile. The mean profile top-1 selects the checkpoint. Our local PoC ran this preset at 224 pixels with a synthetic ImageFolder fixture, exercising training and export at the ImageNet input and output shapes.

For custom profiles, use `--symmetry-profiles 'D4:8,12,24,48,96;C4:2,4,8,16,32'`. With five numbers, the first is the stem width in regular fields, followed by the four stage widths. A four-number profile uses `--budget-stem-width`, or the smallest first-stage width across profiles when the flag is omitted. Each stem width must fit its own first stage. Stage widths must fit within the parent and divide evenly across attention heads. Include a `D4` profile with the full stage widths from the preset or `--dims`. If you specify several full-stage `D4` profiles, the one with the largest stem is the training reference.

### Export, evaluate, and fine-tune

```bash
CKPT=runs_supernet/tiny/checkpoints/imagenet_symmetry_supernet_best_epoch_0.pt
python -m scripts.run_supernet list "$CKPT"
python -m scripts.run_supernet export "$CKPT" 3 runs_supernet/c4_small.pt --verify
python -m scripts.run_budget evaluate runs_supernet/c4_small.pt \
  --dataset imagenet --data-root /path/to/imagenet --check-group
```

For the default profile ordering, index `3` is `C4:4,8,16,32`. Benchmark the export with `scripts.run_budget benchmark --checkpoint ...`. To fine-tune, match the export's group, head counts, depths, stem width and resolution. The default Tiny example uses:

```bash
python imagenet_train_revit.py --model-size tiny --group-str C4 \
  --budget-profiles '4,8,16,32' --budget-stem-width 4 \
  --data-root /path/to/imagenet --output-dir runs_supernet/c4_finetune \
  --pretrained --pretrained-weights-only --pretrained-path runs_supernet/c4_small.pt
```

For the cold-crop control, train a width-only `D4` singleton with the same backbone and stem width. Crop that checkpoint to the target group and width before evaluation and optional fine-tuning.

```bash
python -m scripts.run_supernet crop-d4 d4_only.pt C4 c4_cold.pt --widths 4 8 16 32
```

Prepend a stem width to `--widths` to crop the lift as well. The export metadata and `scripts.run_supernet list` report stem width separately from the stage widths.

### Distributed supernet training on ImageNet-1K

Run from the repository root with CUDA-enabled PyTorch. Launch one process per GPU with `torchrun` and pass `--ddp` to the trainer. Each process loads `--batch-size` images per step, so the global batch size is the per-GPU batch size multiplied by the total number of GPUs. `--num-workers` is also per process. Checkpoints and TensorBoard logs are written by rank zero.

**Single node, two GPUs.** Use the actual Tiny preset for a one-epoch DDP smoke test with custom stem widths. This example uses a global batch of 16.

```bash
torchrun --standalone --nproc-per-node=2 imagenet_train_revit.py --ddp \
  --model-size tiny \
  --symmetry-profiles 'D4:8,12,24,48,96;C4:2,4,8,16,32' \
  --data-root /path/to/imagenet --output-dir runs_supernet/tiny_ddp \
  --epochs 1 --warmup-epochs 0 --batch-size 8 --num-workers 4
```

**Single node, four GPUs.** Use Large with the default six symmetry and width profiles. The global batch size here is 32. `--supernet-val-every 5` evaluates all profiles every fifth epoch and again at the end of training.

```bash
torchrun --standalone --nproc-per-node=4 imagenet_train_revit.py --ddp \
  --symmetry-supernet --model-size large --data-root /path/to/imagenet \
  --output-dir runs_supernet/large --batch-size 8 --num-workers 4 \
  --epochs 300 --warmup-epochs 20 --supernet-val-every 5
```

Choose the per-GPU batch size based on the full-width $D_4$ training step, which expands the largest dense kernels.

**Two nodes, four GPUs per node.** Run the command on both nodes. Set `NODE_RANK` to `0` on the first node and `1` on the second. Set `MASTER_ADDR` on both nodes to the hostname or IP address of node 0. Both nodes need access to the same ImageNet train and validation splits. The global batch size in this example is 64.

```bash
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  imagenet_train_revit.py --ddp --symmetry-supernet --model-size large \
  --data-root /path/to/imagenet --output-dir /shared/runs_supernet/large \
  --batch-size 8 --num-workers 4 --epochs 300 --warmup-epochs 20 \
  --supernet-val-every 5
```

To resume, use the same backbone and ordered profile list, including stem widths. Add `--pretrained --pretrained-path /path/to/checkpoints/imagenet_symmetry_supernet_best_epoch_4.pt` to the launch command and set `--epochs` to the number of additional epochs. Use a training `.pt` checkpoint accessible to every rank so the trainer can restore the optimizer and learning-rate scheduler.

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
python -m pytest -q tests/test_budget_revit.py tests/test_symmetry_supernet.py
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
