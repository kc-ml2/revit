# ReViT — Rotation-Equivariant Vision Transformer

ReViT is a Vision Transformer whose self-attention and feed-forward layers are built entirely from **equivariant convolutions** via [ESCNN](https://github.com/QUVA-Lab/escnn). Every feature map is a `GeometricTensor` that transforms consistently under the chosen discrete rotation (or roto-reflection) group, so the model's predictions are invariant to those symmetries by construction.

## How it works

Standard ViTs lift an image into a flat sequence of patch embeddings and apply standard self-attention. ReViT instead:

1. **Lifts** the input image into a `GeometricTensor` whose channels carry regular representations of the chosen group (e.g. C4, D4).
2. Applies a hierarchy of **Rot2DBlockV2** transformer blocks, each containing:
   - *Windowed group-equivariant self-attention* (`Rot2DWindowAttention`) — Q/K/V projections are `R2Conv` layers; attention is computed inside non-overlapping spatial windows (like Swin Transformer), then the spatial map is reconstructed.
   - *Equivariant MLP* (`Rot2DMLP`) — two 1×1 `R2Conv` layers with a hidden expansion.
   - Learnable layer-scale parameters (`alpha1`, `alpha2`) and stochastic depth (`DropPath`).
3. **Pools** over the group orbit with `GroupPooling` to collapse the symmetry axis, yielding a group-invariant feature vector fed to a standard linear classifier.

## Repository layout

| File | Description |
|---|---|
| `revit_windowed_gcsa.py` | Main model — `Rot2DTransformerV2` and all its building blocks (stem, windowed attention, MLP, classification head). **Start here.** |
| `revit_gcsa.py` | Earlier/alternative model — `Rot2DTransformer` with global (non-windowed) equivariant self-attention; useful as a reference. |
| `group_space.py` | Helper that maps a group string (`"C4"`, `"D4"`, …) to an ESCNN `GSpace`. |
| `imagenet_train_revit.py` | Full training script — DDP, mixed precision, cosine LR with linear warmup, TensorBoard logging, checkpoint save/resume. |
| `requirements.txt` | Python dependencies. |

## Supported symmetry groups

Pass `--group-str` with one of:

| String | Group | Symmetries |
|---|---|---|
| `Z2` | Trivial | None (standard convnet) |
| `D2` | Flip | Horizontal reflection only |
| `C4` | Cyclic-4 | 90° rotations |
| `C8` | Cyclic-8 | 45° rotations |
| `C12` | Cyclic-12 | 30° rotations |
| `C16` | Cyclic-16 | 22.5° rotations |
| `D4` | Dihedral-4 | 90° rotations + reflections |
| `D8` | Dihedral-8 | 45° rotations + reflections |
| `D12` | Dihedral-12 | 30° rotations + reflections |
| `D16` | Dihedral-16 | 22.5° rotations + reflections |


## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch — pick the wheel that matches your CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ESCNN stack (lie_learn requires Cython)
pip install Cython
pip install git+https://github.com/AMLab-Amsterdam/lie_learn.git
pip install -r requirements.txt
```

## Training on ImageNet

The training script expects an ImageNet directory with the standard `train/` and `val/` sub-folders (each containing one folder per class).

**Single GPU**

```bash
python imagenet_train_revit.py \
  --data-root /path/to/imagenet \
  --model-size small-D4 \
  --group-str D4 \
  --batch-size 256 \
  --epochs 300 \
  --warmup-epochs 20 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --drop-path-rate 0.2 \
  --qkv-kernel-size 3 \
  --window-size 7
```

**Multi-GPU (DDP via torchrun)**

```bash
torchrun --standalone --nproc_per_node=4 imagenet_train_revit.py \
  --ddp \
  --data-root /path/to/imagenet \
  --model-size small-D4 \
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

With `--ddp` and 4 GPUs the effective batch size is `4 × 128 = 512`. Scale `--lr` accordingly if you change the number of GPUs or per-GPU batch size.

### Resuming from a checkpoint

```bash
python imagenet_train_revit.py \
  --pretrained \
  --pretrained-path /path/to/checkpoint.pt \
  ... (other args)
```

Add `--pretrained-weights-only` to load only the model weights and restart the optimizer/scheduler from scratch.

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--data-root` | *(required)* | Path to ImageNet root |
| `--output-dir` | `imagenet_es_v2_outputs` | Where to save checkpoints and TensorBoard runs |
| `--model-size` | `tiny` | Architecture preset |
| `--group-str` | `C4` | Symmetry group |
| `--epochs` | 300 | Total training epochs |
| `--warmup-epochs` | 20 | Linear LR warmup length |
| `--batch-size` | 256 | Per-GPU batch size |
| `--lr` | 1e-3 | Peak learning rate (AdamW) |
| `--weight-decay` | 0.05 | AdamW weight decay |
| `--drop-path-rate` | 0.1 | Stochastic depth rate |
| `--window-size` | 7 | Attention window size |
| `--qkv-kernel-size` | 1 | Q/K/V convolution kernel size (must be odd) |
| `--use-checkpoint` | off | Gradient checkpointing (saves memory) |
| `--fast-init` | off | Skip expensive ESCNN basis-aware weight init (useful for quick smoke tests) |

## Monitoring

TensorBoard logs are written to `<output-dir>/runs/`:

```bash
tensorboard --logdir imagenet_es_v2_outputs/runs
```

Metrics logged: `lr`, `train/loss`, `val/loss`, `val/top1`, `val/top5`.

## Quick smoke test

```bash
python revit_windowed_gcsa.py
```

This runs a small forward pass (`2 × 3 × 224 × 224` input, C4 symmetry) and prints the parameter count and output shape.

## Dependencies

- [PyTorch](https://pytorch.org/) ≥ 2.6
- [torchvision](https://github.com/pytorch/vision) ≥ 0.21
- [ESCNN](https://github.com/QUVA-Lab/escnn) ≥ 1.0.11
- [lie_learn](https://github.com/AMLab-Amsterdam/lie_learn)
- NumPy, tqdm, TensorBoard, Pillow, matplotlib

## License

See [LICENSE](LICENSE).
