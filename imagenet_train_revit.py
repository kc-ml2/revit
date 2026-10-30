import argparse
import os
import random
import warnings
from datetime import datetime
from imagenet_utils import (
    DistributedEvalSampler, build_scheduler, evaluate, load_model_state,
    restore_training_state, save_training_checkpoint, setup_distributed,
    train_one_epoch,
)
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T_v2
from group_space import get_gspace
from model_config import MODEL_PRESETS

from revit_windowed_gcsa import Rot2DTransformerV2, count_parameters
from scripts.profile_scaling import estimate_revit


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy for soft labels from Mixup/CutMix."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1).mean()


def build_mixup_cutmix(num_classes: int, mixup_alpha: float, cutmix_alpha: float, switch_prob: float):
    """Return a batch transform that randomly applies Mixup or CutMix, or None if both disabled."""
    ops = []
    probs = []
    if mixup_alpha > 0.0:
        ops.append(T_v2.MixUp(num_classes=num_classes, alpha=mixup_alpha))
        probs.append(switch_prob if cutmix_alpha > 0.0 else 1.0)
    if cutmix_alpha > 0.0:
        ops.append(T_v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha))
        probs.append((1.0 - switch_prob) if mixup_alpha > 0.0 else 1.0)
    if not ops:
        return None
    if len(ops) == 1:
        return ops[0]
    total = sum(probs)
    probs = [p / total for p in probs]
    return T_v2.RandomChoice(ops, p=probs)


def apply_mixup_cutmix(mixup_fn, x: torch.Tensor, y: torch.Tensor, mixup_prob: float):
    """Maybe apply Mixup/CutMix. Returns (x, y, used_soft_targets)."""
    if mixup_fn is None or mixup_prob <= 0.0 or random.random() >= mixup_prob:
        return x, y, False
    x, y = mixup_fn(x, y)
    return x, y, True


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank is None or rank == 0


def load_pretrained_weights(model: nn.Module, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at {path} has no 'model_state_dict' key")
    load_model_state(model, ckpt["model_state_dict"])
    return ckpt


def get_imagenet_loaders(
    data_root,
    batch_size=256,
    num_workers=8,
    use_ddp=False,
    color_jitter=True,
    randaugment=True,
    randaugment_num_ops=2,
    randaugment_magnitude=9,
    random_erase=0.25,
    image_size=224,
):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        # RandomHorizontalFlip omitted: redundant under D4 / reflection groups.
    ]
    if randaugment:
        train_list.append(
            transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude)
        )
    if color_jitter:
        train_list.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
    train_list.extend([transforms.ToTensor(), normalize])
    if random_erase > 0.0:
        train_list.append(transforms.RandomErasing(p=random_erase, value="random"))
    train_tfms = transforms.Compose(train_list)

    val_tfms = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedEvalSampler(val_ds) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_sampler


def main():
    parser = argparse.ArgumentParser("ImageNet training for Rot2DTransformerV2")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="imagenet_es_v2_outputs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size in DDP mode")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=50, help="Batch interval for device-to-host loss logging; 0 disables")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save last.pt every N epochs (0: best only), plus final epoch when N > 0")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # DeiT-style augmentations (defaults on; set magnitudes/probs to 0 or use --no-* to disable)
    parser.add_argument("--color-jitter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randaugment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randaugment-num-ops", type=int, default=2)
    parser.add_argument("--randaugment-magnitude", type=int, default=9)
    parser.add_argument("--random-erase", type=float, default=0.25, help="RandomErasing probability (0 disables)")
    parser.add_argument("--mixup", type=float, default=0.8, help="Mixup alpha (0 disables)")
    parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (0 disables)")
    parser.add_argument("--mixup-prob", type=float, default=1.0, help="Probability of applying Mixup/CutMix per batch")
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of choosing Mixup vs CutMix when both are enabled",
    )
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--group-str", type=str, default='C4')
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "large"],
    )
    parser.add_argument("--dims", type=int, nargs=4, default=None, help="Override stage regular-field counts (actual channels = fields * group order)")
    parser.add_argument("--depths", type=int, nargs=4, default=None, help="Override stage depths")
    parser.add_argument("--heads", type=int, nargs=4, default=None, help="Override stage heads")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stem-downsample", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--attention-backend", choices=["bmm", "sdpa"], default="bmm")
    parser.add_argument("--fused-qkv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--qkv-kernel-size", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--fast-init", action="store_true", help="Use faster non-basis-aware init for R2Conv")

    parser.add_argument("--ddp", action="store_true", help="Use DDP (launch via torchrun)")
    parser.add_argument("--data-parallel", action="store_true", help="Use DataParallel when not using DDP")

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load weights from --pretrained-path before training (and resume optimizer/scheduler unless --pretrained-weights-only)",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to .pt checkpoint (same format as training saves: model_state_dict, optional optimizer/scheduler)",
    )
    parser.add_argument(
        "--pretrained-weights-only",
        action="store_true",
        help="Only load model weights; new optimizer/scheduler, start at epoch 0",
    )

    args = parser.parse_args()
    if args.pretrained and not args.pretrained_path:
        parser.error("--pretrained-path is required when --pretrained is set")
    if args.image_size <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("image size/batch size must be positive and num-workers nonnegative")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("Require 0 <= --warmup-epochs < --epochs")

    preset = MODEL_PRESETS[args.model_size]
    if args.dims is None:
        args.dims = preset["dims"]
    if args.depths is None:
        args.depths = preset["depths"]
    if args.heads is None:
        args.heads = preset["heads"]

    stage_sides = []
    side = (args.image_size + args.stem_downsample - 1) // args.stem_downsample
    for _ in args.dims:
        stage_sides.append(side)
        side = (side + 1) // 2
    if args.window_size > 0 and any(side % args.window_size for side in stage_sides):
        warnings.warn(
            f"Stage sizes {stage_sides} are not all divisible by window size {args.window_size}. "
            "Legacy bottom/right padding changes attention near edges and breaks window-grid "
            "rotation/reflection symmetry. Choose aligned sizes for equivariance checks "
            "(e.g. image=224, window=7, stem=4)."
        )

    rank, world_size, local_rank = setup_distributed(args.ddp)
    seed_everything(args.seed + (rank or 0))
    use_ddp = args.ddp and rank is not None

    if use_ddp:
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader, val_loader, train_sampler = get_imagenet_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ddp=use_ddp,
        image_size=args.image_size,
        color_jitter=args.color_jitter,
        randaugment=args.randaugment,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
        random_erase=args.random_erase,
    )

    mixup_fn = build_mixup_cutmix(
        num_classes=args.num_classes,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        switch_prob=args.mixup_switch_prob,
    )
    if is_main_process(rank):
        print(
            "Augmentations: "
            f"color_jitter={args.color_jitter}, "
            f"randaugment={args.randaugment}(ops={args.randaugment_num_ops},m={args.randaugment_magnitude}), "
            f"random_erase={args.random_erase}, "
            f"mixup={args.mixup}, cutmix={args.cutmix}, "
            f"mixup_prob={args.mixup_prob}, switch_prob={args.mixup_switch_prob}"
        )

    gspace = get_gspace(args.group_str)
    if is_main_process(rank):
        order = gspace.fibergroup.order()
        print(f"Geometry: image={args.image_size}, stage_sides={stage_sides}, "
              f"fields={tuple(args.dims)}, channels={tuple(d * order for d in args.dims)}, "
              f"group_order={order}, window={args.window_size}, "
              f"attention={args.attention_backend}, fused_qkv={args.fused_qkv}, "
              f"amp_dtype={args.amp_dtype}, global_batch={args.batch_size * (world_size or 1)}")
    model = Rot2DTransformerV2(
        gspace=gspace,
        in_channels=3,
        num_classes=args.num_classes,
        dims=tuple(args.dims),
        depths=tuple(args.depths),
        heads=tuple(args.heads),
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        qkv_kernel_size=args.qkv_kernel_size,
        use_checkpoint=args.use_checkpoint,
        fast_init=args.fast_init,
        stem_downsample=args.stem_downsample,
    ).to(device)

    for module in model.modules():
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = args.attention_backend == "sdpa"
            module.fused_qkv = args.fused_qkv

    resume_ckpt = None
    if args.pretrained:
        if not os.path.isfile(args.pretrained_path):
            raise FileNotFoundError(f"--pretrained-path not found: {args.pretrained_path}")
        resume_ckpt = load_pretrained_weights(model, args.pretrained_path, device)
        if is_main_process(rank):
            print(f"Loaded pretrained weights from {args.pretrained_path}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    elif args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if is_main_process(rank):
        count_parameters(model)
        flop_stats = estimate_revit(
            {"dims": tuple(args.dims), "depths": tuple(args.depths), "heads": tuple(args.heads)},
            group_order=gspace.fibergroup.order(),
            image_size=args.image_size,
            window_size=args.window_size,
            kernel_size=args.qkv_kernel_size,
            mlp_ratio=args.mlp_ratio,
            stem_downsample=args.stem_downsample,
            num_classes=args.num_classes,
        )
        print(
            f"Estimated FLOPs (per-image forward, 2 FLOPs/MAC): "
            f"{flop_stats['flops']:,} ({flop_stats['flops'] / 1e9:.3f} GFLOPs)"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs", f"imagenet_revit_{args.group_str}_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    else:
        writer = None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        fused=args.fused_adamw and device.type == "cuda",
    )
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.amp_dtype]
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        parser.error("This CUDA device does not support bfloat16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    start_epoch = 0
    best_top1 = float("-inf")
    best_epoch = -1
    if (
        args.pretrained
        and resume_ckpt is not None
        and not args.pretrained_weights_only
        and "optimizer_state_dict" in resume_ckpt
        and "scheduler_state_dict" in resume_ckpt
    ):
        saved_epochs = resume_ckpt.get("args", {}).get("epochs")
        if saved_epochs is not None and saved_epochs != args.epochs:
            raise ValueError("Resume with the original total --epochs to retain the LR schedule; "
                             "use --pretrained-weights-only to start a new schedule")
        restore_training_state(resume_ckpt, optimizer, scheduler, scaler, rank)
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_top1 = float(resume_ckpt.get("best_top1", resume_ckpt.get("top1", 0.0)))
        best_epoch = int(resume_ckpt.get("best_epoch", resume_ckpt.get("epoch", -1)))
        if is_main_process(rank):
            print(f"Resuming training from epoch {start_epoch} (loaded optimizer and scheduler)")
    elif args.pretrained and resume_ckpt is not None and not args.pretrained_weights_only and is_main_process(rank):
        if "optimizer_state_dict" not in resume_ckpt or "scheduler_state_dict" not in resume_ckpt:
            print(
                "Checkpoint has no optimizer/scheduler state; training from epoch 0 with new optimizer/scheduler"
            )

    del resume_ckpt
    best_ckpt_path = None

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if writer is not None:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            rank if use_ddp else None,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            amp_dtype=amp_dtype,
            log_interval=args.log_interval,
            mixup_fn=mixup_fn,
            mixup_prob=args.mixup_prob,
        )
        scheduler.step()

        val_loss, top1, top5 = evaluate(model, val_loader, device, rank if use_ddp else None, amp_dtype=amp_dtype)

        if is_main_process(rank):
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/top1", top1, epoch)
            writer.add_scalar("val/top5", top5, epoch)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | top1={top1:.4f} | top5={top5:.4f}"
            )

        if top1 > best_top1:
            best_top1, best_epoch = top1, epoch
            best_ckpt_path = os.path.join(args.output_dir, "checkpoints", "best.pt")
            save_training_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, epoch, top1, device,
                scaler=scaler, args=args, best_top1=best_top1, best_epoch=best_epoch,
            )
        if args.checkpoint_interval > 0 and ((epoch + 1) % args.checkpoint_interval == 0 or epoch + 1 == args.epochs):
            save_training_checkpoint(
                os.path.join(args.output_dir, "checkpoints", "last.pt"),
                model, optimizer, scheduler, epoch, top1, device,
                scaler=scaler, args=args, best_top1=best_top1, best_epoch=best_epoch,
            )

    if is_main_process(rank):
        writer.close()
        if best_ckpt_path is not None:
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            torch.save(
                ckpt["model_state_dict"],
                os.path.join(
                    args.output_dir,
                    f"imagenet_revit_{args.group_str}_{args.model_size}_best_top1_{best_top1:.4f}_epoch_{best_epoch}.pth",
                ),
            )
            print(f"Best top1={best_top1:.4f} at epoch={best_epoch}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
