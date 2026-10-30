import argparse
import os
import random
from datetime import datetime
from imagenet_utils import (
    DistributedEvalSampler, build_scheduler, evaluate,
    save_training_checkpoint, setup_distributed,
    train_one_epoch,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    return total_params, trainable_params


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


def get_imagenet_loaders(data_root, batch_size=256, num_workers=8, use_ddp=False, image_size=224):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
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
        batch_size=batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_sampler


def build_vit_small(num_classes=1000, image_size=224, patch_size=16, dropout=0.0, attention_dropout=0.0):
    # Vanilla ViT-Small/16 style config
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        dropout=dropout,
        attention_dropout=attention_dropout,
        num_classes=num_classes,
    )


def main():
    parser = argparse.ArgumentParser("ImageNet training for vanilla ViT-Small")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="imagenet_vit_small_outputs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size in DDP mode")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=50, help="Batch interval for device-to-host loss logging; 0 disables")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save last.pt every N epochs (0: best only), plus final epoch when N > 0")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)

    parser.add_argument("--ddp", action="store_true", help="Use DDP (launch via torchrun)")
    parser.add_argument("--data-parallel", action="store_true", help="Use DataParallel when not using DDP")

    args = parser.parse_args()
    if args.image_size <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("image size/batch size must be positive and num-workers nonnegative")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("Require 0 <= --warmup-epochs < --epochs")

    rank, _, local_rank = setup_distributed(args.ddp)
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
    )

    model = build_vit_small(
        num_classes=1000,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dropout=args.dropout,
        attention_dropout=args.attn_dropout,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    elif args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if is_main_process(rank):
        count_parameters(model)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs", f"imagenet_vit_small_{run_id}"))
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

    best_top1 = float("-inf")
    best_epoch = -1
    best_ckpt_path = None

    for epoch in range(args.epochs):
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
                os.path.join(args.output_dir, f"imagenet_vit_small_best_top1_{best_top1:.4f}_epoch_{best_epoch}.pth"),
            )
            print(f"Best top1={best_top1:.4f} at epoch={best_epoch}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
