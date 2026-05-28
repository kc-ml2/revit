import argparse
import copy
import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from group_space import get_gspace

from revit_windowed_gcsa import Rot2DTransformerV2, count_parameters


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None, None, None
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank is None or rank == 0


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    k0 = next(iter(state_dict.keys()))
    if k0.startswith("module."):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_pretrained_weights(model: nn.Module, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at {path} has no 'model_state_dict' key")
    model.load_state_dict(_strip_module_prefix(ckpt["model_state_dict"]))
    return ckpt


def get_imagenet_loaders(data_root, batch_size=256, num_workers=8, use_ddp=False):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            # transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, rank, grad_clip=1.0, label_smoothing=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    steps = 0

    pbar = tqdm(loader, unit="batch", disable=not is_main_process(rank))
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
        pbar.set_description(f"Epoch {epoch}")
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(steps, 1)
    if rank is not None and dist.is_initialized():
        t = torch.tensor([avg_loss], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        avg_loss = (t / dist.get_world_size()).item()
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, rank):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    top1 = 0
    top5 = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

        _, pred = out.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.eq(y.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct.sum().item()

    if rank is not None and dist.is_initialized():
        t = torch.tensor([total_loss, float(total), float(top1), float(top5)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total, top1, top5 = t.tolist()

    return total_loss / total, top1 / total, top5 / total


def main():
    parser = argparse.ArgumentParser("ImageNet training for Rot2DTransformerV2")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="imagenet_es_v2_outputs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size in DDP mode")
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--group-str", type=str, default='C4')
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small-D4", "small", "base"],
    )
    parser.add_argument("--dims", type=int, nargs=4, default=None, help="Override stage dims")
    parser.add_argument("--depths", type=int, nargs=4, default=None, help="Override stage depths")
    parser.add_argument("--heads", type=int, nargs=4, default=None, help="Override stage heads")
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
    seed_everything(args.seed)

    # ViT-tiny-like default capacity (unless explicitly overridden)
    presets = {
        "tiny": {"dims": [12, 24, 48, 96], "depths": [1, 2, 3, 1], "heads": [1, 2, 4, 8]},
        "small": {"dims": [24, 48, 96, 192], "depths": [1, 2, 4, 1], "heads": [1, 2, 4, 8]},
        "medium": {"dims": [32, 64, 128, 256], "depths": [1, 2, 4, 1], "heads": [1, 2, 4, 8]},
        "base": {"dims": [64, 128, 256, 512], "depths": [2, 2, 6, 2], "heads": [2, 4, 8, 16]},
    }
    preset = presets[args.model_size]
    if args.dims is None:
        args.dims = preset["dims"]
    if args.depths is None:
        args.depths = preset["depths"]
    if args.heads is None:
        args.heads = preset["heads"]

    rank, _, local_rank = setup_distributed()
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
    )

    # gspace = gspaces.rot2dOnR2(N=args.group_n)
    gspace = get_gspace(args.group_str)
    model = Rot2DTransformerV2(
        gspace=gspace,
        in_channels=3,
        num_classes=1000,
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
    ).to(device)

    resume_ckpt = None
    if args.pretrained:
        if not os.path.isfile(args.pretrained_path):
            raise FileNotFoundError(f"--pretrained-path not found: {args.pretrained_path}")
        resume_ckpt = load_pretrained_weights(model, args.pretrained_path, device)
        if is_main_process(rank):
            print(f"Loaded pretrained weights from {args.pretrained_path}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if is_main_process(rank):
        count_parameters(model)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs", f"imagenet_es_v2_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    else:
        writer = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.03, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda")

    start_epoch = 0
    best_top1 = 0.0
    best_epoch = -1
    if (
        args.pretrained
        and resume_ckpt is not None
        and not args.pretrained_weights_only
        and "optimizer_state_dict" in resume_ckpt
        and "scheduler_state_dict" in resume_ckpt
    ):
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_top1 = float(resume_ckpt.get("top1", 0.0))
        best_epoch = int(resume_ckpt.get("epoch", -1))
        if is_main_process(rank):
            print(f"Resuming training from epoch {start_epoch} (loaded optimizer and scheduler)")
    elif args.pretrained and resume_ckpt is not None and not args.pretrained_weights_only and is_main_process(rank):
        if "optimizer_state_dict" not in resume_ckpt or "scheduler_state_dict" not in resume_ckpt:
            print(
                "Checkpoint has no optimizer/scheduler state; training from epoch 0 with new optimizer/scheduler"
            )

    best_state = None

    for epoch in range(start_epoch, start_epoch + args.epochs):
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
        )
        scheduler.step()

        val_loss, top1, top5 = evaluate(model, val_loader, device, rank if use_ddp else None)

        if is_main_process(rank):
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/top1", top1, epoch)
            writer.add_scalar("val/top5", top5, epoch)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | top1={top1:.4f} | top5={top5:.4f}"
            )

            if top1 > best_top1:
                best_top1 = top1
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "top1": best_top1,
                    },
                    os.path.join(args.output_dir, "checkpoints", f"imagenet_es_v2_best_epoch_{best_epoch}.pt"),
                )

    if is_main_process(rank):
        writer.close()
        if best_state is not None:
            torch.save(
                best_state,
                os.path.join(args.output_dir, f"imagenet_es_v2_best_top1_{best_top1:.4f}_epoch_{best_epoch}.pth"),
            )
            print(f"Best top1={best_top1:.4f} at epoch={best_epoch}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
