"""Train REViT OBB detector on HRSC2016."""

from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.hrsc2016 import get_hrsc2016_loaders
from revit_obb_detector import DEFAULT_D4_BASE_CKPT, build_revit_obb_d4_base, count_parameters


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def assign_targets_single_image(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    feat_shapes: Sequence[Tuple[int, int]],
    strides: Sequence[int],
    num_classes: int,
    size_ranges: Sequence[Tuple[float, float]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    FCOS-style center assignment for one image.

    Returns per-level:
      cls_target  (C, H, W) float {0,1}
      reg_target  (6, H, W)  log_w, log_h, dx, dy, cos2θ, sin2θ
      pos_mask    (H, W) bool
    """
    device = boxes.device
    dtype = boxes.dtype if boxes.numel() else torch.float32
    cls_tgts, reg_tgts, pos_masks = [], [], []

    for (fh, fw), stride, (amin, amax) in zip(feat_shapes, strides, size_ranges):
        cls_t = torch.zeros(num_classes, fh, fw, device=device, dtype=dtype)
        reg_t = torch.zeros(6, fh, fw, device=device, dtype=dtype)
        pos = torch.zeros(fh, fw, device=device, dtype=torch.bool)

        if boxes.numel() == 0:
            cls_tgts.append(cls_t)
            reg_tgts.append(reg_t)
            pos_masks.append(pos)
            continue

        # object scale for level selection
        scales = torch.sqrt(boxes[:, 2] * boxes[:, 3])
        level_keep = (scales >= amin) & (scales < amax)
        if not level_keep.any():
            # fall back: assign oversized/undersized boxes to extreme levels
            if amax == float("inf"):
                level_keep = scales >= amin
            elif amin <= 0:
                level_keep = scales < amax

        for b_i in torch.where(level_keep)[0].tolist():
            cx, cy, bw, bh, ang = boxes[b_i].tolist()
            label = int(labels[b_i].item())
            # nearest feature location to box center
            fx = cx / stride - 0.5
            fy = cy / stride - 0.5
            ix = int(round(fx))
            iy = int(round(fy))
            if not (0 <= ix < fw and 0 <= iy < fh):
                continue
            # 3x3 center sampling
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    jx, jy = ix + dx, iy + dy
                    if not (0 <= jx < fw and 0 <= jy < fh):
                        continue
                    loc_x = (jx + 0.5) * stride
                    loc_y = (jy + 0.5) * stride
                    # inside rotated box (axis-aligned approx in object frame)
                    # transform point into box frame
                    c, s = math.cos(ang), math.sin(ang)
                    rx = (loc_x - cx) * c + (loc_y - cy) * s
                    ry = -(loc_x - cx) * s + (loc_y - cy) * c
                    if abs(rx) > bw * 0.5 or abs(ry) > bh * 0.5:
                        if dx != 0 or dy != 0:
                            continue
                    pos[jy, jx] = True
                    cls_t[label, jy, jx] = 1.0
                    reg_t[0, jy, jx] = math.log(max(bw, 1.0))
                    reg_t[1, jy, jx] = math.log(max(bh, 1.0))
                    reg_t[2, jy, jx] = (cx - loc_x) / stride
                    reg_t[3, jy, jx] = (cy - loc_y) / stride
                    reg_t[4, jy, jx] = math.cos(2.0 * ang)
                    reg_t[5, jy, jx] = math.sin(2.0 * ang)

        cls_tgts.append(cls_t)
        reg_tgts.append(reg_t)
        pos_masks.append(pos)

    return cls_tgts, reg_tgts, pos_masks


def build_size_ranges(strides: Sequence[int]) -> List[Tuple[float, float]]:
    """Default FCOS regress ranges scaled to absolute object size in pixels."""
    # For strides (8,16,32): [0,64), [64,128), [128,inf)
    ranges = []
    prev = 0.0
    for i, s in enumerate(strides):
        if i == len(strides) - 1:
            ranges.append((prev, float("inf")))
        else:
            nxt = float(s * 8)
            ranges.append((prev, nxt))
            prev = nxt
    return ranges


class OBBDetectionLoss(nn.Module):
    def __init__(self, num_classes: int, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: Dict,
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        cls_logits: List[torch.Tensor] = outputs["cls_logits"]
        obb_preds: List[torch.Tensor] = outputs["obb_preds"]
        strides: List[int] = outputs["strides"]
        feat_shapes = [(c.shape[-2], c.shape[-1]) for c in cls_logits]
        size_ranges = build_size_ranges(strides)

        device = cls_logits[0].device
        total_cls = cls_logits[0].new_tensor(0.0)
        total_reg = cls_logits[0].new_tensor(0.0)
        num_pos = 0

        B = cls_logits[0].shape[0]
        for b in range(B):
            boxes = targets[b]["boxes"].to(device)
            labels = targets[b]["labels"].to(device)
            cls_t, reg_t, pos_m = assign_targets_single_image(
                boxes,
                labels,
                feat_shapes,
                strides,
                self.num_classes,
                size_ranges,
            )
            for lvl in range(len(strides)):
                logit = cls_logits[lvl][b]  # (C,H,W)
                pred = obb_preds[lvl][b]  # (6,H,W)
                tgt_c = cls_t[lvl]
                tgt_r = reg_t[lvl]
                pos = pos_m[lvl]
                total_cls = total_cls + sigmoid_focal_loss(
                    logit.reshape(-1),
                    tgt_c.reshape(-1),
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction="sum",
                )
                n = int(pos.sum().item())
                if n > 0:
                    # normalize angle vector predictions gently
                    pred_pos = pred[:, pos]
                    tgt_pos = tgt_r[:, pos]
                    # size + offset L1
                    total_reg = total_reg + F.l1_loss(pred_pos[:4], tgt_pos[:4], reduction="sum")
                    # orientation: L1 on (cos2θ, sin2θ)
                    total_reg = total_reg + F.l1_loss(pred_pos[4:], tgt_pos[4:], reduction="sum")
                    num_pos += n

        num_pos = max(num_pos, 1)
        loss_cls = total_cls / num_pos
        loss_reg = total_reg / num_pos
        return {
            "loss": loss_cls + loss_reg,
            "loss_cls": loss_cls.detach(),
            "loss_reg": loss_reg.detach(),
            "num_pos": cls_logits[0].new_tensor(float(num_pos)),
        }


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, grad_clip: float):
    model.train()
    total = 0.0
    total_cls = 0.0
    total_reg = 0.0
    steps = 0
    pbar = tqdm(loader, unit="batch", desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"],
            }
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        total_cls += float(losses["loss_cls"])
        total_reg += float(losses["loss_reg"])
        steps += 1
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            cls=f"{float(losses['loss_cls']):.3f}",
            reg=f"{float(losses['loss_reg']):.3f}",
            pos=int(losses["num_pos"].item()),
        )
    n = max(steps, 1)
    return total / n, total_cls / n, total_reg / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    total_cls = 0.0
    total_reg = 0.0
    steps = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"],
            }
            for t in targets
        ]
        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(images)
            losses = criterion(outputs, targets)
        total += float(losses["loss"])
        total_cls += float(losses["loss_cls"])
        total_reg += float(losses["loss_reg"])
        steps += 1
    n = max(steps, 1)
    return total / n, total_cls / n, total_reg / n


def main():
    parser = argparse.ArgumentParser("Train REViT OBB detector on HRSC2016")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/HRSC2016",
        help="Path to HRSC2016 root (contains AllImages/ or nested HRSC2016/)",
    )
    parser.add_argument("--output-dir", type=str, default="hrsc_obb_outputs")
    parser.add_argument("--pretrained-path", type=str, default=DEFAULT_D4_BASE_CKPT)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fpn-dim", type=int, default=32)
    parser.add_argument("--head-convs", type=int, default=4)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--fast-init", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--fine-grained", action="store_true", help="Use fine-grained ship classes")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = get_hrsc2016_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        single_class=not args.fine_grained,
    )
    print(f"HRSC2016: num_classes={num_classes}, train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    pretrained = None if args.no_pretrained else args.pretrained_path
    model = build_revit_obb_d4_base(
        num_classes=num_classes,
        pretrained_path=pretrained,
        fpn_dim=args.fpn_dim,
        head_convs=args.head_convs,
        freeze_backbone=args.freeze_backbone,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
        fast_init=args.fast_init,
    ).to(device)
    count_parameters(model)

    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=max(args.warmup_epochs, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda")
    criterion = OBBDetectionLoss(num_classes=num_classes)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(
        log_dir=os.path.join(
            args.output_dir,
            "runs",
            f"hrsc_obb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    )

    best_val = float("inf")
    best_path = None
    for epoch in range(args.epochs):
        train_loss, train_cls, train_reg = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch, args.grad_clip
        )
        scheduler.step()
        val_loss, val_cls, val_reg = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/loss_cls", train_cls, epoch)
        writer.add_scalar("train/loss_reg", train_reg, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/loss_cls", val_cls, epoch)
        writer.add_scalar("val/loss_reg", val_reg, epoch)
        writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)
        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} (cls={train_cls:.4f}, reg={train_reg:.4f}) | "
            f"val={val_loss:.4f} (cls={val_cls:.4f}, reg={val_reg:.4f})"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        last_path = os.path.join(args.output_dir, "checkpoints", "last.pt")
        torch.save(ckpt, last_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "checkpoints", f"best_val_{best_val:.4f}_epoch_{epoch}.pt")
            torch.save(ckpt, best_path)
            print(f"  saved best checkpoint -> {best_path}")

    writer.close()
    print(f"Done. best_val={best_val:.4f} path={best_path}")


if __name__ == "__main__":
    main()
