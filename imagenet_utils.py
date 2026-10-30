"""Shared ImageNet execution and checkpoint utilities.

Recipe choices (architecture, augmentations, optimizer hyperparameters) stay in
the entrypoints. Metrics accumulate on the device to avoid batch-wise syncs.
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Sampler
from tqdm import tqdm


class DistributedEvalSampler(Sampler):
    """Partition every example exactly once, including non-divisible datasets."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Invalid evaluation sampler rank/world size")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.num_replicas))


def setup_distributed(enabled):
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not enabled:
        if launched:
            raise ValueError("torchrun requires --ddp; otherwise ranks would train independent models")
        return None, None, None
    if not launched:
        raise ValueError("--ddp requires launching with torchrun")
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def build_scheduler(optimizer, epochs, warmup_epochs):
    if epochs <= 0 or not 0 <= warmup_epochs < epochs:
        raise ValueError("Require epochs > 0 and 0 <= warmup_epochs < epochs")
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    if warmup_epochs == 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=0.03, total_iters=warmup_epochs)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def unwrap_model(model):
    return model.module if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)) else model


def sync_batchnorm_buffers(model):
    # Do this once before uneven-length validation. DDP forward collectives
    # would deadlock when some ranks exhaust their validation shard first.
    if not dist.is_initialized():
        return
    for child in model.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(child, name, None)
                if value is not None:
                    dist.broadcast(value, src=0)


def train_one_epoch(
    model, loader, optimizer, scaler, device, epoch, rank, grad_clip=1.0,
    label_smoothing=0.1, mixup_fn=None, mixup_prob=1.0,
    amp_dtype=torch.float16, log_interval=50,
):
    model.train()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    pbar = tqdm(loader, unit="batch", desc=f"Epoch {epoch}", disable=rank not in (None, 0))
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)
        soft = mixup_fn is not None and mixup_prob > 0 and random.random() < mixup_prob
        if soft:
            x, y = mixup_fn(x, y)
            if label_smoothing > 0:
                y = y * (1.0 - label_smoothing) + label_smoothing / y.size(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(x)
            loss = (-(y * F.log_softmax(out, dim=-1)).sum(dim=-1).mean() if soft
                    else F.cross_entropy(out, y, label_smoothing=label_smoothing))
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        totals[0] += loss.detach().double() * bs
        totals[1] += bs
        if rank in (None, 0) and log_interval > 0 and (step + 1) % log_interval == 0:
            pbar.set_postfix(loss=float(loss.detach()))
    if rank is not None and dist.is_initialized():
        dist.all_reduce(totals)
    return (totals[0] / totals[1].clamp_min(1)).item()


@torch.no_grad()
def evaluate(model, loader, device, rank, amp_dtype=torch.float16):
    # DataParallel can safely retain multi-device inference. Only DDP must be
    # bypassed to avoid collectives on unequal validation shard lengths.
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # Synchronize BN stats, not the large immutable ESCNN basis/filter buffers.
    sync_batchnorm_buffers(module)
    module.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            out = module(x)
            loss = F.cross_entropy(out, y, reduction="sum")
        pred = out.topk(min(5, out.size(1)), dim=1).indices
        correct = pred.eq(y[:, None])
        totals[0] += loss.double()
        totals[1] += y.size(0)
        totals[2] += correct[:, :1].sum()
        totals[3] += correct.sum()
    if rank is not None and dist.is_initialized():
        dist.all_reduce(totals)
    loss_sum, count, top1, top5 = totals.tolist()
    if count == 0:
        raise ValueError("Validation dataset is empty")
    return loss_sum / count, top1 / count, top5 / count


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def strip_derived_state(state):
    """Normalize old DDP/eval checkpoints; ESCNN regenerates expanded filters."""
    return {
        k.removeprefix("module."): v for k, v in state.items()
        if k.split(".")[-1] not in ("filter", "expanded_bias")
    }


def load_model_state(model, state):
    # R2Conv dynamically adds/removes buffers on eval()/train(). Normalize both
    # source and destination, then use strict loading for actual learned state.
    model.train()
    model.load_state_dict(strip_derived_state(state), strict=True)


def save_training_checkpoint(
    path, model, optimizer, scheduler, epoch, top1, device,
    scaler=None, args=None, best_top1=None, best_epoch=None,
):
    """Save canonical CPU state atomically, without forcing allocator eviction."""
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state(device)
    # All ranks participate so a resumed DDP job retains each rank's RNG stream.
    rng_states = [rng]
    if dist.is_initialized():
        rng_states = [None] * dist.get_world_size()
        dist.all_gather_object(rng_states, rng)
        if dist.get_rank() != 0:
            return
    payload = {
        "epoch": epoch, "top1": top1,
        "best_top1": top1 if best_top1 is None else best_top1,
        "best_epoch": epoch if best_epoch is None else best_epoch,
        "model_state_dict": _to_cpu(strip_derived_state(unwrap_model(model).state_dict())),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _to_cpu(scheduler.state_dict()),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
        "rng_states": rng_states, "args": vars(args) if args is not None else {},
    }
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_training_state(ckpt, optimizer, scheduler, scaler, rank):
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    states = ckpt.get("rng_states", [])
    if states:
        if len(states) != (dist.get_world_size() if dist.is_initialized() else 1):
            raise ValueError("RNG resume requires the same distributed world size; use --pretrained-weights-only")
        rng = states[rank or 0]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
