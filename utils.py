import inspect
import random
import copy

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.flop_counter import FlopCounterMode


class RandomRot2DRotation:
    def __call__(self, x):
        k = random.randint(0, 3)  # 0,1,2,3 → 0°,90°,180°,270°
        return TF.rotate(x, angle=90 * k)

class EarlyStopping:
    """Early stopping utility to stop training when validation loss/accuracy stops improving."""
    
    def __init__(self, patience=20, min_delta=0.0, mode='max', restore_best_weights=True, verbose=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
            restore_best_weights: Whether to restore model weights from best epoch
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, score, model, epoch):
        """
        Args:
            score: Current validation score (accuracy or loss)
            model: Model to save weights from
            epoch: Current epoch number
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✓ Improvement! New best score: {score:.4f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop
    
    def _is_better(self, current, best):
        """Check if current score is better than best score."""
        if self.mode == 'max':
            return current > (best + self.min_delta)
        else:  # mode == 'min'
            return current < (best - self.min_delta)
    
    def restore_weights(self, model):
        """Restore model weights to best checkpoint."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored weights from epoch {self.best_epoch} (score: {self.best_score:.4f})")


def _unwrap(model):
    while isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def _readable_flops(flops):
    for scale, unit in ((1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs"), (1e3, "KFLOPs")):
        if flops >= scale:
            return f"{flops / scale:.3f} {unit}"
    return f"{flops} FLOPs"


def _sdpa_flops(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):
    """Match torch.utils.flop_counter.sdpa_flop_count. One MAC is two FLOPs."""
    batch, heads, queries, dim = query_shape
    keys = key_shape[2]
    value_dim = value_shape[-1]
    return 2 * batch * heads * queries * keys * (dim + value_dim)


def _native_mha_flops(
    query_shape, key_shape, value_shape, embed_dim, num_heads,
    qkv_weight_shape, qkv_bias_shape, proj_weight_shape, proj_bias_shape,
    *args, out_shape=None, **kwargs,
):
    """FLOPs for aten._native_multi_head_attention, including its projections."""
    batch, queries, _dim = query_shape
    keys, values = key_shape[-2], value_shape[-2]
    qkv_out, qkv_in = qkv_weight_shape
    proj_out, proj_in = proj_weight_shape
    projections = 2 * batch * qkv_in * (qkv_out // 3) * (queries + keys + values)
    projections += 2 * batch * queries * proj_in * proj_out
    head_dim = embed_dim // num_heads
    value_dim = value_shape[-1] // num_heads
    attention = 2 * batch * num_heads * queries * keys * (head_dim + value_dim)
    return projections + attention


# These kernels are missing from PyTorch's flop registry. CUDA flash, efficient,
# and cuDNN attention are already counted.
_EXTRA_FLOPS = {
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_flops,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable: _sdpa_flops,
    torch.ops.aten._native_multi_head_attention: _native_mha_flops,
}


def _flop_forward(model, images):
    """One inference forward. Nested models are measured at their widest profile."""
    profiles = getattr(model, "profiles", None)
    if profiles and "profile" in inspect.signature(model.forward).parameters:
        return model(images, profile=getattr(model, 'reference_profile', len(profiles) - 1)), len(profiles) > 1
    return model(images), False


def count_flops(model: nn.Module, image_shape):
    """Print FLOPs of one image and return the count.

    image_shape is (channels, height, width), the resolution training will see.
    The count is multiply-adds as reported by PyTorch (one MAC is two FLOPs).
    """
    if len(image_shape) != 3:
        raise ValueError("image_shape must be (channels, height, width)")
    core = _unwrap(model)
    device = next(core.parameters()).device
    example = torch.zeros(1, *image_shape, device=device, dtype=next(core.parameters()).dtype)
    was_training = core.training
    core.eval()
    try:
        with torch.inference_mode():
            # Warm up so filter expansions that are cached after the first eval pass
            # are not charged against the steady-state forward.
            _flop_forward(core, example)
            with FlopCounterMode(display=False, custom_mapping=_EXTRA_FLOPS) as counter:
                _, widest = _flop_forward(core, example)
    finally:
        core.train(was_training)
    flops = counter.get_total_flops()
    shape = "x".join(str(size) for size in example.shape)
    profile_note = ", widest profile" if widest else ""
    print(f"FLOPs ({shape}{profile_note}): {flops} ({_readable_flops(flops)})")
    return flops
