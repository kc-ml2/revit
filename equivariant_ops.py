"""Execution optimizations which retain ESCNN's constrained parameters."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def project_qkv(x, to_q, to_k, to_v, fused=True, packed=False):
    """Return three tensor projections, with the original checkpoint keys.

    Concatenating expanded equivariant kernels is a direct sum of the same
    three maps. No unconstrained weights are introduced. Expansion remains in
    autograd during training; evaluation uses ESCNN's cached expanded kernels.
    Only the bias-free, equal-geometry projections used in this repo are fused.
    """
    if not fused:
        outputs = tuple(layer(x).tensor for layer in (to_q, to_k, to_v))
        return torch.cat(outputs, dim=1) if packed else outputs
    layers = (to_q, to_k, to_v)
    for layer in layers:
        assert layer.in_type == x.type
        assert layer.out_type == to_q.out_type and layer.bias is None
        assert layer.padding_mode == "zeros" and layer.groups == 1
        assert (layer.stride, layer.padding, layer.dilation) == (
            to_q.stride, to_q.padding, to_q.dilation
        )
    kernels = [layer.expand_parameters()[0] if layer.training else layer.filter for layer in layers]
    output = F.conv2d(
        x.tensor, torch.cat(kernels, dim=0), stride=to_q.stride,
        padding=to_q.padding, dilation=to_q.dilation,
    )
    return output if packed else output.chunk(3, dim=1)


@contextmanager
def preserve_batchnorm_buffers(module):
    """Undo *only* running-stat updates during checkpoint recomputation.

    Keep training-mode batch normalization and the same autograd graph. Turning
    tracking off or switching to eval during recomputation changes saved tensors
    or the function being differentiated. RNG preservation is handled by torch.
    """
    snapshots = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm) and child.track_running_stats:
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(child, name, None)
                if value is not None:
                    snapshots.append((value, value.clone()))
    try:
        yield
    finally:
        with torch.no_grad():
            for value, saved in snapshots:
                value.copy_(saved)
