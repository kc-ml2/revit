"""Oriented bounding-box detector on a pretrained REViT D4-base backbone.

Architecture
------------
1. Equivariant hierarchical backbone (`Rot2DTransformerV2`, D4, base preset).
2. Equivariant FPN (``R2Conv``) keeps stage features as GeometricTensors.
3. Equivariant dense towers for classification and OBB regression.
4. Classification becomes invariant via ``GroupPooling`` before class logits.
5. OBB is predicted in a steerable field type so vectors / orientation transform
   under D4 rather than collapsing to invariants too early.

OBB field layout (6 channels after unpacking ``.tensor``):
    (log_w, log_h, dx, dy, cos2θ, sin2θ)
- ``log_w``, ``log_h``: trivial fields (rotation-invariant sizes)
- ``(dx, dy)``: D4 ``irrep(1, 1)`` vector (center offset)
- ``(cos2θ, sin2θ)``: D4 ``irrep(1, 1)`` vector (π-periodic orientation)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ESCNN passes np.matrix into sklearn.randomized_svd; sklearn>=1.8 rejects that.
try:
    import escnn.group._numerical as _escnn_num

    _orig_null = _escnn_num.null

    def _null_compat(A, *args, **kwargs):
        if isinstance(A, np.matrix):
            A = np.asarray(A)
        return _orig_null(A, *args, **kwargs)

    _escnn_num.null = _null_compat
except Exception:
    pass

from escnn.nn import FieldType, GeometricTensor, GroupPooling, InnerBatchNorm, R2Conv, ReLU, init

from group_space import get_gspace
from revit_backbone import Rot2DTransformerV2, count_parameters

# Matches imagenet_train_revit.py "base" preset used for the D4 ImageNet run.
REVIT_BASE_PRESET = {
    "dims": (32, 64, 128, 256),
    "depths": (2, 2, 4, 2),
    "heads": (1, 2, 4, 8),
}

DEFAULT_D4_BASE_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "imagenet_es_v2_outputs",
    "imagenet_revit_D4_base_best_top1_0.8055_epoch_282.pth",
)


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    k0 = next(iter(state_dict.keys()))
    if k0.startswith("module."):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_legacy_stem_keys(state_dict: dict) -> dict:
    """Map pre-Sequential stem names (conv1/bn1/...) to stem.net.* indices."""
    mapping = {
        "stem.conv1.": "stem.net.0.",
        "stem.bn1.": "stem.net.1.",
        "stem.conv2.": "stem.net.3.",
        "stem.bn2.": "stem.net.4.",
    }
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in mapping.items():
            if k.startswith(old):
                new_k = new + k[len(old) :]
                break
        out[new_k] = v
    return out


def load_backbone_checkpoint(
    backbone: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load ImageNet REViT weights into a backbone (skips classification head)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    state = _remap_legacy_stem_keys(_strip_module_prefix(state))
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = backbone.load_state_dict(state, strict=strict)
    return list(missing), list(unexpected)


def build_revit_d4_base_backbone(
    pretrained_path: Optional[str] = DEFAULT_D4_BASE_CKPT,
    window_size: int = 7,
    mlp_ratio: int = 4,
    dropout: float = 0.0,
    attn_dropout: float = 0.0,
    drop_path_rate: float = 0.1,
    qkv_kernel_size: int = 3,
    use_checkpoint: bool = False,
    fast_init: bool = False,
    freeze_backbone: bool = False,
) -> Rot2DTransformerV2:
    """Construct D4-base REViT and optionally load ImageNet-pretrained weights."""
    gspace = get_gspace("D4")
    backbone = Rot2DTransformerV2(
        gspace=gspace,
        in_channels=3,
        num_classes=None,
        dims=REVIT_BASE_PRESET["dims"],
        depths=REVIT_BASE_PRESET["depths"],
        heads=REVIT_BASE_PRESET["heads"],
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        drop_path_rate=drop_path_rate,
        qkv_kernel_size=qkv_kernel_size,
        use_checkpoint=use_checkpoint,
        fast_init=fast_init,
    )
    if pretrained_path is not None:
        missing, unexpected = load_backbone_checkpoint(backbone, pretrained_path)
        if unexpected:
            raise RuntimeError(f"Unexpected backbone keys while loading: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing backbone keys while loading: {missing}")

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    return backbone


def _regular_type(gspace, n_fields: int) -> FieldType:
    return FieldType(gspace, n_fields * [gspace.regular_repr])


def _vector_repr(gspace):
    """Standard 2D vector irrep of D4 (``irrep(1, 1)``)."""
    return gspace.irrep(1, 1)


def _geo_upsample(x: GeometricTensor, size: Tuple[int, int]) -> GeometricTensor:
    return GeometricTensor(F.interpolate(x.tensor, size=size, mode="nearest"), x.type)


def _init_r2convs(module: nn.Module, fast_init: bool = True):
    has_general_orth = hasattr(init, "general_orthogonal_init")
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, R2Conv):
                if fast_init:
                    nn.init.normal_(m.weights, mean=0.0, std=0.02)
                elif has_general_orth:
                    init.general_orthogonal_init(m.weights, m.basisexpansion)
                else:
                    init.deltaorthonormal_init(m.weights, m.basisexpansion)


class EquivariantConvBNAct(nn.Module):
    def __init__(self, in_type: FieldType, out_type: FieldType, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = InnerBatchNorm(out_type)
        self.act = ReLU(out_type)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.act(self.bn(self.conv(x)))


class EquivariantFPN(nn.Module):
    """Top-down FPN with equivariant lateral / smooth convolutions."""

    def __init__(self, in_types: Sequence[FieldType], out_type: FieldType):
        super().__init__()
        assert len(in_types) >= 1
        self.out_type = out_type
        self.lateral = nn.ModuleList(
            [R2Conv(t, out_type, kernel_size=1, bias=False) for t in in_types]
        )
        self.smooth = nn.ModuleList(
            [EquivariantConvBNAct(out_type, out_type, kernel_size=3, padding=1) for _ in in_types]
        )

    def forward(self, features: Sequence[GeometricTensor]) -> List[GeometricTensor]:
        assert len(features) == len(self.lateral)
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = _geo_upsample(laterals[i], laterals[i - 1].tensor.shape[-2:])
            laterals[i - 1] = laterals[i - 1] + up
        return [sm(lat) for sm, lat in zip(self.smooth, laterals)]


class EquivariantDenseOBBHead(nn.Module):
    """Equivariant RetinaNet-style towers for classification and oriented boxes."""

    def __init__(
        self,
        in_type: FieldType,
        num_classes: int,
        num_convs: int = 4,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        gspace = in_type.gspace
        self.in_type = in_type

        cls_layers = []
        reg_layers = []
        for _ in range(num_convs):
            cls_layers.append(EquivariantConvBNAct(in_type, in_type, kernel_size=3, padding=1))
            reg_layers.append(EquivariantConvBNAct(in_type, in_type, kernel_size=3, padding=1))
        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)

        # Invariant class scores: pool regular fields → trivial, then 1x1 to num_classes.
        self.cls_pool = GroupPooling(in_type)
        pooled_type = self.cls_pool.out_type
        self.cls_logits = R2Conv(
            pooled_type,
            FieldType(gspace, num_classes * [gspace.trivial_repr]),
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Steerable OBB: (log_w, log_h) trivials + (dx, dy) vector + (cos2θ, sin2θ) vector.
        vec = _vector_repr(gspace)
        self.obb_type = FieldType(
            gspace,
            [gspace.trivial_repr, gspace.trivial_repr, vec, vec],
        )
        self.obb_pred = R2Conv(in_type, self.obb_type, kernel_size=3, padding=1, bias=True)

        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        with torch.no_grad():
            if self.cls_logits.bias is not None:
                self.cls_logits.bias.fill_(bias_value)
            if self.obb_pred.bias is not None:
                self.obb_pred.bias.zero_()

    def forward(
        self, features: Sequence[GeometricTensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_outs, obb_outs = [], []
        for f in features:
            cls = self.cls_logits(self.cls_pool(self.cls_tower(f)))
            obb = self.obb_pred(self.reg_tower(f))
            cls_outs.append(cls.tensor)  # (B, num_classes, H, W)
            obb_outs.append(obb.tensor)  # (B, 6, H, W): log_w, log_h, dx, dy, cos2θ, sin2θ
        return cls_outs, obb_outs


class REViTOBBDetector(nn.Module):
    """REViT D4-base backbone + equivariant FPN + equivariant cls / OBB heads."""

    def __init__(
        self,
        num_classes: int,
        pretrained_path: Optional[str] = DEFAULT_D4_BASE_CKPT,
        fpn_dim: int = 32,
        fpn_start_stage: int = 1,
        head_convs: int = 4,
        freeze_backbone: bool = False,
        window_size: int = 7,
        qkv_kernel_size: int = 3,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        fast_init: bool = False,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        dims = REVIT_BASE_PRESET["dims"]
        if not (0 <= fpn_start_stage < len(dims)):
            raise ValueError(f"fpn_start_stage must be in [0, {len(dims) - 1}]")

        self.num_classes = num_classes
        self.fpn_start_stage = fpn_start_stage
        self.gspace = get_gspace("D4")

        self.backbone = build_revit_d4_base_backbone(
            pretrained_path=pretrained_path,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            qkv_kernel_size=qkv_kernel_size,
            use_checkpoint=use_checkpoint,
            fast_init=fast_init,
            freeze_backbone=freeze_backbone,
        )

        used_types = self.backbone.stage_types[fpn_start_stage:]
        fpn_type = _regular_type(self.gspace, fpn_dim)
        self.fpn = EquivariantFPN(used_types, fpn_type)
        self.head = EquivariantDenseOBBHead(fpn_type, num_classes=num_classes, num_convs=head_convs)
        _init_r2convs(self.fpn, fast_init=True)
        _init_r2convs(self.head, fast_init=True)

        # Stage i (0-based) feature map before its downsample has stride 4 * 2^i.
        self.strides = tuple(4 * (2**i) for i in range(fpn_start_stage, len(dims)))

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        stage_feats = self.backbone.forward_features(x, return_stages=True)
        stage_feats = stage_feats[self.fpn_start_stage :]
        fpn_feats = self.fpn(stage_feats)
        cls_logits, obb_preds = self.head(fpn_feats)
        return {
            "cls_logits": cls_logits,  # list of (B, num_classes, H_i, W_i)
            "obb_preds": obb_preds,  # list of (B, 6, H_i, W_i)
            "strides": list(self.strides),
        }


def build_revit_obb_d4_base(
    num_classes: int,
    pretrained_path: Optional[str] = DEFAULT_D4_BASE_CKPT,
    **kwargs,
) -> REViTOBBDetector:
    """Factory for the D4-base REViT oriented detector."""
    return REViTOBBDetector(num_classes=num_classes, pretrained_path=pretrained_path, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_revit_obb_d4_base(
        num_classes=15,
        pretrained_path=None,  # skip slow ckpt load for smoke test
        fast_init=True,
    ).to(device)
    count_parameters(model)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x)
    for i, (c, o) in enumerate(zip(out["cls_logits"], out["obb_preds"])):
        print(
            f"level {i}: stride={out['strides'][i]}  "
            f"cls={tuple(c.shape)}  obb={tuple(o.shape)}"
        )
