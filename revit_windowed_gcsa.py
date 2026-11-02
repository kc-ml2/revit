import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from escnn import gspaces
from equivariant_ops import project_qkv, preserve_batchnorm_buffers
from regular_group_ops import regular_conv
from escnn.nn import (
    FieldType,
    GeometricTensor,
    GroupPooling,
    InnerBatchNorm,
    PointwiseDropout,
    R2Conv,
    ReLU,
    init,
)


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    return total_params, trainable_params


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.tensor.shape[0],) + (1,) * (x.tensor.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.tensor.dtype, device=x.tensor.device)
        random_tensor.floor_()
        out = x.tensor.div(keep_prob) * random_tensor
        return GeometricTensor(out, x.type)


class Rot2DStem(nn.Module):
    def __init__(self, gspace, in_channels: int, out_type: FieldType, downsample: int = 4, initialize: bool = True, conv_mode: str = "quadratic"):
        super().__init__()
        if downsample not in (2, 4, 8):
            raise ValueError(f"downsample must be 2, 4, or 8, got {downsample}")
        self.downsample = downsample
        # One stride-2 layer per 2x; default 2 layers keeps the original 4x stem.
        n_layers = {2: 1, 4: 2, 8: 3}[downsample]

        self.in_type = FieldType(gspace, in_channels * [gspace.trivial_repr])
        mid = max(1, len(out_type.representations) // 2)
        mid_type = FieldType(gspace, mid * [gspace.regular_repr])

        layers = []
        for i in range(n_layers):
            if i == 0:
                in_t, k, pad = self.in_type, 5, 2
            else:
                in_t, k, pad = mid_type, 3, 1
            out_t = out_type if i == n_layers - 1 else mid_type
            conv = (R2Conv(in_t, out_t, kernel_size=k, stride=2, padding=pad,
                           bias=False, initialize=initialize) if i == 0 else
                    regular_conv(in_t, out_t, k, stride=2, initialize=initialize, mode=conv_mode))
            layers.extend([
                conv,
                InnerBatchNorm(out_t),
                ReLU(out_t),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> GeometricTensor:
        return self.net(GeometricTensor(x, self.in_type))


class Rot2DDownsample(nn.Module):
    def __init__(self, in_type: FieldType, out_type: FieldType, initialize: bool = True, conv_mode: str = "quadratic"):
        super().__init__()
        self.conv = regular_conv(in_type, out_type, 3, stride=2, initialize=initialize, mode=conv_mode)
        self.bn = InnerBatchNorm(out_type)
        self.act = ReLU(out_type)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.act(self.bn(self.conv(x)))


class Rot2DWindowAttention(nn.Module):
    def __init__(
        self,
        in_type: FieldType,
        num_heads: int,
        window_size: int = 7,
        qkv_kernel_size: int = 1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        initialize: bool = True,
        conv_mode: str = "quadratic",
    ):
        super().__init__()
        if num_heads <= 0 or len(in_type.representations) % num_heads:
            raise ValueError("The positive head count must divide the regular-field count")
        if window_size <= 0 or qkv_kernel_size <= 0 or qkv_kernel_size % 2 != 1:
            raise ValueError("window_size must be positive and qkv_kernel_size positive and odd")
        if any(r != in_type.gspace.regular_repr for r in in_type.representations):
            raise ValueError("Attention heads require complete regular fields")

        self.in_type = in_type
        self.conv_mode = conv_mode
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_size = len(in_type.representations) // num_heads
        self.head_dim = self.head_size * in_type.representations[0].size
        # benchmark (benchmark_results/sdpa_bench_1361213.json): SDPA is 0-4% slower
        # here — 7x7 windows are too small for fused kernels to win — so bmm stays
        # the default. Global attention (revit_gcsa.py) defaults to SDPA instead.
        self.use_sdpa = False
        self.fused_qkv = conv_mode == "quadratic"

        self.to_q = regular_conv(in_type, in_type, qkv_kernel_size, initialize=initialize, mode=conv_mode)
        self.to_k = regular_conv(in_type, in_type, qkv_kernel_size, initialize=initialize, mode=conv_mode)
        self.to_v = regular_conv(in_type, in_type, qkv_kernel_size, initialize=initialize, mode=conv_mode)
        self.to_out = regular_conv(in_type, in_type, initialize=initialize, mode=conv_mode)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = PointwiseDropout(in_type, p=proj_dropout)

    def _window_partition(self, x: torch.Tensor):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x.view(-1, c, ws, ws)
        return windows, (h, w, hp, wp)

    def _window_reverse(self, windows: torch.Tensor, meta):
        h, w, hp, wp = meta
        ws = self.window_size
        bnw, c, _, _ = windows.shape
        nh, nw = hp // ws, wp // ws
        b = bnw // (nh * nw)
        x = windows.view(b, nh, nw, c, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        if self.conv_mode == "quadratic":
            qkv = project_qkv(x, self.to_q, self.to_k, self.to_v, self.fused_qkv, packed=True)
        else:
            # Structured operators do not expand dense QKV kernels. Each uses
            # its native sparse/FFT path, then shares the window packing below.
            qkv = torch.cat([layer(x).tensor for layer in (self.to_q, self.to_k, self.to_v)], dim=1)
        # Partition once while projections share a tensor. Head boundaries still
        # contain complete regular fields, independently for Q, K and V.
        windows, meta = self._window_partition(qkv)
        bnw, three_c, ws, _ = windows.shape
        c = three_c // 3
        n = ws * ws
        qh, kh, vh = windows.view(bnw, 3, self.num_heads, self.head_dim, n).permute(1, 0, 2, 4, 3).unbind(0)

        if self.use_sdpa:
            # same math as below; scale defaults to 1/sqrt(head_dim).
            # contiguous: fused SDPA kernels require stride-1 last dim, else silent math fallback
            out = F.scaled_dot_product_attention(
                qh.contiguous(), kh.contiguous(), vh.contiguous(),
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            attn = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, vh)
        out = out.transpose(-2, -1).contiguous().view(bnw, c, ws, ws)

        out = self._window_reverse(out, meta)
        out = GeometricTensor(out, self.in_type)
        out = self.to_out(out)
        return self.proj_dropout(out)


class Rot2DMLP(nn.Module):
    def __init__(self, in_type: FieldType, hidden_mult: int = 4, dropout: float = 0.0, initialize: bool = True, conv_mode: str = "quadratic"):
        super().__init__()
        hidden_type = FieldType(in_type.gspace, hidden_mult * in_type.representations)
        self.net = nn.Sequential(
            regular_conv(in_type, hidden_type, initialize=initialize, mode=conv_mode),
            InnerBatchNorm(hidden_type),
            ReLU(hidden_type),
            PointwiseDropout(hidden_type, p=dropout),
            regular_conv(hidden_type, in_type, initialize=initialize, mode=conv_mode),
            InnerBatchNorm(in_type),
            PointwiseDropout(in_type, p=dropout),
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.net(x)


class Rot2DBlockV2(nn.Module):
    def __init__(
        self,
        in_type: FieldType,
        num_heads: int,
        window_size: int,
        mlp_ratio: int,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
        qkv_kernel_size: int,
        use_checkpoint: bool,
        initialize: bool = True,
        conv_mode: str = "quadratic",
    ):
        super().__init__()
        self.norm1 = InnerBatchNorm(in_type)
        self.norm2 = InnerBatchNorm(in_type)
        self.attn = Rot2DWindowAttention(
            in_type,
            num_heads=num_heads,
            window_size=window_size,
            qkv_kernel_size=qkv_kernel_size,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            initialize=initialize,
            conv_mode=conv_mode,
        )
        self.mlp = Rot2DMLP(in_type, hidden_mult=mlp_ratio, dropout=dropout, initialize=initialize, conv_mode=conv_mode)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: GeometricTensor) -> GeometricTensor:
        x = x + self.drop_path1(self.alpha1 * self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.alpha2 * self.mlp(self.norm2(x)))
        return x

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        if not (self.use_checkpoint and self.training):
            return self._forward_impl(x)

        xtype = x.type

        def _fn(tensor):
            gx = GeometricTensor(tensor, xtype)
            return self._forward_impl(gx).tensor

        out_t = checkpoint.checkpoint(
            _fn, x.tensor, use_reentrant=False,
            context_fn=lambda: (nullcontext(), preserve_batchnorm_buffers(self)),
        )
        return GeometricTensor(out_t, xtype)


class Rot2DStage(nn.Module):
    def __init__(self, blocks: nn.ModuleList, downsample: Optional[Rot2DDownsample]):
        super().__init__()
        self.blocks = blocks
        self.downsample = downsample

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Rot2DClassificationHeadV2(nn.Module):
    def __init__(self, in_type: FieldType, gspace, num_classes: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.pool = GroupPooling(in_type)
        n = gspace.fibergroup.order() if gspace.fibergroup.order() > 0 else 1
        in_features = in_type.size // n
        hidden = max(256, in_features // 2)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: GeometricTensor, return_probs: bool = False):
        x = self.pool(x).tensor
        x = x.mean(dim=(2, 3))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if return_probs:
            return torch.softmax(x, dim=1)
        return x


class Rot2DTransformerV2(nn.Module):
    def __init__(
        self,
        gspace=gspaces.rot2dOnR2(N=4),
        in_channels: int = 3,
        num_classes: int = 1000,
        dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: Tuple[int, int, int, int] = (2, 4, 8, 16),
        window_size: int = 7,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        qkv_kernel_size: int = 1,
        use_checkpoint: bool = False,
        fast_init: bool = False,
        stem_downsample: int = 4,
        model_config: str = "quadratic",
    ):
        super().__init__()
        if not (len(dims) == len(depths) == len(heads) == 4):
            raise ValueError("dims, depths and heads must each contain four entries")
        if any(d <= 0 or n <= 0 or h <= 0 or d % h for d, n, h in zip(dims, depths, heads)):
            raise ValueError("Positive stage dimensions and depths required; heads must divide dims")
        if mlp_ratio <= 0 or not 0 <= drop_path_rate < 1:
            raise ValueError("mlp_ratio must be positive and drop_path_rate in [0, 1)")
        if model_config not in ("quadratic", "linear", "fourier"):
            raise ValueError("model_config must be quadratic, linear, or fourier")
        self.model_config = model_config
        self.gspace = gspace
        self.fast_init = fast_init

        stage_types = [FieldType(gspace, d * [gspace.regular_repr]) for d in dims]
        self.stem = Rot2DStem(gspace, in_channels, stage_types[0], downsample=stem_downsample, initialize=False, conv_mode=model_config)

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dpr_idx = 0
        stages = []

        for i in range(len(dims)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    Rot2DBlockV2(
                        in_type=stage_types[i],
                        num_heads=heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=dpr[dpr_idx],
                        qkv_kernel_size=qkv_kernel_size,
                        use_checkpoint=use_checkpoint,
                        initialize=False,
                        conv_mode=model_config,
                    )
                )
                dpr_idx += 1

            downsample = Rot2DDownsample(stage_types[i], stage_types[i + 1], initialize=False, conv_mode=model_config) if i < (len(dims) - 1) else None
            stages.append(Rot2DStage(nn.ModuleList(blocks), downsample))

        self.stages = nn.ModuleList(stages)
        self.head = Rot2DClassificationHeadV2(stage_types[-1], gspace, num_classes=num_classes, dropout=dropout)
        self._initialize_weights()
        # R2Conv constructors allocate zero-valued eval filter buffers even in
        # training mode. Drop them before .to(device); eval() rebuilds them from
        # learned coefficients. This can save hundreds of MB during GPU setup.
        self.train()

    def _initialize_weights(self):
        has_general_orth = hasattr(init, "general_orthogonal_init")
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, R2Conv):
                    if self.fast_init:
                        # Fast debug path: skip expensive ESCNN basis-aware init.
                        # ESCNN stores coefficients in 1D tensors for some layers,
                        # so use a shape-agnostic init instead of fan-based Kaiming.
                        nn.init.normal_(m.weights, mean=0.0, std=0.02)
                    elif has_general_orth:
                        init.general_orthogonal_init(m.weights, m.basisexpansion)
                    else:
                        init.deltaorthonormal_init(m.weights, m.basisexpansion)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, InnerBatchNorm)):
                    if hasattr(m, "weight") and m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x, return_probs=return_probs)


if __name__ == "__main__":
    g = gspaces.rot2dOnR2(N=4)
    # Fast smoke-test settings for quick local runs.
    m = Rot2DTransformerV2(
        gspace=g,
        in_channels=3,
        num_classes=1000,
        dims=(32, 64, 128, 256),
        depths=(1, 1, 2, 1),
        heads=(2, 2, 4, 8),
        fast_init=True,
    )
    count_parameters(m)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)
    # count_parameters(m)
