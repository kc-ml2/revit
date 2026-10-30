"""Reproducible per-image dense FLOP estimates and optional exact ESCNN params.

Multiply-add = 2 FLOPs. Excludes basis expansion, norm, activation, softmax,
pooling, data loading, and backward. This is an operation count, not latency.
The dependency-free path makes resolution/group sweeps cheap and memory safe.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_config import MODEL_PRESETS


GROUP_ORDERS = {"Z2": 1, "D2": 2, **{f"C{n}": n for n in (4, 8, 12, 16)},
                **{f"D{n}": 2 * n for n in (4, 8, 12, 16)}}


def estimate_revit(config, group_order, image_size=224, window_size=7,
                   kernel_size=1, mlp_ratio=4, stem_downsample=4, num_classes=1000):
    if min(image_size, window_size, kernel_size, group_order, mlp_ratio) <= 0:
        raise ValueError("All sizes must be positive")
    if kernel_size % 2 != 1 or stem_downsample not in (2, 4, 8):
        raise ValueError("Odd QKV kernels and stem_downsample in (2,4,8) required")
    dims, depths, heads = (config[k] for k in ("dims", "depths", "heads"))
    g, side, input_channels = group_order, image_size, 3
    stem_flops = 0
    stem_layers = int(math.log2(stem_downsample))
    for i in range(stem_layers):
        side = (side + 1) // 2
        out_channels = (dims[0] if i == stem_layers - 1 else max(1, dims[0] // 2)) * g
        kernel = 5 if i == 0 else 3
        stem_flops += 2 * side**2 * input_channels * out_channels * kernel**2
        input_channels = out_channels
    stages = []
    for i, (d, depth, nh) in enumerate(zip(dims, depths, heads)):
        c, area = d * g, side**2
        padded_side = math.ceil(side / window_size) * window_size
        padded_area = padded_side**2
        projection = depth * 2 * area * c**2 * (3 * kernel_size**2 + 1)
        mlp = depth * 4 * area * mlp_ratio * c**2
        attention = depth * 4 * padded_area * window_size**2 * c
        downsample = 0
        if i < len(dims) - 1:
            out_side = (side + 1) // 2
            downsample = 2 * out_side**2 * c * (dims[i + 1] * g) * 9
        stages.append({
            "stage": i + 1, "side": side, "fields": d, "channels": c,
            "head_dim": c // nh, "depth": depth, "padded_side": padded_side,
            "projection_flops": projection, "mlp_flops": mlp,
            "attention_flops": attention, "downsample_flops": downsample,
            # Largest one-block attention-score allocation, excluding batch/dtype.
            "score_elements_per_block": nh * padded_area * window_size**2,
            "flops": projection + mlp + attention + downsample,
        })
        side = (side + 1) // 2
    hidden = max(256, dims[-1] // 2)
    head_flops = 2 * (dims[-1] * hidden + hidden * num_classes)
    return {"stem_flops": stem_flops, "head_flops": head_flops, "stages": stages,
            "flops": stem_flops + head_flops + sum(s["flops"] for s in stages)}


def estimate_vit(image_size=224, patch_size=16, width=384, depth=12, mlp_ratio=4, num_classes=1000):
    if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
        raise ValueError("Positive image_size must be divisible by positive patch_size")
    patches = (image_size // patch_size)**2
    tokens = patches + 1
    # Torchvision uses biased linears and two LayerNorms per encoder block.
    block_params = (4 + 2 * mlp_ratio) * width**2 + (mlp_ratio + 9) * width
    params = (3 * patch_size**2 * width + width + width + tokens * width
              + depth * block_params + 2 * width + width * num_classes + num_classes)
    flops = (2 * patches * 3 * patch_size**2 * width
             + depth * (2 * tokens * width**2 * (4 + 2 * mlp_ratio) + 4 * tokens**2 * width)
             + 2 * width * num_classes)
    return {"tokens": tokens, "parameters": params, "flops": flops}


def exact_parameters(config, group, kernel_size, mlp_ratio, stem_downsample, num_classes):
    import torch
    from escnn.nn import R2Conv
    from group_space import get_gspace
    from revit_windowed_gcsa import Rot2DTransformerV2
    torch.set_num_threads(1)
    model = Rot2DTransformerV2(
        gspace=get_gspace(group), **config, qkv_kernel_size=kernel_size,
        mlp_ratio=mlp_ratio, stem_downsample=stem_downsample,
        num_classes=num_classes, fast_init=True,
    )
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "equivariant_coefficients": sum(m.weights.numel() for m in model.modules() if isinstance(m, R2Conv)),
        "buffer_elements_at_construction": sum(b.numel() for b in model.buffers()),
    }


def kernel_basis_dimensions():
    from escnn.nn import FieldType, R2Conv
    from group_space import get_gspace
    rows = []
    for name, order in GROUP_ORDERS.items():
        g = get_gspace(name)
        regular, trivial = FieldType(g, [g.regular_repr]), FieldType(g, [g.trivial_repr])
        row = {"group": name, "order": order}
        for kernel in (1, 3, 5):
            row[f"basis_regular_k{kernel}"] = R2Conv(regular, regular, kernel, bias=False, initialize=False).weights.numel()
        row["basis_lifting_k5"] = R2Conv(trivial, regular, 5, bias=False, initialize=False).weights.numel()
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-sizes", nargs="+", choices=MODEL_PRESETS, default=["tiny", "small"])
    parser.add_argument("--groups", nargs="+", choices=GROUP_ORDERS, default=["C4", "D4"])
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[224, 448])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[7])
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--stem-downsample", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--exact-parameters", action="store_true", help="Build each unique model; requires torch/escnn, may consume substantial RAM")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--basis-output", type=Path, help="Also enumerate exact single-field kernel basis dimensions; requires escnn")
    args = parser.parse_args()
    rows, parameter_cache = [], {}
    for size, group, image, window, kernel in itertools.product(
        args.model_sizes, args.groups, args.image_sizes, args.window_sizes, args.kernel_sizes
    ):
        row = {"model": size, "group": group, "group_order": GROUP_ORDERS[group],
               "image_size": image, "window_size": window, "qkv_kernel_size": kernel,
               "mlp_ratio": args.mlp_ratio, "stem_downsample": args.stem_downsample}
        row.update(estimate_revit(MODEL_PRESETS[size], GROUP_ORDERS[group], image, window,
                                  kernel, args.mlp_ratio, args.stem_downsample, args.num_classes))
        if args.exact_parameters:
            key = size, group, kernel
            if key not in parameter_cache:
                parameter_cache[key] = exact_parameters(MODEL_PRESETS[size], group, kernel,
                                                       args.mlp_ratio, args.stem_downsample, args.num_classes)
            row.update(parameter_cache[key])
        rows.append(row)
        print(f"{size:5s} {group:3s} {image:4d}px w={window:2d} k={kernel} "
              f"{row['flops']/1e9:8.3f} GFLOPs params={row.get('parameters', 'not built')}", flush=True)
    payload = {"convention": "Per-image forward, 2 FLOPs/MAC, dense conv+linear+attention matmuls only; no basis expansion",
               "revit": rows, "vit_small": [{"image_size": i, **estimate_vit(i)} for i in args.image_sizes if i % 16 == 0]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.basis_output:
        args.basis_output.parent.mkdir(parents=True, exist_ok=True)
        args.basis_output.write_text(json.dumps(kernel_basis_dimensions(), indent=2) + "\n")


if __name__ == "__main__":
    main()
