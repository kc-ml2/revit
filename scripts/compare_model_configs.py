"""Measure parameters, arithmetic estimates and latency of all three families.

Example:
  python scripts/compare_model_configs.py --groups C4 C8 D4 D8 \
    --kernel-sizes 1 3 --output analysis/model_configs/runtime_results.json

Default fast initialization is for synthetic timing only. Variants have
different capacities/weights; this script does not compare their accuracy.
"""

import argparse
import gc
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from group_complexity import MODEL_CONFIGS
from group_space import get_gspace
from model_config import MODEL_PRESETS
from revit_windowed_gcsa import Rot2DTransformerV2
from scripts.profile_scaling import GROUP_ORDERS, estimate_revit


def measure(model, x, training, dtype, warmup, iterations):
    device = x.device
    with torch.no_grad():
        model.train(training)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
        ):
            output = model(x)
            loss = output.float().square().mean()
        if training:
            loss.backward()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        step()
    model.zero_grad(set_to_none=True)
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        step()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    result = {"median_ms": statistics.median(samples), "samples_ms": samples,
              "images_per_second": x.size(0) * 1000 / statistics.median(samples),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-configs", nargs="+", choices=MODEL_CONFIGS, default=list(MODEL_CONFIGS))
    parser.add_argument("--model-size", choices=MODEL_PRESETS, default="tiny")
    parser.add_argument("--groups", nargs="+", choices=GROUP_ORDERS, default=["C4", "C8", "D4", "D8"])
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--stem-downsample", choices=[2, 4, 8], type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attention-backend", choices=["bmm", "sdpa"], default="bmm")
    parser.add_argument("--fast-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-timing", action="store_true", help="Build models and report counts only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 2 or args.iterations <= 0 or args.warmup < 0:
        parser.error("batch size >=2, iterations >0, and warmup >=0 required")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU comparison uses float32; use CUDA to compare AMP")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA device does not support BF16")
    torch.set_num_threads(args.threads)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    rows = []
    for mode, name, kernel in itertools.product(args.model_configs, args.groups, args.kernel_sizes):
        torch.manual_seed(42)
        gs = get_gspace(name)
        config = MODEL_PRESETS[args.model_size]
        print(f"Building {mode} {name} k={kernel}", flush=True)
        model = Rot2DTransformerV2(
            gspace=gs, **config, window_size=args.window_size, qkv_kernel_size=kernel,
            stem_downsample=args.stem_downsample, mlp_ratio=args.mlp_ratio,
            model_config=mode, drop_path_rate=0, fast_init=args.fast_init,
        ).to(device)
        for module in model.modules():
            if hasattr(module, "use_sdpa"):
                module.use_sdpa = args.attention_backend == "sdpa"
        stats = estimate_revit(config, gs.fibergroup.order(), args.image_size, args.window_size,
                               kernel, args.mlp_ratio, args.stem_downsample, 1000, mode,
                               "dihedral" if type(gs.fibergroup).__name__ == "DihedralGroup" else "cyclic")
        row = {"model_config": mode, "model_size": args.model_size, "group": name,
               "group_order": gs.fibergroup.order(), "qkv_kernel_size": kernel,
               "image_size": args.image_size, **stats,
               "parameters": sum(p.numel() for p in model.parameters()),
               "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}
        if not args.no_timing:
            x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
            try:
                row["forward"] = measure(model, x, False, dtype, args.warmup, args.iterations)
                row["forward_backward"] = measure(model, x, True, dtype, args.warmup, args.iterations)
                print(f"  {row['flops']/1e9:.4f} GFLOPs; {row['parameters']:,} params; "
                      f"fwd={row['forward']['median_ms']:.3f}ms; "
                      f"fwd+bwd={row['forward_backward']['median_ms']:.3f}ms", flush=True)
            except torch.cuda.OutOfMemoryError:
                row["error"] = "CUDA out of memory"
                model.zero_grad(set_to_none=True)
            del x
        rows.append(row)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": sys.version,
                            "platform": platform.platform(),
                            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
            "convention": "Per-image forward; real MAC=2, complex MAC=8; FFT estimate=2.5*N*log2(N) per signal including kernel transforms. Excludes basis expansion/norm/backward.",
            "timing_scope": "Synthetic batch forward or forward+backward, no optimizer/data pipeline; fast-init is benchmarking only; no accuracy comparison.",
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
