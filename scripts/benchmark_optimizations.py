"""Benchmark full REViT forward/backward, optionally against a Git revision.

Example: python scripts/benchmark_optimizations.py --baseline-ref 2558962 \
    --group D4 --model-size tiny --image-size 224 --output analysis/cpu_benchmark.json
For deployment numbers, run on the target CUDA node with its actual batch size.
"""

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from group_space import get_gspace
from imagenet_utils import load_model_state, strip_derived_state
from model_config import MODEL_PRESETS
from revit_windowed_gcsa import Rot2DTransformerV2


def load_baseline(ref, directory):
    source = subprocess.run(["git", "show", f"{ref}:revit_windowed_gcsa.py"],
                            cwd=Path(__file__).resolve().parents[1], check=True,
                            capture_output=True, text=True).stdout
    path = Path(directory) / "baseline_model.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("baseline_revit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Rot2DTransformerV2


def configure(model, fused, sdpa):
    for module in model.modules():
        if hasattr(module, "use_sdpa"):
            module.use_sdpa = sdpa
            module.fused_qkv = fused


def benchmark(model, x, training, dtype, warmup, iterations):
    device = x.device
    with torch.no_grad():
        model.train(training)
    def step():
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32
        ):
            out = model(x)
            if training:
                out.float().square().mean().backward()
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        step()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        step()
        synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return {"median_ms": statistics.median(timings), "samples_ms": timings,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--model-size", choices=MODEL_PRESETS, default="tiny")
    parser.add_argument("--group", default="C4")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--qkv-kernel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--fast-init", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.batch_size < 2:
        parser.error("iterations > 0, warmup >= 0, batch size >= 2 required")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU measurements currently use float32; choose CUDA for AMP measurements")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("This CUDA device does not support bfloat16")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    kwargs = dict(gspace=get_gspace(args.group), **MODEL_PRESETS[args.model_size],
                  window_size=args.window_size, qkv_kernel_size=args.qkv_kernel_size,
                  drop_path_rate=0, fast_init=args.fast_init)
    with tempfile.TemporaryDirectory() as directory:
        baseline_class = load_baseline(args.baseline_ref, directory) if args.baseline_ref else Rot2DTransformerV2
        started = time.perf_counter()
        baseline = baseline_class(**kwargs)
        baseline_init = time.perf_counter() - started
        print(f"Baseline constructed in {baseline_init:.3f}s", flush=True)
        started = time.perf_counter()
        optimized = Rot2DTransformerV2(**kwargs)
        optimized_init = time.perf_counter() - started
        print(f"Optimized constructed in {optimized_init:.3f}s", flush=True)
        initial_state = {k: v.detach().cpu().clone() for k, v in strip_derived_state(baseline.state_dict()).items()}
        load_model_state(optimized, initial_state)
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    configure(baseline, False, False)
    with torch.no_grad():
        baseline.to(device)
        baseline.eval()
        expected = baseline(x)
        baseline_invariance = (baseline(x.rot90(1, (-2, -1))) - expected).abs().max().item()
    rows, checks = [], []
    for name, model, fused, sdpa in (
        ("baseline_bmm_separate_qkv", baseline, False, False),
        ("optimized_bmm_fused_qkv", optimized, True, False),
        ("optimized_sdpa_fused_qkv", optimized, True, True),
    ):
        # Keep only the current model on CUDA so peak allocation does not
        # accidentally count both implementations and two sets of gradients.
        for other in (baseline, optimized):
            if other is not model:
                other.cpu()
        model.to(device)
        load_model_state(model, initial_state)
        configure(model, fused, sdpa)
        with torch.no_grad():
            model.eval()
            actual = model(x)
            checks.append({"variant": name, "fp32_max_abs_logit_difference": (actual - expected).abs().max().item(),
                           "rotation90_max_abs_logit_difference": (model(x.rot90(1, (-2, -1))) - actual).abs().max().item()})
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
        for training in (False, True):
            row = {"variant": name, "training": training,
                   **benchmark(model, x, training, dtype, args.warmup, args.iterations)}
            rows.append(row)
            print(f"{name} train={training}: {row['median_ms']:.3f} ms", flush=True)
    payload = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "environment": {"torch": torch.__version__, "platform": platform.platform(),
                               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
               "construction_seconds": {"baseline": baseline_init, "optimized": optimized_init},
               "baseline_rotation90_logit_error": baseline_invariance, "checks": checks, "rows": rows,
               "notes": "Synthetic batches; training = forward+backward without optimizer/data loader. CPU results do not predict CUDA speed."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
