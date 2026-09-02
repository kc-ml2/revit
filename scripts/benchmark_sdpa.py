"""Speed/memory comparison of the bmm attention path vs F.scaled_dot_product_attention.

Toggles the `use_sdpa` flag on the attention modules of Rot2DTransformer (global
GCSA) and Rot2DTransformerV2 (windowed GCSA) and measures, per config:
  - eval-mode forward latency + peak memory
  - train-mode forward+backward latency + peak memory
  - fp32 max output difference between the two paths (same weights)

Run on a GPU node: python benchmark_sdpa.py
"""
import contextlib
import json
import os
import statistics
import time

import torch

from group_space import get_gspace
from revit_gcsa import Rot2DTransformer
from revit_windowed_gcsa import Rot2DTransformerV2

# presets from imagenet_train_revit.py
PRESETS = {
    "small": {"dims": (24, 48, 96, 192), "depths": (1, 2, 4, 1), "heads": (1, 2, 4, 8)},
    "base": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2), "heads": (2, 4, 8, 16)},
}


def build_v2(preset, group):
    p = PRESETS[preset]
    return Rot2DTransformerV2(
        gspace=get_gspace(group), num_classes=1000, dims=p["dims"],
        depths=p["depths"], heads=p["heads"], fast_init=True,
    )


def build_v1(in_channels):
    # train_revit.py config
    return Rot2DTransformer(
        depth=4, in_channels=in_channels, channels=12, heads=6, num_classes=10,
        gspace=get_gspace("C4"), downsize=2, use_conv_attn=True, conv_kernel_size=5,
    )


CONFIGS = [
    dict(name="v2_small_C4_224_bs128", build=lambda: build_v2("small", "C4"), shape=(128, 3, 224, 224), amp=torch.float16),
    dict(name="v2_small_D4_224_bs128", build=lambda: build_v2("small", "D4"), shape=(128, 3, 224, 224), amp=torch.float16),
    dict(name="v2_base_D4_224_bs64", build=lambda: build_v2("base", "D4"), shape=(64, 3, 224, 224), amp=torch.float16),
    dict(name="v1_rotmnist_28_bs64", build=lambda: build_v1(1), shape=(64, 1, 28, 28), amp=torch.bfloat16),
    dict(name="v1_cifar_32_bs64", build=lambda: build_v1(3), shape=(64, 3, 32, 32), amp=torch.bfloat16),
    dict(name="v1_pcam_96_bs64", build=lambda: build_v1(3), shape=(64, 3, 96, 96), amp=torch.bfloat16),
    dict(name="v1_pcam_96_bs16", build=lambda: build_v1(3), shape=(16, 3, 96, 96), amp=torch.bfloat16),
]


def set_sdpa(model, flag):
    n = 0
    for m in model.modules():
        if hasattr(m, "use_sdpa"):
            m.use_sdpa = flag
            n += 1
    assert n > 0, "no attention modules found"


def bench(model, x, train, amp_dtype, warmup=10, iters=30):
    """Returns (median ms/iter, peak GiB) or 'OOM'."""
    amp = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

    def step():
        if train:
            with amp:
                loss = model(x).float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad(), amp:
                model(x)

    model.train(train)
    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times), torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return "OOM"


def correctness(model, x):
    model.eval()
    with torch.no_grad():
        set_sdpa(model, False)
        a = model(x[:4])
        set_sdpa(model, True)
        b = model(x[:4])
    return (a - b).abs().max().item()


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0)
    env = dict(
        gpu=dev, torch=torch.__version__, cuda=torch.version.cuda,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
    )
    print(f"env: {env}", flush=True)

    rows, checks = [], {}
    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        model = cfg["build"]().cuda()
        print(f"built in {time.perf_counter() - t0:.1f}s", flush=True)
        x = torch.randn(cfg["shape"], device="cuda")

        checks[name] = correctness(model, x)
        print(f"fp32 max |out_bmm - out_sdpa| = {checks[name]:.3e}", flush=True)

        for impl in ("bmm", "sdpa"):
            set_sdpa(model, impl == "sdpa")
            for dt_name, dt in (("fp32", None), ("amp", cfg["amp"])):
                fwd = bench(model, x, train=False, amp_dtype=dt)
                bwd = bench(model, x, train=True, amp_dtype=dt)
                row = dict(config=name, impl=impl, dtype=dt_name, fwd=fwd, fwd_bwd=bwd)
                rows.append(row)
                print(row, flush=True)

        del model, x
        torch.cuda.empty_cache()

    # markdown summary
    def fmt(r):
        return "OOM / OOM" if r == "OOM" else f"{r[0]:.1f} ms / {r[1]:.2f} GiB"

    print("\n| config | impl | dtype | fwd (eval) | fwd+bwd (train) |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['impl']} | {r['dtype']} | {fmt(r['fwd'])} | {fmt(r['fwd_bwd'])} |")
    print("\ncorrectness (fp32 max abs diff):")
    for k, v in checks.items():
        print(f"  {k}: {v:.3e}")

    os.makedirs("benchmark_results", exist_ok=True)
    out = f"benchmark_results/sdpa_bench_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env, correctness=checks, rows=rows), f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
