"""Benchmark, evaluate, and export Budget-adaptive REViT checkpoints."""

import argparse
from contextlib import ExitStack, contextmanager
from itertools import islice
import json
import math
from pathlib import Path
import statistics
import time

import torch

from budget_cli import load_budget_model, model_config
from budget_revit import BudgetAdaptiveREViT, BudgetREViTConfig, ProfileConv, ProfileNorm


def autocast(device, precision):
    return torch.autocast(device.type, enabled=precision != 'fp32',
                          dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}.get(precision))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


@contextmanager
def module_ranges(model):
    handles = []
    stack = ExitStack()

    def annotate(name):
        def start(module, args):
            stack.enter_context(torch.profiler.record_function(name))
        return start

    def end(module, args, output):
        # Each timed module here is a leaf relative to the other timed modules.
        stack.close()

    try:
        for name, module in model.named_modules():
            if isinstance(module, (ProfileConv, ProfileNorm)) or name == 'router':
                handles.append(module.register_forward_pre_hook(annotate(name)))
                handles.append(module.register_forward_hook(end, always_call=True))
        yield
    finally:
        stack.close()
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def benchmark(args):
    device = torch.device(args.device)
    if args.checkpoint:
        model = load_budget_model(args.checkpoint).eval()
    else:
        window = args.smoke_backbone == 'windowed'
        profiles = ((2, 4), (4, 8), (6, 12)) if window else ((2,), (4,), (6,))
        model = BudgetAdaptiveREViT(BudgetREViTConfig(
            profiles=profiles, depths=(1, 1) if window else (1,), heads=(1, 2) if window else (1,),
            image_size=32, window_size=7 if window else None, group=args.group,
        )).eval()
    if args.pointwise_backend == 'fourier':
        model.use_cyclic_fourier()

    def measure(run, inputs):
        with autocast(device, args.precision):
            for index in range(max(args.warmup, len(inputs))):
                run(inputs[index % len(inputs)].to(device))
            synchronize(device)
            if device.type == 'cuda':
                resident = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for index in range(args.iterations):
                x = inputs[index % len(inputs)].to(device)
                synchronize(device)
                start = time.perf_counter()
                run(x)
                synchronize(device)
                samples.append((time.perf_counter() - start) * 1e3)
                del x
        result = {
            'p50_ms': statistics.median(samples),
            'p95_ms': sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
            'images_per_second': 1e3 * len(inputs[0]) / statistics.mean(samples),
        }
        if device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device)
            result.update(peak_allocated_bytes=peak, resident_allocated_bytes=resident,
                          peak_above_resident_bytes=peak - resident)
        return result

    # Keep each measured model separate so its resident filter memory is counted once.
    jobs = [(f'fixed_{i}', i, None) for i in range(len(model.profiles))]
    budgets = args.budgets if args.budgets is not None else model.budget_thresholds
    if model.router is not None:
        jobs += [('adaptive', None, budget) for budget in budgets]
    inputs_by_size = {}
    size, channels = model.config.image_size, model.config.in_channels
    for batch in args.batch_sizes:
        input_rng = torch.Generator().manual_seed(args.seed)
        if args.dataset is None:
            inputs = [torch.randn(batch, channels, size, size, generator=input_rng)]
        else:
            batches = validation_loader(args, device, batch_size=batch)
            full = (images for images, _ in batches if len(images) == batch)
            inputs = [orient(images, args.orientation, input_rng) for images in islice(full, args.input_batches)]
            if not inputs:
                raise ValueError(f'The evaluation set cannot supply a full batch of {batch}')
        inputs_by_size[batch] = inputs
    records = []
    for mode, profile, budget in jobs:
        candidate = model if profile is None else model.export_profile(profile)
        candidate.to(device)
        if args.compile_profiles or args.prewarm_all_profiles:
            # Warm the selected-width filters before capturing the tensor-only paths.
            example = torch.zeros(1, model.config.in_channels, model.config.image_size,
                                  model.config.image_size, device=device)
            for index in range(len(candidate.profiles)):
                candidate(example, profile=index)
            del example
        if args.compile_profiles:
            candidate._predict = torch.compile(candidate._predict, dynamic=True)
        for batch in args.batch_sizes:
            inputs = inputs_by_size[batch]

            def run(tensor):
                return candidate(tensor, budget=budget)

            stats = measure(run, inputs)
            histogram = torch.zeros(len(candidate.profiles), dtype=torch.long)
            with autocast(device, args.precision):
                for images in inputs:
                    _, routes = candidate(images.to(device), budget=budget, return_routes=True)
                    histogram += torch.bincount(routes, minlength=len(candidate.profiles)).cpu()
            records.append(dict(mode=mode, budget=budget, batch_size=batch, routes=histogram.tolist(),
                                input_batches=len(inputs),
                                parameters=sum(p.numel() for p in candidate.parameters()), **stats))
            if args.trace:
                x = inputs[0].to(device)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                label = f'{mode}-b{batch}-cap{budget}'
                with module_ranges(candidate), torch.profiler.profile(
                    activities=activities, record_shapes=True, profile_memory=True,
                ) as profiler, autocast(device, args.precision):
                    with torch.profiler.record_function(label):
                        run(x)
                        synchronize(device)
                profiler.export_chrome_trace(str(args.trace / f'{label}.json'))
                del x
            del routes
        if args.compile_profiles:
            del candidate._predict
        candidate.cpu()
    return {
        'device': str(device), 'torch': str(torch.__version__), 'precision': args.precision,
        'pointwise_backend': args.pointwise_backend,
        'synthetic_inputs': args.dataset is None, 'random_weights': args.checkpoint is None,
        'dataset': args.dataset, 'orientation': args.orientation,
        'compiled_profiles': args.compile_profiles, 'reference_size': model.config.image_size,
        'prewarmed_all_profiles': args.prewarm_all_profiles or args.compile_profiles,
        'threads': torch.get_num_threads(), 'warmup': args.warmup, 'iterations': args.iterations,
        'profiles': model.profiles, 'profile_costs': model.profile_costs,
        'budget_thresholds': model.budget_thresholds, 'results': records,
    }


def validation_loader(args, device, batch_size=None):
    from torch.utils.data import DataLoader

    benchmark_batches = batch_size is not None
    batch_size = args.batch_size if batch_size is None else batch_size
    if args.dataset == 'imagenet':
        from torchvision.datasets import ImageFolder
        from imagenet_train_revit import imagenet_validation_transform

        dataset = ImageFolder(args.data_root / 'val', transform=imagenet_validation_transform())
    elif args.dataset == 'pcam':
        from datasets import get_pcam_loaders

        root = {} if args.data_root is None else {'root': args.data_root}
        dataset = get_pcam_loaders(**root, eval_split=args.pcam_split)[1].dataset
    else:
        from train_revit import get_loaders

        dataset = get_loaders(args.dataset, data_root=args.data_root)[1].dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=benchmark_batches,
                      generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers,
                      pin_memory=device.type == 'cuda')


def action(x, index):
    if index >= 4:
        x = x.flip(-1)
    return x.rot90(index % 4, (-2, -1))


def orient(x, orientation, generator):
    if orientation == 'identity':
        return x
    if orientation == 'rotate':
        return action(x, 1)
    if orientation == 'reflect':
        return action(x, 4)
    order = 4 if orientation == 'uniform-c4' else 8
    indices = torch.randint(order, (len(x),), generator=generator).tolist()
    return torch.stack([action(image, index) for image, index in zip(x, indices)])


def binary_auroc(scores, targets):
    _, bins = scores.unique(sorted=True, return_inverse=True)
    positives = torch.bincount(bins[targets == 1], minlength=int(bins.max()) + 1).double()
    negatives = torch.bincount(bins[targets == 0], minlength=len(positives)).double()
    pairs = positives.sum() * negatives.sum()
    if pairs == 0:
        return None
    return (positives * (negatives.cumsum(0) - negatives / 2)).sum().item() / pairs.item()


@torch.inference_mode()
def evaluate(args):
    device = torch.device(args.device)
    model = load_budget_model(args.checkpoint, device).eval()
    if args.pointwise_backend == 'fourier':
        model.use_cyclic_fourier()
    thresholds = torch.tensor(model.budget_thresholds, device=device)
    costs = torch.tensor(model.profile_costs, dtype=torch.float64)
    budgets = model.budget_thresholds if args.budgets is None else args.budgets
    count = len(model.profiles)
    fixed_correct = torch.zeros(count, dtype=torch.long)
    total = rescued = regressions = 0
    orientation_rng = torch.Generator().manual_seed(args.seed)
    random_rng = torch.Generator().manual_seed(args.seed)
    records = [dict(budget=budget, correct=0, top5=0, shuffled_correct=0,
                    oracle_correct=0, oracle_cost=0.0, routes=torch.zeros(count, dtype=torch.long),
                    rescue_increments=[], transforms={}) for budget in budgets]
    pcam_targets = []
    pcam_scores = [[] for _ in budgets]
    order = {'C4': 4, 'D4': 8, 'Z2': 1}[model.config.group]
    for step, (images, targets) in enumerate(validation_loader(args, device)):
        if args.max_batches is not None and step >= args.max_batches:
            break
        images = orient(images.to(device), args.orientation, orientation_rng)
        targets = targets.to(device)
        total += len(images)
        rows = torch.arange(len(images), device=device)
        if args.dataset == 'pcam':
            pcam_targets.append(targets.cpu())
        with autocast(device, args.precision):
            early = model._stem(images)
            fixed = torch.stack([model._predict(early, i) for i in range(count)], dim=1)
            matches = fixed.argmax(dim=-1).eq(targets[:, None])
            fixed_correct += matches.sum(dim=0).cpu()
            rescued += ((~matches[:, 0]) & matches[:, -1]).sum().item()
            regressions += (matches[:, 0] & (~matches[:, -1])).sum().item()
            for index, (budget, record) in enumerate(zip(budgets, records)):
                logits, routes = model(images, budget=budget, return_routes=True)
                permutation = torch.randperm(len(images), generator=random_rng).to(device)
                feasible = matches & (thresholds <= budget)
                cheapest_correct = feasible.to(torch.int64).argmax(dim=1).cpu()
                record['correct'] += logits.argmax(dim=1).eq(targets).sum().item()
                record['top5'] += logits.topk(min(5, logits.shape[-1]), dim=1).indices.eq(targets[:, None]).sum().item()
                record['shuffled_correct'] += matches[rows, routes[permutation]].sum().item()
                record['oracle_correct'] += feasible.any(dim=1).sum().item()
                record['oracle_cost'] += costs[cheapest_correct].sum().item()
                rescued_rows = ((~matches[:, 0]) & feasible.any(dim=1)).cpu()
                record['rescue_increments'].append(costs[cheapest_correct[rescued_rows]] - costs[0])
                record['routes'] += torch.bincount(routes, minlength=count).cpu()
                if args.dataset == 'pcam':
                    pcam_scores[index].append((logits[:, 1] - logits[:, 0]).float().cpu())
                if args.check_group:
                    if model.router is None:
                        margin = torch.full_like(routes, torch.inf, dtype=torch.float32)
                    else:
                        scores, _ = model._routing(early, budget, sample_budgets=False)
                        best_two = scores.topk(2, dim=1).values
                        margin = best_two[:, 0] - best_two[:, 1]
                    groups = {
                        'below_1e-4': margin < 1e-4,
                        '1e-4_to_1e-2': (margin >= 1e-4) & (margin < 1e-2),
                        'above_1e-2': (margin >= 1e-2) & margin.isfinite(),
                        'single_eligible': margin.isinf(),
                    }
                    for g in range(1, order):
                        transformed, transformed_routes = model(action(images, g), budget=budget, return_routes=True)
                        current = record['transforms'].setdefault(g, dict(route_matches=0, max_logit_error=0.0,
                                                                          by_margin={}))
                        agreement = transformed_routes.eq(routes)
                        errors = (transformed.float() - logits.float()).abs().amax(dim=1)
                        current['route_matches'] += agreement.sum().item()
                        current['max_logit_error'] = max(current['max_logit_error'], errors.max().item())
                        for name, mask in groups.items():
                            samples = mask.sum().item()
                            if samples:
                                result = current['by_margin'].setdefault(name, dict(samples=0, route_matches=0,
                                                                                   max_logit_error=0.0))
                                result['samples'] += samples
                                result['route_matches'] += (agreement & mask).sum().item()
                                result['max_logit_error'] = max(result['max_logit_error'], errors[mask].max().item())
    if total == 0:
        raise ValueError('The evaluation data contains no samples')
    for index, record in enumerate(records):
        record['top1'] = record.pop('correct') / total
        record['top5'] /= total
        record['matched_shuffle_top1'] = record.pop('shuffled_correct') / total
        record['feasible_oracle_top1'] = record.pop('oracle_correct') / total
        record['mean_oracle_cost'] = record.pop('oracle_cost') / total
        spent = (record['routes'] * costs).sum().item()
        record['mean_profile_cost'] = spent / total
        rescue_increments = torch.cat(record.pop('rescue_increments')).sort().values
        extra = spent - total * costs[0].item()
        matched_rescues = torch.searchsorted(rescue_increments.cumsum(0), extra, right=True).item()
        record['cost_matched_oracle_top1'] = (fixed_correct[0].item() + matched_rescues) / total
        record['routes'] = record['routes'].tolist()
        for transform in record['transforms'].values():
            transform['route_agreement'] = transform.pop('route_matches') / total
            for group in transform['by_margin'].values():
                group['route_agreement'] = group.pop('route_matches') / group['samples']
        if args.dataset == 'pcam':
            record['auroc'] = binary_auroc(torch.cat(pcam_scores[index]), torch.cat(pcam_targets))
    return dict(dataset=args.dataset, examples=total, seed=args.seed, orientation=args.orientation,
                pointwise_backend=args.pointwise_backend,
                group=model.config.group, precision=args.precision, fixed_top1=(fixed_correct / total).tolist(),
                wide_rescues=rescued, wide_regressions=regressions, profile_costs=model.profile_costs,
                budget_thresholds=model.budget_thresholds, results=records)


def export(args):
    model = load_budget_model(args.checkpoint)
    if args.widths is None:
        fixed = model.export_profile(args.profile)
    else:
        fixed = model.prune_profile(args.profile, args.widths)
    torch.save({**model_config(fixed), 'model_state_dict': fixed.state_dict()}, args.destination)
    return {'destination': str(args.destination), 'widths': fixed.profiles[0], 'stem_width': fixed.stem_width}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    bench = commands.add_parser('benchmark')
    data = commands.add_parser('evaluate')
    save = commands.add_parser('export')
    for command in (bench, data):
        command.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        command.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
        command.add_argument('--pointwise-backend', choices=('dense', 'fourier'), default='dense',
                             help='Compare against the unfused C4 Fourier reference')
        command.add_argument('--budgets', type=float, nargs='+')
        command.add_argument('--threads', type=int)
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--dataset', choices=('cifar10', 'rotmnist', 'pcam', 'imagenet'), required=command is data)
        command.add_argument('--data-root', type=Path)
        command.add_argument('--pcam-split', choices=('valid', 'test'), default='valid')
        command.add_argument('--num-workers', type=int, default=4)
        command.add_argument('--orientation', choices=('identity', 'rotate', 'reflect', 'uniform-c4', 'uniform-d4'), default='identity')
    bench.add_argument('--checkpoint', type=Path)
    bench.add_argument('--smoke-backbone', choices=('global', 'windowed'), default='windowed')
    bench.add_argument('--group', choices=('C4', 'D4', 'Z2'), default='C4')
    bench.add_argument('--batch-sizes', type=int, nargs='+', default=(1, 16))
    bench.add_argument('--warmup', type=int, default=5)
    bench.add_argument('--iterations', type=int, default=30)
    bench.add_argument('--input-batches', type=int, default=8, help='Number of full dataset batches to rotate through')
    bench.add_argument('--trace', type=Path, help='Directory for named PyTorch Chrome traces')
    bench.add_argument('--compile-profiles', action='store_true', help='Compile post-stem paths with torch.compile')
    bench.add_argument('--prewarm-all-profiles', action='store_true', help='Include all cached kernels in resident memory')
    data.add_argument('checkpoint', type=Path)
    data.add_argument('--batch-size', type=int, default=32)
    data.add_argument('--check-group', action='store_true')
    data.add_argument('--max-batches', type=int)
    save.add_argument('checkpoint', type=Path)
    save.add_argument('profile', type=int)
    save.add_argument('destination', type=Path)
    save.add_argument('--widths', type=int, nargs='+', help='Prune the source profile to these field counts')
    args = parser.parse_args()
    if args.command != 'export':
        if args.pointwise_backend == 'fourier' and (args.precision != 'fp32' or args.device == 'mps'):
            parser.error('The Fourier reference expects FP32 on CPU or CUDA')
        if args.dataset == 'imagenet' and args.data_root is None:
            parser.error('--data-root is required for ImageNet')
        if args.budgets is not None and any(not 0 <= budget <= 1 for budget in args.budgets):
            parser.error('--budgets must lie in [0, 1]')
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
    if args.command == 'benchmark':
        if min(args.warmup, args.iterations, args.input_batches, *args.batch_sizes) < 1:
            parser.error('Use positive warmup, iteration and batch counts')
        if args.dataset and not args.checkpoint:
            parser.error('--checkpoint is required when benchmarking on a dataset')
        if args.trace and args.compile_profiles:
            parser.error('Use a separate eager run for named module traces')
        if args.compile_profiles and args.pointwise_backend == 'fourier':
            parser.error('Use the eager path for the complex-valued Fourier reference')
        if args.trace:
            args.trace.mkdir(parents=True, exist_ok=True)
        result = benchmark(args)
    elif args.command == 'evaluate':
        result = evaluate(args)
    else:
        result = export(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
