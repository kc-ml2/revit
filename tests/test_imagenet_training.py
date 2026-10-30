from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from imagenet_utils import (
    DistributedEvalSampler, build_scheduler, evaluate, load_model_state,
    restore_training_state, save_training_checkpoint, train_one_epoch,
)
from model_config import MODEL_PRESETS
from scripts.profile_scaling import estimate_revit, estimate_vit


torch.set_num_threads(1)


@pytest.mark.parametrize("size,world", [(50000, 3), (2, 3), (0, 3), (17, 4)])
def test_eval_sampler_covers_each_example_once(size, world):
    data = range(size)
    samplers = [DistributedEvalSampler(data, world, rank) for rank in range(world)]
    shards = [list(s) for s in samplers]
    assert sorted(i for shard in shards for i in shard) == list(data)
    assert [len(s) for s in samplers] == [len(shard) for shard in shards]


def _distributed_worker(rank, world, rendezvous, output, count):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    torch.manual_seed(21)
    module = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 6))
    model = DDP(module, broadcast_buffers=False)
    # Deliberately different running stats: evaluation must use rank zero's.
    module[0].running_mean.fill_(rank * 10)
    module[0].running_var.fill_(rank + 1)
    x, y = torch.randn(count, 4), torch.arange(count)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2, sampler=DistributedEvalSampler(dataset))
    result = evaluate(model, loader, torch.device("cpu"), rank)
    module.eval()
    with torch.no_grad():
        logits = module(x)
        expected_loss = torch.nn.functional.cross_entropy(logits, y).item()
        pred = logits.topk(5).indices.eq(y[:, None])
        expected = (expected_loss, pred[:, :1].float().mean().item(), pred.any(dim=1).float().mean().item())
    torch.testing.assert_close(torch.tensor(result), torch.tensor(expected))
    opt = torch.optim.AdamW(module.parameters())
    sched = build_scheduler(opt, 3, 0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_training_checkpoint(Path(output) / "ddp.pt", model, opt, sched, 0, result[1],
                             torch.device("cpu"), scaler=scaler)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo not installed")
@pytest.mark.parametrize("count", [1, 5])
def test_uneven_distributed_evaluation_and_collective_checkpoint(tmp_path, count):
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    mp.spawn(_distributed_worker, args=(2, rendezvous, str(tmp_path), count), nprocs=2, join=True)
    ckpt = torch.load(tmp_path / "ddp.pt", weights_only=False)
    assert len(ckpt["rng_states"]) == 2
    assert not any(k.startswith("module.") for k in ckpt["model_state_dict"])


def test_train_loss_is_sample_weighted_and_checkpoint_resumes(tmp_path):
    torch.manual_seed(31)
    model = torch.nn.Linear(4, 6)
    data = TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 3, 4]))
    loader = DataLoader(data, batch_size=3)
    opt = torch.optim.AdamW(model.parameters(), lr=0)
    sched = build_scheduler(opt, 3, 0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(model(data.tensors[0]), data.tensors[1], label_smoothing=0.1).item()
    loss = train_one_epoch(model, loader, opt, scaler, torch.device("cpu"), 0, None, log_interval=0)
    assert loss == pytest.approx(expected, rel=1e-6)
    sched.step()
    path = tmp_path / "model.pt"
    save_training_checkpoint(path, model, opt, sched, 0, .5, torch.device("cpu"), scaler=scaler)
    expected_rng = torch.rand(4)
    ckpt = torch.load(path, weights_only=False)
    reloaded = torch.nn.Linear(4, 6)
    load_model_state(reloaded, ckpt["model_state_dict"])
    restored_opt = torch.optim.AdamW(reloaded.parameters())
    restored_sched = build_scheduler(restored_opt, 3, 0)
    restore_training_state(ckpt, restored_opt, restored_sched, scaler, None)
    torch.testing.assert_close(torch.rand(4), expected_rng)
    assert restored_sched.state_dict() == sched.state_dict()
    assert not Path(str(path) + ".tmp").exists()
    # Same next update, including Adam moments.
    for m, o in ((model, opt), (reloaded, restored_opt)):
        o.zero_grad(set_to_none=True)
        m(data.tensors[0]).square().mean().backward()
        o.step()
    for a, b in zip(model.parameters(), reloaded.parameters()):
        torch.testing.assert_close(a, b)


def test_validation_retains_dataparallel_wrapper():
    model = torch.nn.DataParallel(torch.nn.Linear(4, 6))
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    loader = DataLoader(TensorDataset(torch.randn(5, 4), torch.arange(5)), batch_size=3)
    evaluate(model, loader, torch.device("cpu"), None)
    handle.remove()
    assert len(calls) == 2


@pytest.mark.parametrize("warmup", [0, 2])
def test_scheduler_stays_within_total_epochs(warmup):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, 6, warmup)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(6):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[-1] == pytest.approx(0)
    assert max(lrs) == pytest.approx(1e-3)


@pytest.mark.parametrize("entrypoint", ["revit", "vit"])
def test_configured_image_size_reaches_both_loaders(tmp_path, entrypoint):
    for split in ("train", "val"):
        directory = tmp_path / split / "class0"
        directory.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (64, 64), color=(30 + i, 80, 90)).save(directory / f"{i}.png")
    if entrypoint == "revit":
        from imagenet_train_revit import get_imagenet_loaders
    else:
        from imagenet_train_vit import get_imagenet_loaders
    train, val, _ = get_imagenet_loaders(str(tmp_path), batch_size=2, num_workers=0, image_size=32)
    assert next(iter(train))[0].shape == next(iter(val))[0].shape == (2, 3, 32, 32)


def test_scaling_counts_match_executed_convolutions():
    from escnn.nn import R2Conv
    from group_space import get_gspace
    from revit_windowed_gcsa import Rot2DTransformerV2, Rot2DWindowAttention
    config = {"dims": (2, 4, 4, 8), "depths": (1, 1, 1, 1), "heads": (1, 2, 2, 4)}
    model = Rot2DTransformerV2(gspace=get_gspace("D4"), **config, qkv_kernel_size=3,
                              window_size=2, fast_init=True).eval()
    flops = []
    def hook(module, inputs, output):
        if isinstance(module, R2Conv):
            flops.append(2 * output.tensor.shape[-1]**2 * module.in_type.size * module.out_type.size * module.kernel_size**2)
        elif isinstance(module, torch.nn.Linear):
            flops.append(2 * module.in_features * module.out_features)
        else:
            side = inputs[0].tensor.shape[-1]
            padded = ((side + 1) // 2) * 2
            flops.append(4 * padded**2 * 4 * module.in_type.size)
    handles = []
    for module in model.modules():
        if isinstance(module, Rot2DWindowAttention):
            module.fused_qkv = False
        if isinstance(module, (R2Conv, torch.nn.Linear, Rot2DWindowAttention)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
    for handle in handles:
        handle.remove()
    assert estimate_revit(config, 8, image_size=64, window_size=2, kernel_size=3)["flops"] == sum(flops)


def test_vit_parameter_formula_matches_torchvision():
    from imagenet_train_vit import build_vit_small
    for image in (32, 224):
        model = build_vit_small(image_size=image)
        assert estimate_vit(image)["parameters"] == sum(p.numel() for p in model.parameters())


def test_resolution_does_not_change_revit_parameters_and_group_cost_is_quadratic():
    c4 = estimate_revit(MODEL_PRESETS["tiny"], 4)
    d4 = estimate_revit(MODEL_PRESETS["tiny"], 8)
    larger = estimate_revit(MODEL_PRESETS["tiny"], 4, image_size=448)
    assert larger["flops"] - larger["head_flops"] == 4 * (c4["flops"] - c4["head_flops"])
    assert d4["stages"][0]["projection_flops"] == 4 * c4["stages"][0]["projection_flops"]
    assert d4["stages"][0]["attention_flops"] == 2 * c4["stages"][0]["attention_flops"]
