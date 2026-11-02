import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from escnn import gspaces
from escnn.nn import FieldType, GeometricTensor, R2Conv

from group_space import get_gspace
from group_complexity import regular_conv_cost
from imagenet_utils import load_model_state
from regular_group_ops import (
    EquivariantSpatialDepthwise, FourierRegularConv, FourierRightGroupLinear,
    LinearRegularConv, SharedFieldLinear, SparseRightGroupMix, regular_layout,
)
from revit_windowed_gcsa import Rot2DTransformerV2, Rot2DWindowAttention, Rot2DBlockV2
from scripts.profile_scaling import estimate_revit


torch.set_num_threads(1)


def direct_right_convolution(module, x):
    """Independent O(g^2) reference, used only for correctness tests."""
    group = module.in_type.gspace.fibergroup
    _, _, elements, to_canonical, from_canonical = regular_layout(group)
    index = {a: int(to_canonical[j]) for j, a in enumerate(elements)}
    b, _, h, w = x.shape
    x = x.reshape(b, module.in_fields, module.order, h, w)
    weight = module.weight.reshape(module.out_fields, module.in_fields, module.order)
    outputs = []
    for a in elements:
        terms = [torch.einsum("oi,bihw->bohw", weight[:, :, j], x[:, :, index[a @ t]])
                 for j, t in enumerate(elements)]
        outputs.append(torch.stack(terms).sum(dim=0))
    return torch.stack(outputs, dim=2).index_select(2, from_canonical).reshape(b, module.out_fields * module.order, h, w)


@pytest.mark.parametrize("dihedral,n", [(False, 1), (False, 2), (False, 3), (False, 4),
                                        (False, 8), (True, 3), (True, 4), (True, 8)])
def test_fourier_matches_direct_group_convolution_and_gradients(dihedral, n):
    gs = gspaces.flipRot2dOnR2(N=n) if dihedral else gspaces.rot2dOnR2(N=n)
    it, ot = FieldType(gs, 2 * [gs.regular_repr]), FieldType(gs, 3 * [gs.regular_repr])
    m = FourierRightGroupLinear(it, ot).double()
    raw = torch.randn(2, it.size, 3, 3, dtype=torch.float64, requires_grad=True)
    expected = direct_right_convolution(m, raw)
    probe = torch.randn_like(expected)
    gradients = torch.autograd.grad((expected * probe).sum(), (raw, m.weight))
    actual = m(GeometricTensor(raw, it)).tensor
    actual_gradients = torch.autograd.grad((actual * probe).sum(), (raw, m.weight))
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    for a, b in zip(actual_gradients, gradients):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", ["Z2", "D2", "C4", "C8", "C12", "C16", "D4", "D8", "D12", "D16"])
@pytest.mark.parametrize("kind", ["sparse", "fourier"])
def test_group_mix_commutes_with_all_fiber_actions(name, kind):
    gs = get_gspace(name)
    ft = FieldType(gs, 2 * [gs.regular_repr])
    m = SparseRightGroupMix(ft) if kind == "sparse" else FourierRightGroupLinear(ft, ft)
    with torch.no_grad():
        m.weight.normal_(std=.2)
        raw = GeometricTensor(torch.randn(1, ft.size, 3, 3), ft)
        y = m(raw)
        for a in gs.fibergroup.elements:
            torch.testing.assert_close(m(raw.transform_fibers(a)).tensor,
                                       y.transform_fibers(a).tensor, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("name", ["C4", "D4"])
@pytest.mark.parametrize("kind", ["linear", "fourier"])
@pytest.mark.parametrize("kernel", [1, 3])
def test_structured_convolutions_are_spatially_equivariant(name, kind, kernel):
    gs = get_gspace(name)
    it, ot = FieldType(gs, 2 * [gs.regular_repr]), FieldType(gs, 3 * [gs.regular_repr])
    cls = LinearRegularConv if kind == "linear" else FourierRegularConv
    m = cls(it, ot, kernel).eval()
    with torch.no_grad():
        m.group_mix.weight.normal_(std=.2)
        raw = GeometricTensor(torch.randn(2, it.size, 9, 9), it)
        output = m(raw)
        for a in gs.fibergroup.elements:
            torch.testing.assert_close(m(raw.transform(a)).tensor, output.transform(a).tensor,
                                       rtol=5e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["linear", "fourier"])
@pytest.mark.parametrize("name", ["C4", "D4"])
def test_structured_window_attention_equivariance_and_sdpa(mode, name):
    gs = get_gspace(name)
    ft = FieldType(gs, 4 * [gs.regular_repr])
    model = Rot2DWindowAttention(ft, 2, window_size=7, qkv_kernel_size=3, conv_mode=mode).eval()
    raw = GeometricTensor(torch.randn(2, ft.size, 14, 14), ft)
    with torch.no_grad():
        expected = model(raw)
        for a in gs.fibergroup.elements:
            torch.testing.assert_close(model(raw.transform(a)).tensor,
                                       expected.transform(a).tensor, rtol=2e-4, atol=2e-5)
        model.use_sdpa = True
        torch.testing.assert_close(model(raw).tensor, expected.tensor, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("mode", ["quadratic", "linear", "fourier"])
def test_full_model_backward_checkpoint_and_roundtrip(mode):
    m = Rot2DTransformerV2(gspace=get_gspace("D4"), dims=(2, 4, 4, 8), depths=(1, 1, 1, 1),
                           heads=(1, 2, 2, 4), num_classes=6, window_size=2,
                           qkv_kernel_size=3, model_config=mode, use_checkpoint=True)
    x = torch.randn(2, 3, 64, 64)
    m(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with torch.no_grad():
        m.eval()
        expected = m(x)
        state = {k: v.clone() for k, v in m.state_dict().items()}
        load_model_state(m, state)
        m.eval()
        torch.testing.assert_close(m(x), expected)
    if mode != "quadratic":
        # Only the RGB lifting layer may be a regular dense R2Conv. Depthwise
        # kernel generators have trivial inputs, so cannot hide g^2 mixing.
        for layer in m.modules():
            if isinstance(layer, R2Conv):
                assert all(r == layer.in_type.gspace.trivial_repr for r in layer.in_type.representations)


@pytest.mark.parametrize("mode", ["linear", "fourier"])
def test_checkpointing_does_not_change_structured_block_statistics(mode):
    gs = get_gspace("D4")
    ft = FieldType(gs, 2 * [gs.regular_repr])
    plain = Rot2DBlockV2(ft, 1, 2, 2, .1, .1, .1, 3, False, conv_mode=mode)
    checked = copy.deepcopy(plain)
    checked.use_checkpoint = True
    raw = torch.randn(2, ft.size, 4, 4)
    results = []
    for model in (plain, checked):
        torch.manual_seed(33)
        x = raw.clone().requires_grad_()
        out = model(GeometricTensor(x, ft)).tensor
        out.square().mean().backward()
        results.append((out, x.grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b)
    for a, b in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    for name, value in plain.named_buffers():
        torch.testing.assert_close(value, dict(checked.named_buffers())[name], msg=name)


@pytest.mark.parametrize("mode", ["quadratic", "linear", "fourier"])
def test_analytic_complexity_matches_executed_operator_shapes(mode):
    config = {"dims": (2, 4, 4, 8), "depths": (1, 1, 1, 1), "heads": (1, 2, 2, 4)}
    gs = get_gspace("D4")
    model = Rot2DTransformerV2(gspace=gs, **config, num_classes=6, window_size=2,
                              qkv_kernel_size=3, model_config=mode, fast_init=True).eval()
    costs, fft = [], []
    def hook(layer, inputs, output):
        if isinstance(layer, (LinearRegularConv, FourierRegularConv)):
            area = output.tensor.shape[-2] * output.tensor.shape[-1]
            row = regular_conv_cost(len(layer.in_type.representations), len(layer.out_type.representations),
                                    8, area, layer.kernel_size, layer.stride, mode, "dihedral")
            costs.append(row["flops"])
            fft.append(row["fft_flops_estimate"])
        elif isinstance(layer, R2Conv):
            costs.append(2 * output.tensor.shape[-2] * output.tensor.shape[-1] *
                         layer.in_type.size * layer.out_type.size * layer.kernel_size**2)
        elif isinstance(layer, torch.nn.Linear):
            costs.append(2 * layer.in_features * layer.out_features)
        elif isinstance(layer, Rot2DWindowAttention):
            side = inputs[0].tensor.shape[-1]
            padded = ((side + 1) // 2) * 2
            costs.append(4 * padded**2 * 4 * layer.in_type.size)
    handles = []
    for layer in model.modules():
        if isinstance(layer, Rot2DWindowAttention):
            layer.fused_qkv = False
        if isinstance(layer, (LinearRegularConv, FourierRegularConv, R2Conv, torch.nn.Linear, Rot2DWindowAttention)):
            handles.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
    for handle in handles:
        handle.remove()
    expected = estimate_revit(config, 8, image_size=64, window_size=2, kernel_size=3,
                              num_classes=6, model_config=mode, group_kind="dihedral")
    assert sum(costs) == pytest.approx(expected["flops"])
    assert sum(fft) == pytest.approx(expected["fft_flops_estimate"])


def test_linear_complexity_doubles_with_group_order_and_fourier_is_not_omitted():
    config = {"dims": (4, 8, 16, 32), "depths": (1, 1, 1, 1), "heads": (1, 2, 4, 8)}
    for kind in ("cyclic", "dihedral"):
        sparse = [estimate_revit(config, g, model_config="linear", group_kind=kind) for g in (8, 16, 32)]
        for a, b in zip(sparse, sparse[1:]):
            assert b["flops"] - b["head_flops"] == 2 * (a["flops"] - a["head_flops"])
        dense = [regular_conv_cost(8, 8, g, 49) for g in (8, 16, 32)]
        assert dense[1]["flops"] == 4 * dense[0]["flops"]
        spectral = [estimate_revit(config, g, model_config="fourier", group_kind=kind) for g in (8, 16, 32)]
        for a, b in zip(spectral, spectral[1:]):
            assert a["fft_flops_estimate"] > 0
            assert a["flops"] == a["counted_flops"] + a["fft_flops_estimate"]
            assert b["flops"] / a["flops"] < 3


def test_old_quadratic_default_and_explicit_config_have_identical_state():
    kwargs = dict(gspace=get_gspace("C4"), dims=(2, 2, 2, 2), depths=(1, 1, 1, 1),
                  heads=(1, 1, 1, 1), num_classes=6, window_size=2, fast_init=True)
    torch.manual_seed(41)
    default = Rot2DTransformerV2(**kwargs)
    torch.manual_seed(41)
    explicit = Rot2DTransformerV2(**kwargs, model_config="quadratic")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)


def test_cross_config_checkpoint_has_a_clear_error(tmp_path):
    from imagenet_train_revit import load_pretrained_weights
    path = tmp_path / "quadratic.pt"
    torch.save({"model_state_dict": {}, "args": {"model_config": "quadratic"}}, path)
    model = Rot2DTransformerV2(gspace=get_gspace("C4"), dims=(2, 2, 2, 2), depths=(1, 1, 1, 1),
                              heads=(1, 1, 1, 1), window_size=2, model_config="linear", fast_init=True)
    with pytest.raises(ValueError, match="not have interchangeable weights"):
        load_pretrained_weights(model, str(path), torch.device("cpu"))


@pytest.mark.parametrize("mode", ["quadratic", "linear", "fourier"])
def test_training_cli_selects_counts_checkpoints_and_resumes_each_family(tmp_path, mode):
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    for split in ("train", "val"):
        for label in range(2):
            directory = data / split / str(label)
            directory.mkdir(parents=True)
            for j in range(2):
                Image.new("RGB", (72, 72), color=(30 + label * 40, 60 + j * 20, 100)).save(directory / f"{j}.png")
    output = tmp_path / "output"
    command = [sys.executable, str(root / "imagenet_train_revit.py"),
               "--data-root", str(data), "--output-dir", str(output),
               "--model-config", mode, "--group-str", "D4", "--num-classes", "2",
               "--dims", "2", "4", "4", "8", "--heads", "1", "2", "2", "4",
               "--depths", "1", "1", "1", "1", "--image-size", "64", "--window-size", "2",
               "--qkv-kernel-size", "3", "--epochs", "2", "--warmup-epochs", "0",
               "--batch-size", "2", "--num-workers", "0", "--use-checkpoint",
               "--no-randaugment", "--checkpoint-interval", "1", "--log-interval", "0"]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90, check=True)
    assert f"model_config={mode}" in result.stdout
    assert "Estimated FLOPs" in result.stdout
    assert ("estimated FFT work" in result.stdout) == (mode == "fourier")
    checkpoint = torch.load(output / "checkpoints" / "last.pt", weights_only=False)
    assert checkpoint["args"]["model_config"] == mode
    assert checkpoint["epoch"] == 1
    resumed = subprocess.run(command + ["--pretrained", "--pretrained-path", str(output / "checkpoints" / "last.pt")],
                             cwd=root, text=True, capture_output=True, timeout=90, check=True)
    assert "Resuming training from epoch 2" in resumed.stdout
    assert "Epoch 002" not in resumed.stdout


@pytest.mark.parametrize("name", ["C12", "D12"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_non_power_of_two_fourier_accepts_low_precision_features(name, dtype):
    gs = get_gspace(name)
    ft = FieldType(gs, 2 * [gs.regular_repr])
    model = FourierRightGroupLinear(ft, ft)
    raw = torch.randn(2, ft.size, 3, 3, dtype=dtype, requires_grad=True)
    out = model(GeometricTensor(raw, ft)).tensor
    assert out.dtype == dtype
    out.float().square().mean().backward()
    assert torch.isfinite(raw.grad).all() and torch.isfinite(model.weight.grad).all()


@pytest.mark.parametrize("mode", ["linear", "fourier"])
@pytest.mark.parametrize("kernel", [1, 3])
def test_structured_operator_autocast_backward_on_cpu(mode, kernel):
    gs = get_gspace("D4")
    ft = FieldType(gs, 2 * [gs.regular_repr])
    cls = LinearRegularConv if mode == "linear" else FourierRegularConv
    model = cls(ft, ft, kernel)
    raw = torch.randn(2, ft.size, 5, 5, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(GeometricTensor(raw, ft)).tensor
    if mode == "linear" or kernel == 3:
        assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
