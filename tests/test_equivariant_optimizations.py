import copy

import pytest
import torch
from escnn.nn import FieldType, GeometricTensor

from equivariant_ops import project_qkv
from group_space import get_gspace
from imagenet_utils import load_model_state
from revit_gcsa import Rot2DMultiHeadAttention, Rot2DConvMultiHeadAttention
from revit_windowed_gcsa import Rot2DWindowAttention, Rot2DBlockV2, Rot2DTransformerV2


torch.set_num_threads(1)


@pytest.mark.parametrize("group", ["C4", "D4"])
@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("training", [True, False])
def test_fused_projection_outputs_and_gradients(group, kernel, training):
    g = get_gspace(group)
    ft = FieldType(g, 4 * [g.regular_repr])
    attention = copy.deepcopy(Rot2DWindowAttention(ft, 2, qkv_kernel_size=kernel)).double()
    attention.train(training)
    projections = (attention.to_q, attention.to_k, attention.to_v)
    raw = torch.randn(2, ft.size, 7, 7, dtype=torch.float64, requires_grad=True)
    args = (raw,) + tuple(m.weights for m in projections)
    gx = GeometricTensor(raw, ft)
    expected = project_qkv(gx, *projections, fused=False)
    # A nonconstant cotangent checks all output elements, not just their sum.
    cotangents = [torch.randn_like(t) for t in expected]
    grads = torch.autograd.grad(sum((t * v).sum() for t, v in zip(expected, cotangents)), args, retain_graph=True)
    actual = project_qkv(gx, *projections, fused=True)
    actual_grads = torch.autograd.grad(sum((t * v).sum() for t, v in zip(actual, cotangents)), args)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    for a, b in zip(actual_grads, grads):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("group", ["C4", "D4"])
@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("sdpa", [False, True])
def test_attention_equivariance_on_aligned_square_windows(group, kernel, sdpa):
    g = get_gspace(group)
    ft = FieldType(g, 4 * [g.regular_repr])
    module = Rot2DWindowAttention(ft, 2, window_size=7, qkv_kernel_size=kernel).eval()
    module.use_sdpa = sdpa
    x = GeometricTensor(torch.randn(2, ft.size, 14, 14), ft)
    with torch.no_grad():
        y = module(x)
        for element in g.fibergroup.elements:
            torch.testing.assert_close(module(x.transform(element)).tensor,
                                       y.transform(element).tensor, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("kind", ["window", "global", "global_conv"])
def test_attention_backends_match_with_backward(kind):
    g = get_gspace("D4")
    ft = FieldType(g, 4 * [g.regular_repr])
    if kind == "window":
        module = Rot2DWindowAttention(ft, 2, window_size=7, qkv_kernel_size=3)
    elif kind == "global":
        module = Rot2DMultiHeadAttention(g, ft, 2, dropout=0, attn_dropout=0)
    else:
        module = Rot2DConvMultiHeadAttention(g, ft, 2, kernel_size=3, dropout=0, attn_dropout=0)
    module.train()
    raw = torch.randn(2, ft.size, 14, 14, requires_grad=True)
    module.fused_qkv, module.use_sdpa = False, False
    expected = module(GeometricTensor(raw, ft)).tensor
    probe = torch.randn_like(expected)
    args = (raw,) + tuple(module.parameters())
    grads = torch.autograd.grad((expected * probe).sum(), args)
    module.fused_qkv, module.use_sdpa = True, True
    actual = module(GeometricTensor(raw, ft)).tensor
    actual_grads = torch.autograd.grad((actual * probe).sum(), args)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    for a, b in zip(actual_grads, grads):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-4)


def test_checkpoint_preserves_gradients_rng_and_batchnorm_state():
    g = get_gspace("C4")
    ft = FieldType(g, 4 * [g.regular_repr])
    plain = Rot2DBlockV2(ft, 2, 7, 2, 0.15, 0.1, 0.2, 3, False)
    checked = copy.deepcopy(plain)
    checked.use_checkpoint = True
    raw = torch.randn(2, ft.size, 14, 14)
    outputs, input_grads = [], []
    for model in (plain, checked):
        torch.manual_seed(1234)
        x = raw.clone().requires_grad_()
        out = model(GeometricTensor(x, ft)).tensor
        out.square().mean().backward()
        outputs.append(out)
        input_grads.append(x.grad)
    torch.testing.assert_close(*outputs)
    torch.testing.assert_close(*input_grads)
    for p, q in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    plain_buffers, checked_buffers = dict(plain.named_buffers()), dict(checked.named_buffers())
    assert plain_buffers.keys() == checked_buffers.keys()
    for name in plain_buffers:
        torch.testing.assert_close(plain_buffers[name], checked_buffers[name], msg=name)


def test_full_model_eval_checkpoint_compatibility_and_fused_logits():
    model = Rot2DTransformerV2(gspace=get_gspace("D4"), dims=(2, 4, 4, 8),
                              depths=(1, 1, 1, 1), heads=(1, 2, 2, 4),
                              num_classes=6, window_size=2, fast_init=True).eval()
    raw = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = model(raw)
    # Old trainer saved in eval mode, with module. and derived filter buffers.
    old_state = {"module." + k: v.clone() for k, v in model.state_dict().items()}
    load_model_state(model, old_state)
    model.eval()
    for module in model.modules():
        if hasattr(module, "fused_qkv"):
            module.fused_qkv = False
    with torch.no_grad():
        torch.testing.assert_close(model(raw), expected, rtol=2e-5, atol=2e-6)


def test_single_initialization_default_is_finite():
    model = Rot2DTransformerV2(gspace=get_gspace("C4"), dims=(2, 2, 2, 2),
                              depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window_size=2)
    assert not any(name.endswith(".filter") for name, _ in model.named_buffers())
    out = model(torch.randn(2, 3, 32, 32))
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
