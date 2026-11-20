from dataclasses import replace

import pytest
import torch
from escnn.nn import FieldType, GeometricTensor, R2Conv
from torch.nn import functional as F

from budget_cli import load_budget_model, model_config
from budget_revit import BudgetAdaptiveREViT, BudgetREViTConfig


torch.set_num_threads(2)


def config(group='C4', window=None, size=12):
    if window is None:
        return BudgetREViTConfig(((2,), (4,), (6,)), (1,), (2,), group=group, image_size=size)
    return BudgetREViTConfig(((2, 2), (4, 4), (6, 6)), (1, 1), (2, 2),
                             group=group, image_size=size, window_size=window)


@pytest.mark.parametrize('group', ['C4', 'D4'])
@pytest.mark.parametrize(('size', 'window', 'sdpa'), [(12, None, False), (12, 4, False), (13, 4, True)])
def test_every_group_element_preserves_fields_routes_and_logits(group, size, window, sdpa):
    torch.manual_seed(4)
    cfg = replace(config(group, window, size), window_sdpa=sdpa)
    model = BudgetAdaptiveREViT(cfg).eval()
    # Force a mixture of narrow and wide allocations using two budget caps.
    model.router[-1].bias.data.copy_(torch.tensor([-4., -2., 4.]))
    x = torch.randn(2, 3, size, size)
    budgets = torch.tensor([0., 1.])
    output_type = FieldType(model.gspace, [model.gspace.regular_repr] * cfg.profiles[-1][-1])
    with torch.inference_mode():
        logits, routes = model(x, budget=budgets, return_routes=True)
        assert routes.tolist() == [0, 2]
        features = GeometricTensor(model._features(model._stem(x), 2), output_type)
        for g in model.gspace.fibergroup.elements:
            transformed = GeometricTensor(x, model.input_type).transform(g).tensor
            values, new_routes = model(transformed, budget=budgets, return_routes=True)
            torch.testing.assert_close(new_routes, routes)
            torch.testing.assert_close(values, logits, atol=3e-5, rtol=2e-5)
            found = model._features(model._stem(transformed), 2)
            torch.testing.assert_close(found, features.transform(g).tensor, atol=5e-5, rtol=3e-5)


def test_basis_and_profile_filters_are_nested_and_actually_smaller():
    torch.manual_seed(12)
    model = BudgetAdaptiveREViT(config('D4'))
    projection = model.stages[0][0].qkv[0]
    regular = model.gspace.regular_repr
    field_type = FieldType(model.gspace, [regular] * 2)
    original = R2Conv(field_type, field_type, 3, padding=1, bias=False, initialize=False)
    original.weights.data.copy_(projection.weights[:2, :2].reshape(-1))
    narrow = projection.expand(0)
    wide = projection.expand(2)
    assert narrow.shape == (16, 16, 3, 3)
    assert wide.shape == (48, 48, 3, 3)
    torch.testing.assert_close(narrow, wide[:16, :16])
    torch.testing.assert_close(narrow, original.expand_parameters()[0])


def test_route_bucketing_and_budget_caps():
    model = BudgetAdaptiveREViT(config()).eval()
    model.router[-1].weight.data.zero_()
    model.router[-1].bias.data.copy_(torch.tensor([-2., 0., 2.]))
    x = torch.randn(4, 3, 12, 12)
    budgets = torch.tensor([1., 0., model.budget_thresholds[1], 0.])
    with torch.inference_mode():
        out, routes = model(x, budget=budgets, return_routes=True)
        assert routes.tolist() == [2, 0, 1, 0]
        for i, profile in enumerate(routes.tolist()):
            torch.testing.assert_close(out[i:i + 1], model(x[i:i + 1], profile=profile))
    fresh = BudgetAdaptiveREViT(config()).eval()
    with torch.inference_mode():
        fresh(x, budget=0.)
    filters = fresh.stages[0][0].qkv[0]._filters
    assert list(filters) == [0]
    assert filters[0].shape[0] == filters[0].shape[1] == 8


def test_controller_can_assign_different_images_different_widths_at_one_budget():
    torch.manual_seed(3)
    model = BudgetAdaptiveREViT(config()).eval()
    x = torch.stack([torch.zeros(3, 12, 12), torch.randn(3, 12, 12)])
    with torch.inference_mode():
        means = model._fields(model._stem(x)).mean(dim=(2, 3, 4))
        field = means[1].argmax().item()
        model.router[0].weight.zero_()
        model.router[0].bias.zero_()
        model.router[0].weight[0, field] = 1
        model.router[0].bias[0] = -means[1, field] / 2
        model.router[-1].weight.zero_()
        model.router[-1].bias.copy_(torch.tensor([0., -5., 0.]))
        model.router[-1].weight[2, 0] = 1
        _, routes = model(x, budget=1.0, return_routes=True)
        assert routes.tolist() == [0, 2]


def test_training_and_static_export_round_trip(tmp_path):
    torch.manual_seed(9)
    model = BudgetAdaptiveREViT(config())
    x = torch.randn(3, 3, 12, 12)
    targets = F.one_hot(torch.tensor([0, 1, 2]), 10).float()
    before = model.stages[0][0].norm1.norms[0].running_mean.clone()
    _, loss = model(x, targets=targets, budget=1.)
    loss.mean().backward()
    assert torch.isfinite(loss).all()
    assert model.router[-1].weight.grad.abs().sum() > 0
    assert model.stages[0][0].qkv[0].weights.grad[-1, -1].abs().sum() > 0
    assert all(norm.num_batches_tracked == 1 for norm in model.stages[0][0].norm1.norms)
    assert not torch.equal(before, model.stages[0][0].norm1.norms[0].running_mean)

    model.eval()
    with torch.inference_mode():
        expected = model(x, profile=1)
    fixed = model.export_profile(1)
    assert fixed.router is None and fixed.config.stem_width == 2
    assert fixed.stages[0][0].qkv[0].weights.shape[:2] == (4, 4)
    assert sum(p.numel() for p in fixed.parameters()) < sum(p.numel() for p in model.parameters())
    path = tmp_path / 'static.pt'
    torch.save({**model_config(fixed), 'model_state_dict': fixed.state_dict()}, path)
    restored = load_budget_model(path).eval()
    with torch.inference_mode():
        torch.testing.assert_close(expected, restored(x))
        torch.testing.assert_close(expected, fixed(x))
    assert all('filter' not in key for key in model.state_dict())


def test_validation_and_trivial_control():
    with pytest.raises(ValueError, match='head count'):
        BudgetAdaptiveREViT(replace(config(), profiles=((1,), (2,))))
    with pytest.raises(ValueError, match='never shrink'):
        BudgetAdaptiveREViT(replace(config(), profiles=((4,), (2,))))
    with pytest.raises(ValueError, match='strictly increasing'):
        BudgetAdaptiveREViT(replace(config(), profile_costs=(1., 2., 2.)))
    model = BudgetAdaptiveREViT(replace(config('Z2'), stem_downsample=4))
    assert model.order == 1
    with pytest.raises(ValueError, match='square'):
        model(torch.randn(1, 3, 4, 5))
    with pytest.raises(ValueError, match='values'):
        model(torch.randn(1, 3, 4, 4), budget=torch.tensor(float('nan')))
    _, loss = model(torch.randn(1, 3, 4, 4), targets=torch.tensor([0]))
    assert torch.isfinite(loss).all()


def test_cyclic_fourier_matches_dense_profiles():
    torch.manual_seed(2)
    model = BudgetAdaptiveREViT(replace(config('C4', 4), qkv_kernel=1)).eval()
    x = torch.randn(2, 3, 12, 12)
    with torch.inference_mode():
        dense = [model(x, profile=i) for i in range(len(model.profiles))]
        model.use_cyclic_fourier()
        for i, expected in enumerate(dense):
            torch.testing.assert_close(model(x, profile=i), expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match='only for C4'):
        BudgetAdaptiveREViT(config('D4')).use_cyclic_fourier()


def test_pruning_independent_wide_model_keeps_complete_prefixes():
    torch.manual_seed(19)
    cfg = replace(config('D4', 4), profiles=((6, 8),), stem_width=2)
    wide = BudgetAdaptiveREViT(cfg).eval()
    narrow = wide.prune_profile(0, (2, 4))
    assert narrow.stem_width == 2
    assert narrow.profiles == ((2, 4),)
    torch.testing.assert_close(narrow.entries[1].weights, wide.entries[1].weights[:4, :2])
    x = torch.randn(2, 3, 12, 12)
    with torch.inference_mode():
        torch.testing.assert_close(narrow(x), narrow(x.rot90(1, (-2, -1)).flip(-1)), atol=3e-5, rtol=2e-5)
