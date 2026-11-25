from dataclasses import replace

import pytest
import torch

from budget_cli import load_budget_model, load_supernet_model, model_config
from budget_revit import BudgetAdaptiveREViT, BudgetREViTConfig
from group_space import D4_SUBGROUPS
from scripts.run_budget import action
from symmetry_supernet import (
    SymmetrySupernet, SymmetrySupernetConfig, crop_d4_profile, default_symmetry_profiles,
    training_profile,
)


def config(window_size=4):
    backbone = BudgetREViTConfig(
        ((4, 8),) if window_size is not None else ((4,),),
        (1, 1) if window_size is not None else (1,),
        (2, 2) if window_size is not None else (2,),
        group='D4', image_size=13, window_size=window_size, stem_width=2,
    )
    full = backbone.profiles[0]
    narrow = tuple(width // 2 for width in full)
    profiles = (('D4', full),) + tuple((group, narrow) for group in D4_SUBGROUPS)
    return SymmetrySupernetConfig(backbone, profiles)


@pytest.mark.parametrize('window_size', [None, 4])
def test_all_subgroups_export_with_matching_logits_and_invariance(window_size, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = SymmetrySupernet(config(window_size))
    x = torch.randn(2, 3, 13, 13)
    # Make each profile's BatchNorm state distinct before exporting.
    with torch.no_grad():
        for profile in range(len(model.profiles)):
            model(x * (1 + profile), profile=profile)
    path = tmp_path / 'supernet.pt'
    torch.save({**model_config(model), 'model_state_dict': model.state_dict()}, path)
    model = load_supernet_model(path).eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x), model(x, profile=model.reference_profile))
        for i, (group, widths) in enumerate(model.profiles):
            fixed = model.export_profile(i)
            assert fixed.config.group == group
            assert fixed.config.profiles == (widths,)
            expected = model(x, profile=i)
            torch.testing.assert_close(fixed(x), expected, atol=3e-5, rtol=3e-5)
            for g in D4_SUBGROUPS[group][1]:
                transformed = action(x, g)
                torch.testing.assert_close(model(transformed, profile=i), expected, atol=8e-5, rtol=3e-5)
                torch.testing.assert_close(fixed(transformed), expected, atol=8e-5, rtol=3e-5)


def test_narrow_subgroup_updates_shared_coefficients_and_its_normalization():
    torch.manual_seed(7)
    model = SymmetrySupernet(config())
    x = torch.randn(2, 3, 12, 12)
    narrow = model.profiles.index(('C4', (2, 4)))
    ref = model.reference_profile
    projection = model.stages[0][0].qkv[0]
    assert projection.expand(ref).shape[:2] == (32, 32)
    assert projection.expand(narrow).shape[:2] == (8, 8)
    model(x, profile=narrow).square().mean().backward()
    assert projection.weights.grad[:2, :2].abs().sum() > 0
    assert projection.weights.grad[2:].count_nonzero() == 0
    assert projection.weights.grad[:, 2:].count_nonzero() == 0
    assert model.stem_norm.norms[narrow].num_batches_tracked == 1
    assert model.stem_norm.norms[ref].num_batches_tracked == 0


def test_cold_crop_of_d4_is_a_trainable_native_checkpoint(tmp_path):
    torch.manual_seed(17)
    source = BudgetAdaptiveREViT(config().backbone)
    fixed = crop_d4_profile(source, 'C4', (2, 4))
    expected_crop = source.entries[0].expand(0)[:16]
    # Pick the first four D4 channels of each retained field.
    inputs = [0, 1, 2, 3, 8, 9, 10, 11]
    torch.testing.assert_close(fixed.entries[0].expand(0), expected_crop[inputs][:, inputs], atol=2e-6, rtol=1e-5)
    path = tmp_path / 'fixed.pt'
    torch.save({**model_config(fixed), 'model_state_dict': fixed.state_dict()}, path)
    learned = load_budget_model(path)
    x = torch.randn(2, 3, 12, 12)
    optimizer = torch.optim.SGD(learned.parameters(), lr=0.01)
    before = learned.entries[0].weights.detach().clone()
    learned(x).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, learned.entries[0].weights)


def test_presets_schedule_and_configuration():
    profiles = default_symmetry_profiles((12, 24, 48, 96), (1, 2, 4, 8))
    assert profiles == tuple((group, (w, w*2, w*4, w*8)) for group in ('D4', 'C4') for w in (4, 8, 12))
    reference = profiles.index(('D4', (12, 24, 48, 96)))
    schedule = [training_profile(reference, len(profiles), i) for i in range(10)]
    assert schedule[::2] == [reference] * 5
    assert set(schedule) == set(range(len(profiles)))
    with pytest.raises(ValueError, match='full-width D4'):
        SymmetrySupernet(replace(config(), profiles=(('C4', (2, 4)),)))
    with pytest.raises(ValueError, match='head-aligned'):
        SymmetrySupernet(replace(config(), profiles=config().profiles + (('C2', (3, 4)),)))


def test_profile_specific_stem_trains_and_exports(tmp_path):
    cfg = config()
    profiles = (('D4', (4, 8)), ('C4', (1, 2, 4)), ('D4', (3, 4, 8)))
    model = SymmetrySupernet(replace(cfg, profiles=profiles))
    assert model.stem_widths == (2, 1, 3)
    assert model.reference_profile == 2
    assert model.stem.expand(1).shape[:2] == (4, 3)
    assert model.entries[0].expand(2).shape[:2] == (32, 24)
    x = torch.randn(2, 3, 13, 13)
    model(x, profile=1).square().mean().backward()
    assert model.stem.weights.grad[:1].count_nonzero() > 0
    assert model.stem.weights.grad[1:].count_nonzero() == 0
    path = tmp_path / 'variable-stems.pt'
    torch.save({**model_config(model), 'model_state_dict': model.state_dict()}, path)
    model = load_supernet_model(path).eval()
    with torch.inference_mode():
        for index, width in enumerate(model.stem_widths):
            fixed = model.export_profile(index)
            assert fixed.stem_width == width
            torch.testing.assert_close(fixed(x), model(x, profile=index), atol=3e-5, rtol=3e-5)
    cold = crop_d4_profile(model.export_profile(2), 'C4', (1, 2, 4))
    assert cold.stem_width == 1 and cold.profiles == ((2, 4),)
    with pytest.raises(ValueError, match='fit that profile'):
        SymmetrySupernet(replace(cfg, profiles=profiles + (('C4', (3, 2, 4)),)))
