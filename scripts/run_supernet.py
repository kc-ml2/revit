"""Inspect symmetry supernet checkpoints, export profiles, or cold-crop a D4 model."""

import argparse
import json
from pathlib import Path

import torch

from budget_cli import load_budget_model, load_supernet_model, model_config
from group_space import SQUARE_GROUPS
from symmetry_supernet import crop_d4_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    show = commands.add_parser('list', help='List profiles and the number of training batches seen')
    export = commands.add_parser('export', help='Export a trained profile as a trainable standalone checkpoint')
    crop = commands.add_parser('crop-d4', help='Build the cold-crop baseline from a fixed-group D4 checkpoint')
    for command in (show, export, crop):
        command.add_argument('checkpoint', type=Path)
        command.add_argument('--threads', type=int)
    export.add_argument('profile', type=int)
    export.add_argument('destination', type=Path)
    export.add_argument('--verify', action='store_true', help='Check logits against the selected supernet path')
    crop.add_argument('group', choices=SQUARE_GROUPS)
    crop.add_argument('destination', type=Path)
    crop.add_argument('--widths', type=int, nargs='+', required=True,
                      help='Stage widths with an optional leading stem width')
    crop.add_argument('--from-profile', type=int, help='Source width index (defaults to the widest)')
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.command == 'crop-d4':
        model = load_budget_model(args.checkpoint)
        fixed = crop_d4_profile(model, args.group, args.widths, args.from_profile)
    else:
        model = load_supernet_model(args.checkpoint)
        if args.command == 'list':
            profiles = [dict(index=i, group=group, widths=model.stage_widths[i], stem_width=model.stem_widths[i],
                             training_batches=int(model.stem_norm.norms[i].num_batches_tracked))
                        for i, (group, _) in enumerate(model.profiles)]
            print(json.dumps(dict(reference_profile=model.reference_profile, profiles=profiles), indent=2))
            return
        fixed = model.export_profile(args.profile)

    result = dict(destination=str(args.destination), group=fixed.config.group,
                  widths=fixed.profiles[0], stem_width=fixed.stem_width)
    if args.command == 'export' and args.verify:
        cfg = model.config
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(1, cfg.in_channels, cfg.image_size, cfg.image_size, generator=generator)
        model.eval()
        fixed.eval()
        with torch.inference_mode():
            expected, actual = model(x, profile=args.profile), fixed(x)
            torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        result['max_logit_error'] = (actual - expected).abs().max().item()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**model_config(fixed), 'model_state_dict': fixed.state_dict()}, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
