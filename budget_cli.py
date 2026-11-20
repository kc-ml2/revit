"""Shared CLI and checkpoint helpers for the adaptive classification models."""

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from budget_revit import BudgetAdaptiveREViT, BudgetREViTConfig, parse_profiles


def add_budget_arguments(parser):
    parser.add_argument('--budget-profiles', type=parse_profiles,
                        help='Semicolon-separated profiles, comma-separated widths within a profile')
    parser.add_argument('--budget', type=float, default=1.0, help='Normalized evaluation compute cap in [0, 1]')
    parser.add_argument('--profile-costs', type=float, nargs='+', help='Increasing costs, one per profile')
    parser.add_argument('--budget-stem-width', type=int, help='Shared lifting width in regular fields')
    parser.add_argument('--router-hidden', type=int, default=32)
    parser.add_argument('--cost-weight', type=float, default=0.2, help='Routing penalty per normalized cost unit')
    parser.add_argument('--window-sdpa', action='store_true', help='Use SDPA for the adaptive window backbone')


def budget_config(args, **backbone):
    return BudgetREViTConfig(
        profiles=args.budget_profiles, group=args.group_str, stem_width=args.budget_stem_width,
        router_hidden=args.router_hidden, cost_weight=args.cost_weight, window_sdpa=args.window_sdpa,
        profile_costs=None if args.profile_costs is None else tuple(args.profile_costs), **backbone,
    )


def model_config(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    if isinstance(model, BudgetAdaptiveREViT):
        return {'budget_config': asdict(model.config)}
    return {}


def load_budget_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'budget_config' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise ValueError('Expected an adaptive training checkpoint or exported profile')
    model = BudgetAdaptiveREViT(BudgetREViTConfig(**checkpoint['budget_config']))
    weights = {name.removeprefix('module.'): value for name, value in checkpoint['model_state_dict'].items()}
    model.load_state_dict(weights)
    return model.to(device)


def smooth_targets(targets, num_classes, smoothing):
    if smoothing == 0:
        return targets
    return F.one_hot(targets, num_classes).float() * (1 - smoothing) + smoothing / num_classes
