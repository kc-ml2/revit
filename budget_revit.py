"""Nested, budget-adaptive REViT on square image grids."""

import copy
import math
from dataclasses import dataclass, replace

import torch
from escnn.nn import FieldType, R2Conv
from torch import nn
from torch.nn import functional as F

from group_space import get_gspace
from revit_gcsa import _sdpa_spatial


def centered_downsample(x, factor):
    """Pool to ceil(size / factor) using bins paired by grid reflection."""
    if factor == 1:
        return x
    size = x.shape[-1]
    if size % factor == 0:
        return F.avg_pool2d(x, factor)
    return F.adaptive_avg_pool2d(x, math.ceil(size / factor))


def window_layout(size, requested):
    """Return the window side and equal padding at both ends of each axis."""
    side = min(size, requested)
    if size % 2 and side % 2 == 0:
        side -= 1
    count = math.ceil(size / side)
    if (count * side - size) % 2:
        count += 1
    return side, (count * side - size) // 2


def window_attention(q, k, v, heads, requested, use_sdpa):
    batch, channels, size, _ = q.shape
    side, pad = window_layout(size, requested)
    count = (size + 2 * pad) // side

    def partition(tensor):
        if pad:
            tensor = F.pad(tensor, (pad,) * 4)
        tensor = tensor.reshape(batch, channels, count, side, count, side)
        return tensor.permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, side, side)

    q, k, v = (partition(tensor) for tensor in (q, k, v))
    if use_sdpa:
        out = _sdpa_spatial(q, k, v, heads, 0.0)
    else:
        head_dim = channels // heads
        q, k, v = (tensor.reshape(-1, heads, head_dim, side**2) for tensor in (q, k, v))
        scores = (q.transpose(-2, -1) @ k) * head_dim**-0.5
        out = v @ scores.softmax(dim=-1).transpose(-2, -1)
    out = out.reshape(batch, count, count, channels, side, side)
    out = out.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, count * side, count * side)
    return out[..., pad:pad + size, pad:pad + size].contiguous()


def sampled_basis(gspace, in_repr, out_repr, kernel_size):
    """Use ESCNN's public expansion on one input/output field pair."""
    conv = R2Conv(
        FieldType(gspace, [in_repr]), FieldType(gspace, [out_repr]),
        kernel_size, bias=False, initialize=False,
    )
    expansion = conv.basisexpansion
    basis = torch.vmap(expansion)(torch.eye(expansion.dimension(), dtype=torch.float32))
    return basis.flatten(1).contiguous().to(torch.get_default_dtype())


class ProfileConv(nn.Module):
    """Share coefficients on field pairs and expand only the active filter."""

    def __init__(self, basis, in_fields, out_fields, in_rep_size, out_rep_size, kernel):
        super().__init__()
        self.in_fields = tuple(in_fields)
        self.out_fields = tuple(out_fields)
        self.in_rep_size = in_rep_size
        self.out_rep_size = out_rep_size
        self.kernel = kernel
        self.register_buffer('basis', basis, persistent=False)
        self.weights = nn.Parameter(torch.empty(out_fields[-1], in_fields[-1], basis.shape[0]))
        # Scale by the energy of the sampled basis and the widest fan-in.
        std = math.sqrt(2 * out_rep_size / (in_fields[-1] * basis.square().sum().item()))
        nn.init.normal_(self.weights, std=std)
        self._filters = {}
        self.use_fourier = False

    def expand(self, profile):
        num_in, num_out = self.in_fields[profile], self.out_fields[profile]
        coefficients = self.weights[:num_out, :num_in]
        blocks = (coefficients @ self.basis).reshape(
            num_out, num_in, self.out_rep_size, self.in_rep_size, self.kernel, self.kernel,
        )
        return blocks.transpose(1, 2).reshape(
            num_out * self.out_rep_size, num_in * self.in_rep_size, self.kernel, self.kernel,
        )

    def _fourier_filter(self, profile):
        num_in, num_out = self.in_fields[profile], self.out_fields[profile]
        # A C4 pointwise block is circulant. Its first column is the kernel of
        # circular convolution, in the same orientation order as the input.
        column = self.weights[:num_out, :num_in] @ self.basis[:, ::self.in_rep_size]
        return torch.fft.rfft(column, dim=-1)

    def train(self, mode=True):
        if mode:
            self._filters.clear()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._filters.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn, recurse=True):
        self._filters.clear()
        return super()._apply(fn, recurse=recurse)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._filters = {}
        return replica

    def forward(self, x, profile):
        expand = self._fourier_filter if self.use_fourier else self.expand
        if self.training or torch.is_grad_enabled():
            weight = expand(profile)
        else:
            weight = self._filters.get(profile)
            if weight is None:
                # Cache at parameter precision even if warmed under autocast.
                with torch.autocast(x.device.type, enabled=False):
                    weight = expand(profile)
                self._filters[profile] = weight
        if self.use_fourier:
            batch, _, height, width = x.shape
            fields = x.reshape(batch, self.in_fields[profile], self.in_rep_size, height, width)
            spectrum = torch.fft.rfft(fields, dim=2)
            result = torch.einsum('oif,bifhw->bofhw', weight, spectrum)
            return torch.fft.irfft(result, n=self.out_rep_size, dim=2).reshape(batch, -1, height, width)
        return F.conv2d(x, weight, padding=self.kernel // 2)

    def restrict(self, profile):
        num_in, num_out = self.in_fields[profile], self.out_fields[profile]
        self._filters.clear()
        self.in_fields, self.out_fields = (num_in,), (num_out,)
        self.weights = nn.Parameter(self.weights[:num_out, :num_in].detach().clone(),
                                    requires_grad=self.weights.requires_grad)


class ProfileNorm(nn.Module):
    def __init__(self, widths, repr_size):
        super().__init__()
        # Regular fields are permutations. BatchNorm3d shares its affine and
        # statistics across the entire orientation axis of each field.
        self.repr_size = repr_size
        self.norms = nn.ModuleList(nn.BatchNorm3d(width) for width in widths)

    def forward(self, x, profile):
        batch, channels, height, width = x.shape
        norm = self.norms[profile]
        fields = x.reshape(batch, channels // self.repr_size, self.repr_size, height, width)
        if norm.training and batch * self.repr_size * height * width == 1:
            # A singleton at a 1x1 final stage is possible for the trivial group.
            return F.batch_norm(fields, norm.running_mean, norm.running_var,
                                norm.weight, norm.bias, training=False, eps=norm.eps).reshape_as(x)
        return norm(fields).reshape_as(x)

    def restrict(self, profile):
        self.norms = nn.ModuleList([self.norms[profile]])


class ProfileBlock(nn.Module):
    def __init__(self, widths, heads, conv, norm, kernel, mlp_ratio, window_size, use_sdpa):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.use_sdpa = use_sdpa
        hidden = tuple(width * mlp_ratio for width in widths)
        self.norm1 = norm(widths)
        self.norm2 = norm(widths)
        self.qkv = nn.ModuleList(conv(widths, widths, kernel) for _ in range(3))
        self.proj = conv(widths, widths, 1)
        self.up = conv(widths, hidden, 1)
        self.up_norm = norm(hidden)
        self.down = conv(hidden, widths, 1)
        self.down_norm = norm(widths)
        self.alpha1 = nn.Parameter(torch.ones(()))
        self.alpha2 = nn.Parameter(torch.ones(()))

    def forward(self, x, profile):
        normalized = self.norm1(x, profile)
        q, k, v = (projection(normalized, profile) for projection in self.qkv)
        if self.window_size is None:
            mixed = _sdpa_spatial(q, k, v, self.heads, 0.0)
        else:
            mixed = window_attention(q, k, v, self.heads, self.window_size, self.use_sdpa)
        x = x + self.alpha1 * self.proj(mixed, profile)
        hidden = self.up(self.norm2(x, profile), profile)
        hidden = self.up_norm(hidden, profile).relu()
        return x + self.alpha2 * self.down_norm(self.down(hidden, profile), profile)


@dataclass(frozen=True)
class BudgetREViTConfig:
    """Widths are regular-field counts, ordered from cheapest to widest."""

    profiles: tuple[tuple[int, ...], ...]
    depths: tuple[int, ...]
    heads: tuple[int, ...]
    group: str = 'C4'
    in_channels: int = 3
    num_classes: int = 10
    image_size: int = 32
    stem_downsample: int = 2
    stem_width: int | None = None
    window_size: int | None = None
    qkv_kernel: int = 3
    mlp_ratio: int = 4
    router_hidden: int = 32
    cost_weight: float = 0.2
    profile_costs: tuple[float, ...] | None = None
    window_sdpa: bool = False


class BudgetAdaptiveREViT(nn.Module):
    """One invariant route per image with a shared stem and nested REViT stages."""

    def __init__(self, config: BudgetREViTConfig):
        super().__init__()
        self._validate(config)
        self.config = config
        self.profiles = config.profiles
        self.gspace = get_gspace(config.group)
        regular = self.gspace.regular_repr
        self.order = regular.size
        self.input_type = FieldType(self.gspace, [self.gspace.trivial_repr] * config.in_channels)
        self.stem_width = config.stem_width or self.profiles[0][0]
        by_stage = list(zip(*self.profiles))
        bases = {}

        def conv(inputs, outputs, kernel, in_repr=regular):
            key = (in_repr.name, kernel)
            if key not in bases:
                bases[key] = sampled_basis(self.gspace, in_repr, regular, kernel)
            return ProfileConv(bases[key], inputs, outputs, in_repr.size, regular.size, kernel)

        def norm(widths):
            return ProfileNorm(widths, self.order)

        self.stem = conv((config.in_channels,), (self.stem_width,), 5, self.gspace.trivial_repr)
        self.stem_norm = norm((self.stem_width,))
        self.entries = nn.ModuleList()
        self.entry_norms = nn.ModuleList()
        self.stages = nn.ModuleList()
        inputs = (self.stem_width,) * len(self.profiles)
        for stage, (widths, depth, heads) in enumerate(zip(by_stage, config.depths, config.heads)):
            self.entries.append(conv(inputs, widths, 1 if stage == 0 else 3))
            self.entry_norms.append(norm(widths))
            self.stages.append(nn.ModuleList(
                ProfileBlock(widths, heads, conv, norm, config.qkv_kernel, config.mlp_ratio,
                             config.window_size, config.window_sdpa) for _ in range(depth)
            ))
            inputs = widths

        self.classifier = nn.Linear(self.profiles[-1][-1], config.num_classes)
        self.router = None
        if len(self.profiles) > 1:
            self.router = nn.Sequential(nn.Linear(2 * self.stem_width + 1, config.router_hidden),
                                        nn.GELU(), nn.Linear(config.router_hidden, len(self.profiles)))
        self.profile_costs = config.profile_costs or tuple(self.estimate_macs(profile) for profile in self.profiles)
        span = self.profile_costs[-1] - self.profile_costs[0]
        self.budget_thresholds = tuple((c - self.profile_costs[0]) / span for c in self.profile_costs) if span else (0.0,)

    @staticmethod
    def _validate(config):
        if config.group not in ('C4', 'D4', 'Z2'):
            raise ValueError('Use C4 or D4, or Z2 for the trivial-group baseline')
        if not config.profiles or not config.heads or len(config.heads) != len(config.depths):
            raise ValueError('Supply at least one profile and matching nonempty heads and depths')
        if config.window_size is None and len(config.depths) != 1:
            raise ValueError('The global backbone has one stage')
        positive = (*config.depths, *config.heads, config.in_channels, config.num_classes,
                    config.image_size, config.stem_downsample, config.qkv_kernel,
                    config.mlp_ratio, config.router_hidden)
        if any(not isinstance(value, int) or value < 1 for value in positive):
            raise ValueError('Model dimensions, depths, heads and kernels must be positive integers')
        if config.qkv_kernel % 2 != 1 or (config.window_size is not None and (
            not isinstance(config.window_size, int) or config.window_size < 1
        )):
            raise ValueError('Use an odd QKV kernel and a positive window size')
        for profile in config.profiles:
            if len(profile) != len(config.heads) or any(
                not isinstance(width, int) or width < head or width % head
                for width, head in zip(profile, config.heads)
            ):
                raise ValueError('Each stage width must be a positive multiple of its head count')
        for small, large in zip(config.profiles, config.profiles[1:]):
            if small == large or any(a > b for a, b in zip(small, large)):
                raise ValueError('Profiles must increase in at least one stage and never shrink')
        if config.stem_width is not None and (
            not isinstance(config.stem_width, int) or not 1 <= config.stem_width <= config.profiles[0][0]
        ):
            raise ValueError('The shared stem must fit in the smallest first-stage profile')
        if not math.isfinite(config.cost_weight) or config.cost_weight < 0:
            raise ValueError('cost_weight must be finite and nonnegative')
        if config.profile_costs is not None:
            costs = config.profile_costs
            if (len(costs) != len(config.profiles) or any(not math.isfinite(c) or c <= 0 for c in costs)
                    or any(a >= b for a, b in zip(costs, costs[1:]))):
                raise ValueError('Profile costs must be positive, finite and strictly increasing')

    def estimate_macs(self, profile):
        """Dense post-stem conv, attention and classifier MACs at the reference resolution."""
        cfg = self.config
        side = math.ceil(cfg.image_size / cfg.stem_downsample)
        in_channels = self.stem_width * self.order
        macs = 0
        for stage, (fields, depth) in enumerate(zip(profile, cfg.depths)):
            channels = fields * self.order
            if stage:
                side = math.ceil(side / 2)
            tokens = side**2
            macs += tokens * in_channels * channels * (9 if stage else 1)
            window, pad = (side, 0) if cfg.window_size is None else window_layout(side, cfg.window_size)
            attention = 2 * (side + 2 * pad)**2 * window**2 * channels
            projections = tokens * channels**2 * (3 * cfg.qkv_kernel**2 + 1 + 2 * cfg.mlp_ratio)
            macs += depth * (projections + attention)
            in_channels = channels
        return macs + profile[-1] * cfg.num_classes

    def _stem(self, x):
        x = centered_downsample(x, self.config.stem_downsample)
        return self.stem_norm(self.stem(x, 0), 0).relu()

    def _features(self, x, profile):
        for stage, blocks in enumerate(self.stages):
            if stage:
                x = centered_downsample(x, 2)
            x = self.entry_norms[stage](self.entries[stage](x, profile), profile).relu()
            for block in blocks:
                x = block(x, profile)
        return x

    def _fields(self, x):
        return x.reshape(x.shape[0], -1, self.order, *x.shape[-2:])

    def _predict(self, x, profile):
        x = self._fields(self._features(x, profile)).amax(dim=2).mean(dim=(-2, -1))
        return F.linear(x, self.classifier.weight[:, :x.shape[1]], self.classifier.bias)

    @staticmethod
    def _validate_budget(early, budget):
        if isinstance(budget, torch.Tensor):
            if budget.ndim > 1 or (budget.ndim == 1 and budget.shape[0] != early.shape[0]):
                raise ValueError('budget must be a scalar or one value per image')
            budget = budget.to(device=early.device, dtype=torch.float32).expand(early.shape[0])
            if not ((budget >= 0) & (budget <= 1)).all():
                raise ValueError('budget values must lie in [0, 1]')
        else:
            if not 0 <= budget <= 1:
                raise ValueError('budget must lie in [0, 1]')
            budget = torch.full((early.shape[0],), budget, device=early.device, dtype=torch.float32)
        return budget

    def _routing(self, early, budget, sample_budgets):
        costs = early.new_tensor(self.budget_thresholds, dtype=torch.float32)
        if budget is None and sample_budgets:
            budget = costs[torch.randint(len(self.profiles), (early.shape[0],), device=early.device)]
        else:
            budget = self._validate_budget(early, 1.0 if budget is None else budget)
        fields = self._fields(early.float())
        mean = fields.mean(dim=(2, 3, 4))
        peak = fields.amax(dim=(2, 3, 4))
        descriptor = torch.cat((mean, peak, budget[:, None]), dim=1)
        with torch.autocast(early.device.type, enabled=False):
            scores = self.router(descriptor.to(self.router[0].weight.dtype)).float()
        return scores.masked_fill(costs[None, :] > budget[:, None], -torch.inf), costs

    def _dispatch(self, early, routes):
        # One host transfer for the routing decision. Profile kernels stay on device.
        selected = routes.tolist()
        if min(selected) == max(selected):
            return self._predict(early, selected[0])
        buckets = [[] for _ in self.profiles]
        for row, profile in enumerate(selected):
            buckets[profile].append(row)
        outputs = []
        restore = [0] * len(selected)
        offset = 0
        for profile, bucket in enumerate(buckets):
            if bucket:
                rows = torch.tensor(bucket, device=early.device)
                outputs.append(self._predict(early.index_select(0, rows), profile))
                for position, row in enumerate(bucket, offset):
                    restore[row] = position
                offset += len(bucket)
        return torch.cat(outputs).index_select(0, torch.tensor(restore, device=early.device))

    def forward(self, x, *, budget=None, profile=None, targets=None, return_routes=False):
        """Return logits, optionally route IDs, or (routed logits, per-image training loss).

        A budget of zero admits the cheapest profile and one admits all profiles.
        Supplying targets trains every profile and the cost-sensitive router.
        Omitting targets executes only selected profiles. Use eval mode for routing.
        """
        if x.ndim != 4 or x.shape[0] == 0 or x.shape[-1] != x.shape[-2] or x.shape[-1] == 0:
            raise ValueError('Expected a nonempty batch of square NCHW images')
        if profile is not None and (not isinstance(profile, int) or not 0 <= profile < len(self.profiles)):
            raise ValueError('profile must be a valid integer index')
        if profile is not None and (budget is not None or targets is not None):
            raise ValueError('Specify a forced profile by itself')
        if targets is not None and return_routes:
            raise ValueError('Training returns logits and per-image losses')
        early = self._stem(x)
        if profile is not None or self.router is None:
            if budget is not None:
                self._validate_budget(early, budget)
            profile = 0 if profile is None else profile
            logits = self._predict(early, profile)
            if targets is not None:
                return logits, F.cross_entropy(logits, targets, reduction='none')
            if return_routes:
                return logits, torch.full((x.shape[0],), profile, dtype=torch.long, device=x.device)
            return logits
        scores, costs = self._routing(early, budget, sample_budgets=targets is not None and self.training)
        routes = scores.argmax(dim=1)
        if targets is None:
            logits = self._dispatch(early, routes)
            return (logits, routes) if return_routes else logits

        predictions = [self._predict(early, index) for index in range(len(self.profiles))]
        errors = torch.stack([F.cross_entropy(logits, targets, reduction='none') for logits in predictions], dim=1)
        # All profiles receive CE gradients, irrespective of current routing.
        # Detached errors give the router a cost-sensitive per-image objective.
        risk = errors.detach() + self.config.cost_weight * costs
        router_loss = (scores.softmax(dim=1) * risk).sum(dim=1)
        logits = torch.stack(predictions, dim=1)[torch.arange(x.shape[0], device=x.device), routes]
        return logits, errors.mean(dim=1) + router_loss

    def export_profile(self, profile):
        """Create an independent fixed model with physically cropped coefficients and norms."""
        if not isinstance(profile, int) or not 0 <= profile < len(self.profiles):
            raise ValueError('profile must be a valid integer index')
        fixed = copy.deepcopy(self)
        fixed.router = None
        fixed.profiles = (self.profiles[profile],)
        fixed.profile_costs = (self.profile_costs[profile],)
        fixed.budget_thresholds = (0.0,)
        for module in (*fixed.entries.modules(), *fixed.entry_norms.modules(), *fixed.stages.modules()):
            if isinstance(module, (ProfileConv, ProfileNorm)):
                module.restrict(profile)
        features = fixed.profiles[0][-1]
        fixed.classifier.weight = nn.Parameter(fixed.classifier.weight[:, :features].detach().clone(),
                                               requires_grad=self.classifier.weight.requires_grad)
        fixed.classifier.in_features = features
        fixed.config = replace(self.config, profiles=fixed.profiles, stem_width=self.stem_width,
                               profile_costs=fixed.profile_costs)
        return fixed

    @torch.no_grad()
    def prune_profile(self, profile, widths):
        """Crop a fixed-width model to whole-field prefixes for later fine-tuning."""
        if not isinstance(profile, int) or not 0 <= profile < len(self.profiles):
            raise ValueError('profile must be a valid integer index')
        widths = tuple(widths)
        old_widths = self.profiles[profile]
        if len(widths) != len(old_widths) or any(a > b for a, b in zip(widths, old_widths)):
            raise ValueError('Pruned widths must fit inside the source profile at every stage')
        source = self if len(self.profiles) == 1 else self.export_profile(profile)
        config = replace(source.config, profiles=(widths,), stem_width=self.stem_width,
                         profile_costs=None)
        with torch.random.fork_rng(devices=[]):
            fixed = BudgetAdaptiveREViT(config)
        fixed.to(device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)
        source_state = source.state_dict()
        for name, tensor in fixed.state_dict().items():
            prefix = tuple(slice(size) for size in tensor.shape)
            tensor.copy_(source_state[name][prefix])
        return fixed.train(self.training)

    def use_cyclic_fourier(self):
        """Use an unfused C4 Fourier reference for regular-to-regular 1x1 layers."""
        if self.config.group != 'C4':
            raise ValueError('The cyclic Fourier reference is available only for C4')
        for module in self.modules():
            if isinstance(module, ProfileConv) and module.kernel == 1:
                module._filters.clear()
                module.use_fourier = True
        return self


def parse_profiles(spec):
    """Parse e.g. '4,8,16,32;8,16,32,64' (or '4;8' for global attention)."""
    try:
        return tuple(tuple(int(width) for width in profile.split(',')) for profile in spec.split(';'))
    except ValueError as error:
        raise ValueError('Use semicolon-separated profiles of comma-separated field counts') from error
