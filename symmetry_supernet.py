"""Train shared D4 kernels and export width and subgroup profiles."""

from dataclasses import dataclass, replace

import torch
from escnn.nn import FieldType

from budget_revit import (
    BudgetAdaptiveREViT, BudgetREViTConfig, ProfileBlock, ProfileConv, ProfileNorm,
    REViTBackbone, sampled_basis,
)
from group_space import D4_SUBGROUPS, canonical_square_group, get_gspace


@dataclass(frozen=True)
class SymmetrySupernetConfig:
    """One full-width D4 backbone and profiles with an optional leading stem width."""

    backbone: BudgetREViTConfig
    profiles: tuple[tuple[str, tuple[int, ...]], ...]


def parse_symmetry_profiles(spec):
    """Parse 'D4:12,24,48,96;C4:6,12,24,48'."""
    try:
        profiles = [item.split(':') for item in spec.split(';')]
        return tuple((group.strip(), tuple(map(int, widths.split(',')))) for group, widths in profiles)
    except ValueError as error:
        raise ValueError('Use semicolon-separated GROUP:WIDTH,WIDTH entries') from error


def split_width_profile(widths, stage_count):
    widths = tuple(widths)
    if len(widths) == stage_count:
        return None, widths
    if len(widths) == stage_count + 1:
        return widths[0], widths[1:]
    raise ValueError(f'Supply {stage_count} stage widths, optionally prefixed by the stem width')


def default_symmetry_profiles(dims, heads):
    """Use three head-aligned widths per group when the preset permits them."""
    if any(head < 1 for head in heads):
        raise ValueError('Attention head counts must be positive')
    widths = dict.fromkeys(
        tuple(max(head, (dim * numerator // (3 * head)) * head) for dim, head in zip(dims, heads))
        for numerator in (1, 2, 3)
    )
    return tuple((group, width) for group in ('D4', 'C4') for width in widths)


def training_profile(reference, count, step):
    """Alternate the full D4 profile with a round-robin of all other profiles."""
    if count == 1 or step % 2 == 0:
        return reference
    other = step // 2 % (count - 1)
    return other + (other >= reference)


def subgroup_orbit(parent, group):
    """D4's identity orbit under the subgroup in standalone ESCNN channel order."""
    subgroup, include, _ = parent.restrict(D4_SUBGROUPS[group][0])
    if subgroup != get_gspace(group):
        raise RuntimeError(f'D4 embedding does not match the standalone {group} gspace')
    identity = parent.regular_repr(parent.fibergroup.identity)[:, 0].argmax()
    return tuple(int(parent.regular_repr(include(h))[:, identity].argmax())
                 for h in subgroup.fibergroup.elements)


def crop_basis(basis, kernel, in_order, out_order, orbit):
    blocks = basis.reshape(-1, out_order, in_order, kernel, kernel)
    # Input RGB/gray channels carry separate trivial fields in the lifting layer.
    return blocks[:, orbit][:, :, (0,) if in_order == 1 else orbit].flatten(1).contiguous()


class SubgroupProfileConv(ProfileConv):
    """Slice each D4 basis block before expansion, sharing its coefficients."""

    def __init__(self, basis, in_fields, out_fields, in_order, kernel, groups, cropped_bases):
        super().__init__(basis, in_fields, out_fields, in_order, 8, kernel)
        self.groups = groups
        self.orders = tuple(len(D4_SUBGROUPS[group][1]) for group in groups)
        for group, cropped in cropped_bases.items():
            if group != 'D4':
                self.register_buffer(f'basis_{group}', cropped, persistent=False)

    def expand(self, profile):
        group = self.groups[profile]
        basis = self.basis if group == 'D4' else getattr(self, f'basis_{group}')
        order = self.orders[profile]
        return self._expand_basis(profile, basis, 1 if self.in_rep_size == 1 else order, order)

    def forward(self, x, profile):
        # Validation sweeps reuse the currently selected kernels. Retaining every
        # expanded D4/C4 filter would multiply the largest model's resident size.
        if profile not in self._filters:
            self._filters.clear()
        return super().forward(x, profile)


class SymmetrySupernet(REViTBackbone):
    """An offline profile selector with one D4 coefficient bank for all widths and groups."""

    def __init__(self, config: SymmetrySupernetConfig):
        super().__init__()
        backbone = config.backbone
        if backbone.group != 'D4' or len(backbone.profiles) != 1:
            raise ValueError('The supernet backbone must specify a single full-width D4 profile')
        BudgetAdaptiveREViT._validate(backbone)
        full = backbone.profiles[0]
        profiles = tuple((canonical_square_group(group), tuple(widths)) for group, widths in config.profiles)
        profile_widths = tuple(split_width_profile(widths, len(full)) for _, widths in profiles)
        groups = tuple(group for group, _ in profiles)
        stage_widths = tuple(widths for _, widths in profile_widths)
        references = [i for i, (group, widths) in enumerate(zip(groups, stage_widths))
                      if group == 'D4' and widths == full]
        if not references:
            raise ValueError('Supply profiles including the full-width D4 profile')
        for group, widths in zip(groups, stage_widths):
            if group not in D4_SUBGROUPS:
                raise ValueError(f'{group} is not a supported D4 subgroup')
            if any(
                not isinstance(width, int) or not head <= width <= cap or width % head
                for width, cap, head in zip(widths, full, backbone.heads)
            ):
                raise ValueError('Profile widths must be head-aligned and fit inside the D4 backbone')

        self.config = backbone
        self.supernet_config = replace(config, profiles=profiles)
        self.profiles = profiles
        self.stage_widths = stage_widths
        default_stem = backbone.stem_width or min(widths[0] for widths in stage_widths)
        self.stem_widths = tuple(default_stem if stem is None else stem for stem, _ in profile_widths)
        if any(not isinstance(stem, int) or not 1 <= stem <= widths[0]
               for stem, widths in zip(self.stem_widths, stage_widths)):
            raise ValueError('Each stem width must be a positive integer and fit that profile’s first stage')
        if len(set(zip(groups, self.stem_widths, stage_widths))) != len(profiles):
            raise ValueError('Supply unique group, stem width and stage width combinations')
        self.reference_profile = max(references, key=self.stem_widths.__getitem__)
        self.stem_width = self.stem_widths[self.reference_profile]
        parent = get_gspace('D4')
        orders = tuple(len(D4_SUBGROUPS[group][1]) for group in groups)
        self.orders = orders
        orbits = {group: subgroup_orbit(parent, group) for group in dict.fromkeys(groups)}
        bases = {}

        def conv(inputs, outputs, kernel, lifting=False):
            key = (kernel, lifting)
            in_repr = parent.trivial_repr if lifting else parent.regular_repr
            if key not in bases:
                basis = sampled_basis(parent, in_repr, parent.regular_repr, kernel)
                crops = {group: crop_basis(basis, kernel, in_repr.size, 8, orbit)
                         for group, orbit in orbits.items() if group != 'D4'}
                bases[key] = (basis, crops)
            basis, crops = bases[key]
            return SubgroupProfileConv(basis, inputs, outputs, in_repr.size, kernel, groups, crops)

        def norm(widths):
            return ProfileNorm(widths, orders)

        stem_widths = self.stem_widths
        self.stem = conv((backbone.in_channels,) * len(profiles), stem_widths, 5, lifting=True)
        self.stem_norm = norm(stem_widths)
        self._build_stages(backbone, stage_widths, stem_widths, conv, norm)

    def forward(self, x, *, profile=None):
        if profile is None:
            profile = self.reference_profile
        if not isinstance(profile, int) or not 0 <= profile < len(self.profiles):
            raise ValueError('profile must be a valid integer index')
        if x.ndim != 4 or x.shape[0] == 0 or x.shape[-1] != x.shape[-2] or x.shape[-1] == 0:
            raise ValueError('Expected a nonempty batch of square NCHW images')
        return self._classify(self._features(self._stem(x, profile), profile), self.orders[profile])

    def export_profile(self, profile):
        """Convert a trained path into an ordinary trainable, static REViT checkpoint."""
        if not isinstance(profile, int) or not 0 <= profile < len(self.profiles):
            raise ValueError('profile must be a valid integer index')
        group, _ = self.profiles[profile]
        return _materialize(self, profile, group, self.stage_widths[profile], self.stem_widths[profile])


def crop_d4_profile(source, group, widths, profile=None):
    """Cold-crop a width-only D4 checkpoint to a subgroup for baseline comparisons."""
    if not isinstance(source, BudgetAdaptiveREViT) or source.config.group != 'D4':
        raise ValueError('Cold cropping expects a D4 BudgetAdaptiveREViT checkpoint')
    if profile is None:
        profile = len(source.profiles) - 1
    if not isinstance(profile, int) or not 0 <= profile < len(source.profiles):
        raise ValueError('profile must be a valid integer index')
    group = canonical_square_group(group)
    if group not in D4_SUBGROUPS:
        raise ValueError(f'{group} is not a supported D4 subgroup')
    source_widths = source.profiles[profile]
    stem_width, widths = split_width_profile(widths, len(source_widths))
    stem_width = source.stem_width if stem_width is None else stem_width
    if any(w > cap for w, cap in zip(widths, source_widths)) or stem_width > source.stem_width:
        raise ValueError('Cropped widths must fit the source profile')
    return _materialize(source, profile, group, widths, stem_width)


@torch.no_grad()
def _materialize(source, profile, group, widths, stem_width):
    config = replace(source.config, profiles=(widths,), group=group, stem_width=stem_width,
                     profile_costs=None)
    with torch.random.fork_rng(devices=[]):
        fixed = BudgetAdaptiveREViT(config)
    parent_modules = dict(source.named_modules())
    orbit = subgroup_orbit(get_gspace('D4'), group)
    transfers = {}
    # Solve the tiny field-pair changes of basis on CPU once per kernel and field type.
    for name, child in fixed.named_modules():
        if isinstance(child, ProfileConv):
            parent = parent_modules[name]
            weights = parent.weights[:child.out_fields[0], :child.in_fields[0]]
            if group != 'D4':
                key = (parent.kernel, parent.in_rep_size)
                if key not in transfers:
                    cropped = crop_basis(parent.basis.detach().cpu().double(), parent.kernel,
                                         parent.in_rep_size, parent.out_rep_size, orbit)
                    basis = child.basis.double()
                    transfer = cropped @ torch.linalg.pinv(basis, rtol=1e-6)
                    if not torch.allclose(transfer @ basis, cropped, atol=2e-6, rtol=1e-5):
                        raise RuntimeError(f'The cropped {group} kernel falls outside the standalone basis')
                    transfers[key] = transfer.to(device=weights.device, dtype=torch.float32)
                with torch.autocast(weights.device.type, enabled=False):
                    weights = weights.float() @ transfers[key]
            child.weights.copy_(weights)
        elif isinstance(child, ProfileNorm):
            parent = parent_modules[name]
            norm_profile = 0 if len(parent.norms) == 1 else profile
            state = parent.norms[norm_profile].state_dict()
            for key, tensor in child.norms[0].state_dict().items():
                tensor.copy_(state[key][tuple(slice(n) for n in tensor.shape)])
        elif isinstance(child, ProfileBlock):
            parent = parent_modules[name]
            child.alpha1.copy_(parent.alpha1)
            child.alpha2.copy_(parent.alpha2)
    fixed.classifier.weight.copy_(source.classifier.weight[:, :widths[-1]])
    fixed.classifier.bias.copy_(source.classifier.bias)
    return fixed.to(device=source.classifier.weight.device, dtype=source.classifier.weight.dtype).train(source.training)
