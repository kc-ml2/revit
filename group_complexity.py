"""Forward arithmetic models; estimates FFT work explicitly rather than omitting it.

Real MAC=2 FLOPs; complex MAC=8 real FLOPs. RFFT/IRFFT work is an algorithm-
independent comparison proxy 2.5*N*log2(N) per real signal, not a measured FFT
instruction count. Includes weight transforms on every forward, as implemented.
Basis expansion, normalization, activations, copies, and backward are excluded.
"""

import math


MODEL_CONFIGS = ("quadratic", "linear", "fourier")


def sparse_degree(group_order, group_kind="cyclic"):
    if group_kind == "cyclic":
        return min(group_order, 3)
    if group_kind == "dihedral" and group_order % 2 == 0:
        return 2 if group_order == 2 else (3 if group_order == 4 else 4)
    raise ValueError("Invalid group kind/order")


def regular_conv_cost(in_fields, out_fields, group_order, area, kernel_size=1,
                      stride=1, model_config="quadratic", group_kind="cyclic"):
    if model_config not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_config: {model_config}")
    if min(in_fields, out_fields, group_order, area, kernel_size, stride) <= 0:
        raise ValueError("Positive dimensions required")
    i, o, g, s, k = in_fields, out_fields, group_order, area, kernel_size
    spatial = mixing = transforms = 0
    if model_config == "quadratic":
        mixing = 2 * s * i * o * g**2 * k**2
    else:
        if k != 1 or stride != 1:
            spatial = 2 * s * i * g * k**2
        if model_config == "linear":
            degree = sparse_degree(g, group_kind)
            mixing = 2 * s * g * i * o + (2 * degree - 1) * s * g * o
        else:
            if group_kind not in ("cyclic", "dihedral") or (group_kind == "dihedral" and g % 2):
                raise ValueError("Fourier mode requires cyclic or even-order dihedral group")
            branches = 2 if group_kind == "dihedral" else 1
            n = g // branches
            f = n // 2 + 1
            mixing = 8 * s * i * o * branches**2 * f
            mixing += 2 * s * o * f * branches * (branches - 1)
            # Input FFTs, output inverse FFTs, and learned real kernel FFTs.
            transforms = 2.5 * n * math.log2(n) * branches * (s * (i + o) + i * o)
    counted = spatial + mixing
    return {"counted_flops": counted, "fft_flops_estimate": transforms,
            "spatial_flops": spatial, "mixing_flops": mixing,
            "flops": counted + transforms}
