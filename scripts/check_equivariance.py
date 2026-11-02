"""Reproducible equivariance audit shared by pytest and the research report.

Strict cases check exact lattice/fiber symmetries up to FP32 error. Diagnostic
cases deliberately measure inherited sampling, window, and stochastic limits;
passing a diagnostic means detecting a violation, not certifying equivariance.
No architecture or training defaults are changed by this audit.
"""

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from escnn.nn import FieldType, GeometricTensor

from group_complexity import MODEL_CONFIGS
from group_space import get_gspace
from model_config import MODEL_PRESETS
from regular_group_ops import SparseRightGroupMix, regular_conv
from revit_windowed_gcsa import Rot2DBlockV2, Rot2DTransformerV2, Rot2DWindowAttention


GROUPS = ("Z2", "D2", "C4", "C8", "C12", "C16", "D4", "D8", "D12", "D16")
LATTICE_GROUPS = ("C4", "D4")


@dataclass(frozen=True)
class Case:
    protocol: str
    model_config: str
    group: str
    kernel: int = 1
    side: int = 9
    backend: str = "bmm"
    training: bool = False
    diagnostic: bool = False

    @property
    def id(self):
        return (f"{self.protocol}-{self.model_config}-{self.group}-k{self.kernel}-"
                f"n{self.side}-{self.backend}-{'train' if self.training else 'eval'}")


def cases():
    result = []
    for mode in MODEL_CONFIGS:
        for group in GROUPS:
            result.append(Case("pointwise_fiber", mode, group))
        for group in LATTICE_GROUPS:
            for kernel, side in itertools.product((1, 3, 5), (8, 9)):
                result.append(Case("spatial_convolution", mode, group, kernel, side))
            for side in (16, 17):
                result.append(Case("stride_two", mode, group, 3, side, diagnostic=side == 16))
            for kernel, side, backend in itertools.product((1, 3), (13, 14), ("bmm", "sdpa")):
                result.append(Case("window_attention", mode, group, kernel, side, backend,
                                   diagnostic=side == 13))
            for training in (False, True):
                result.append(Case("residual_block", mode, group, 3, 14, training=training))
            for backend in ("bmm", "sdpa"):
                result.append(Case("attention_gradient", mode, group, 3, 14, backend, training=True))
            result.append(Case("full_model", mode, group, 3, 33))
            result.append(Case("full_model", mode, group, 3, 224, diagnostic=True))
        for group in ("C8", "D8"):
            result.append(Case("nonlattice_attention", mode, group, 1, 28, diagnostic=True))
        result.append(Case("stochastic_block", mode, "D4", 3, 14, training=True, diagnostic=True))
    return result


def error_metrics(actual, expected):
    a, b = actual.detach().double(), expected.detach().double()
    difference = a - b
    return {"relative_l2": (difference.norm() / max(a.norm(), b.norm(), 1e-30)).item(),
            "max_abs": difference.abs().max().item(),
            "rms_error": difference.square().mean().sqrt().item(),
            "reference_rms": b.square().mean().sqrt().item()}


def _exercise_sparse_weights(module):
    # Test nonidentity stencils instead of relying on the identity initialization.
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, SparseRightGroupMix):
                child.weight[:, 1:].normal_(std=.15)


def _lattice_element(gs, element):
    matrix = gs.basespace_action(element)
    return bool(np.allclose(matrix, np.round(matrix), atol=1e-6))


def _summarize(case, rows):
    return {"case_id": case.id, **asdict(case), "group_order": get_gspace(case.group).fibergroup.order(),
            "elements_tested": len({r["element"] for r in rows}), "measurements": len(rows),
            "max_relative_l2": max(r["relative_l2"] for r in rows),
            "max_abs": max(r["max_abs"] for r in rows), "by_element": rows}


def _full_features(model, tensor):
    features = {}
    output = model.stem(tensor)
    features["stem"] = output
    for j, stage in enumerate(model.stages):
        output = stage(output)
        features[f"stage_{j + 1}"] = output
    return features, model.head(output)


def run_case(case):
    """Measure every group element; never fit tolerances to a measured result."""
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    gs = get_gspace(case.group)
    ft = FieldType(gs, 4 * [gs.regular_repr])
    protocol = case.protocol
    rows = []
    if protocol == "full_model":
        # Five stride-2 operations. n=33 is 1 mod 32, so every input to
        # downsampling is odd. Whole-stage windows remove the partition
        # obstruction while retaining nontrivial QK attention at every stage.
        window = 7
        model = Rot2DTransformerV2(
            gspace=gs, **MODEL_PRESETS["tiny"], num_classes=1000,
            window_size=window, qkv_kernel_size=case.kernel,
            model_config=case.model_config, drop_path_rate=0, fast_init=True,
        ).eval()
        stage_side = (case.side + 3) // 4
        windows = []
        for stage in model.stages:
            effective_window = stage_side if case.side == 33 else window
            for block in stage.blocks:
                block.attn.window_size = effective_window
            windows.append(effective_window)
            stage_side = (stage_side + 1) // 2
        _exercise_sparse_weights(model)
        torch.manual_seed(2026)
        raw = GeometricTensor(torch.randn(2, 3, case.side, case.side), model.stem.in_type)
        with torch.no_grad():
            expected_features, expected_logits = _full_features(model, raw.tensor)
            for element in gs.fibergroup.elements:
                actual_features, logits = _full_features(model, raw.transform(element).tensor)
                for name, feature in expected_features.items():
                    rows.append({"element": str(element), "location": name, "lattice_action": True,
                                 **error_metrics(actual_features[name].tensor, feature.transform(element).tensor)})
                rows.append({"element": str(element), "location": "logits", "lattice_action": True,
                             **error_metrics(logits, expected_logits)})
        result = _summarize(case, rows)
        result["window_sizes"] = windows
        result["window_policy"] = "whole_stage" if case.side == 33 else "fixed"
        result["model_size"] = "tiny"
        result["dims"], result["depths"], result["heads"] = (
            MODEL_PRESETS["tiny"][key] for key in ("dims", "depths", "heads"))
        result["mlp_ratio"], result["num_classes"] = 4, 1000
        result["stage_output_sides"] = [f.tensor.shape[-1] for f in expected_features.values()]
        return result

    if protocol in ("pointwise_fiber", "spatial_convolution", "stride_two"):
        output_type = FieldType(gs, 3 * [gs.regular_repr])
        model = regular_conv(ft, output_type, case.kernel, stride=2 if protocol == "stride_two" else 1,
                             mode=case.model_config)
    elif protocol in ("residual_block", "stochastic_block"):
        stochastic = protocol == "stochastic_block"
        model = Rot2DBlockV2(ft, 2, 7, 2, .25 if stochastic else 0., .2 if stochastic else 0.,
                             0., case.kernel, False, conv_mode=case.model_config)
    else:
        model = Rot2DWindowAttention(ft, 2, window_size=7, qkv_kernel_size=case.kernel,
                                     conv_mode=case.model_config)
        model.use_sdpa = case.backend == "sdpa"
    _exercise_sparse_weights(model)
    model.train(case.training)
    torch.manual_seed(2026)
    raw = GeometricTensor(torch.randn(2, ft.size, case.side, case.side), ft)
    transform = "transform_fibers" if protocol == "pointwise_fiber" else "transform"

    if protocol == "attention_gradient":
        raw.tensor.requires_grad_()
        output = model(raw)
        probe = torch.randn_like(output.tensor)
        args = (raw.tensor,) + tuple(model.parameters())
        base_gradients = torch.autograd.grad((output.tensor * probe).sum(), args)
        for element in gs.fibergroup.elements:
            # GeometricTensor.transform detaches spatial transforms. Treat the
            # transformed input as an independent leaf, then compare VJPs.
            transformed = raw.transform(element)
            transformed.tensor.requires_grad_()
            transformed_probe = GeometricTensor(probe, output.type).transform(element).tensor
            prediction = model(transformed)
            gradients = torch.autograd.grad((prediction.tensor * transformed_probe).sum(),
                                            (transformed.tensor,) + tuple(model.parameters()))
            expected_input = GeometricTensor(base_gradients[0], ft).transform(element).tensor
            rows.append({"element": str(element), "location": "input_gradient", "lattice_action": True,
                         **error_metrics(gradients[0], expected_input)})
            rows.append({"element": str(element), "location": "parameter_gradient", "lattice_action": True,
                         **error_metrics(torch.cat([g.flatten() for g in gradients[1:]]),
                                         torch.cat([g.flatten() for g in base_gradients[1:]]))})
    else:
        with torch.no_grad():
            torch.manual_seed(9001)
            output = model(raw)
            for element in gs.fibergroup.elements:
                torch.manual_seed(9001)  # same RNG is not a transformed dropout mask
                actual = model(getattr(raw, transform)(element))
                expected = getattr(output, transform)(element)
                rows.append({"element": str(element), "location": "output",
                             "lattice_action": _lattice_element(gs, element),
                             **error_metrics(actual.tensor, expected.tensor)})
    return _summarize(case, rows)


def validate_result(result):
    """Fixed strict tolerance and explicit nonzero diagnostic threshold."""
    if result["diagnostic"]:
        relevant = result["by_element"]
        if result["protocol"] == "nonlattice_attention":
            relevant = [row for row in relevant if not row["lattice_action"]]
        elif result["protocol"] == "full_model":
            relevant = [row for row in relevant if row["location"] != "logits"]
        assert max(row["relative_l2"] for row in relevant) > 1e-3, result["case_id"]
    else:
        assert result["max_relative_l2"] < 1e-4, (result["case_id"], result["max_relative_l2"])
    assert all(np.isfinite(row["relative_l2"]) and np.isfinite(row["max_abs"])
               for row in result["by_element"]), result["case_id"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-configs", nargs="+", choices=MODEL_CONFIGS, default=list(MODEL_CONFIGS))
    parser.add_argument("--protocols", nargs="+", help="Optional subset of protocol names")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = [c for c in cases() if c.model_config in args.model_configs
                and (not args.protocols or c.protocol in args.protocols)]
    if not selected:
        parser.error("No cases selected")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in selected:
        result = run_case(case)
        validate_result(result)
        result["validation"] = "detected_limitation" if case.diagnostic else "within_tolerance"
        rows.append(result)
        print(f"{case.id}: {result['validation']} rel_L2={result['max_relative_l2']:.3e}", flush=True)
        payload = {
            "environment": {"torch": torch.__version__, "escnn": importlib.metadata.version("escnn"),
                            "python": sys.version, "platform": platform.platform(), "device": "cpu", "dtype": "float32",
                            "threads": 1},
            "protocol": {"weight_seed": 1729, "input_seed": 2026, "dropout_seed": 9001,
                         "strict_relative_l2_tolerance": 1e-4, "diagnostic_min_relative_l2": 1e-3,
                         "metric": "||actual-expected||_2 / max(||actual||_2, ||expected||_2, 1e-30)",
                         "spatial_transform": "ESCNN centered action; interpolation for non-lattice rotations",
                         "initialization": "random untrained weights; sparse nonidentity stencil weights randomized; full models fast_init=True"},
            "cases": rows,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
