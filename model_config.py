"""Regular-field counts, block depths and heads used by ImageNet REViT."""

MODEL_PRESETS = {
    "tiny": {"dims": (12, 24, 48, 96), "depths": (1, 2, 3, 1), "heads": (1, 2, 4, 8)},
    "small": {"dims": (24, 48, 96, 192), "depths": (1, 2, 4, 1), "heads": (1, 2, 4, 8)},
    "base": {"dims": (32, 64, 128, 256), "depths": (2, 2, 4, 2), "heads": (1, 2, 4, 8)},
    "large": {"dims": (64, 128, 192, 384), "depths": (2, 2, 6, 2), "heads": (2, 4, 8, 16)},
}
