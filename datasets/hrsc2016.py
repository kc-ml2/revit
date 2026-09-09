"""HRSC2016 dataset for oriented ship detection."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Official Level-1 protocol treats every ship as one class.
HRSC_SHIP_CLASS_IDS = {
    "000001",
    "100000001",
    "100000002",
    "100000003",
    "100000004",
    "100000005",
    "100000006",
    "100000007",
    "100000008",
    "100000009",
    "100000010",
    "100000011",
    "100000012",
    "100000013",
    "100000015",
    "100000016",
    "100000017",
    "100000018",
    "100000019",
    "100000020",
    "100000022",
    "100000024",
    "100000025",
    "100000026",
    "100000027",
    "100000028",
    "100000029",
    "100000030",
    "100000032",
}


def _resolve_root(root: str) -> str:
    """Accept either .../HRSC2016 or .../HRSC2016/HRSC2016."""
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "AllImages")) and os.path.isdir(os.path.join(root, "Annotations")):
        return root
    nested = os.path.join(root, "HRSC2016")
    if os.path.isdir(os.path.join(nested, "AllImages")):
        return nested
    raise FileNotFoundError(
        f"Could not find AllImages/Annotations under {root}. "
        "Expected data/HRSC2016/HRSC2016 from the downloaded archive."
    )


def _parse_annotation(xml_path: str, single_class: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return boxes (N, 5) as cx,cy,w,h,angle_rad and labels (N,)."""
    tree = ET.parse(xml_path)
    boxes: List[List[float]] = []
    labels: List[int] = []
    class_to_idx = {cid: i for i, cid in enumerate(sorted(HRSC_SHIP_CLASS_IDS))}

    for obj in tree.findall(".//HRSC_Object"):
        class_id = (obj.findtext("Class_ID") or "").strip()
        if class_id not in HRSC_SHIP_CLASS_IDS:
            continue
        try:
            cx = float(obj.findtext("mbox_cx"))
            cy = float(obj.findtext("mbox_cy"))
            w = float(obj.findtext("mbox_w"))
            h = float(obj.findtext("mbox_h"))
            ang = float(obj.findtext("mbox_ang"))
        except (TypeError, ValueError):
            continue
        if w <= 1 or h <= 1:
            continue
        boxes.append([cx, cy, w, h, ang])
        labels.append(0 if single_class else class_to_idx[class_id])

    if not boxes:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def letterbox_image_and_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    size: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, np.ndarray, float, Tuple[int, int]]:
    """Resize with aspect ratio, pad to square. Boxes are cx,cy,w,h,angle."""
    w0, h0 = image.size
    scale = min(size / w0, size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    image = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(image, (pad_x, pad_y))

    if boxes.size:
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] * scale + pad_x
        out[:, 1] = boxes[:, 1] * scale + pad_y
        out[:, 2] = boxes[:, 2] * scale
        out[:, 3] = boxes[:, 3] * scale
        # angle unchanged under isotropic scale + translation
    else:
        out = boxes
    return canvas, out, scale, (pad_x, pad_y)


def hflip_image_and_boxes(image: Image.Image, boxes: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes.size:
        w, _ = image.size
        out = boxes.copy()
        out[:, 0] = w - 1 - boxes[:, 0]
        out[:, 4] = -boxes[:, 4]
        boxes = out
    return image, boxes


class HRSC2016Dataset(Dataset):
    """
    HRSC2016 oriented ship detection.

    Returns:
        image: FloatTensor (3, S, S)
        target: dict with
            boxes  — FloatTensor (N, 5) cx, cy, w, h, angle_rad
            labels — LongTensor (N,)
            image_id — str
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 800,
        single_class: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        self.root = _resolve_root(root)
        self.split = split
        self.image_size = image_size
        self.single_class = single_class
        self.hflip_prob = hflip_prob if split == "train" else 0.0
        self.train = split == "train"

        split_file = os.path.join(self.root, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(split_file)
        with open(split_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        jitter = (
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            if (self.train and color_jitter)
            else None
        )
        tfms: List[Callable] = [transforms.ToTensor()]
        if normalize:
            tfms.append(
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )
        self.to_tensor = transforms.Compose(tfms)
        self.jitter = jitter

        self.num_classes = 1 if single_class else len(HRSC_SHIP_CLASS_IDS)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        img_path = os.path.join(self.root, "AllImages", f"{image_id}.bmp")
        xml_path = os.path.join(self.root, "Annotations", f"{image_id}.xml")
        if not os.path.isfile(img_path):
            # some mirrors use .jpg
            alt = os.path.join(self.root, "AllImages", f"{image_id}.jpg")
            if os.path.isfile(alt):
                img_path = alt
            else:
                raise FileNotFoundError(img_path)

        image = Image.open(img_path).convert("RGB")
        boxes, labels = _parse_annotation(xml_path, single_class=self.single_class)
        image, boxes, _, _ = letterbox_image_and_boxes(image, boxes, self.image_size)

        if self.hflip_prob > 0 and torch.rand(1).item() < self.hflip_prob:
            image, boxes = hflip_image_and_boxes(image, boxes)

        if self.jitter is not None:
            image = self.jitter(image)

        image_t = self.to_tensor(image)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": image_id,
        }
        return image_t, target


def hrsc_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def get_hrsc2016_loaders(
    root: str,
    batch_size: int = 2,
    image_size: int = 800,
    num_workers: int = 4,
    single_class: bool = True,
    hflip_prob: float = 0.5,
):
    train_set = HRSC2016Dataset(
        root,
        split="train",
        image_size=image_size,
        single_class=single_class,
        hflip_prob=hflip_prob,
    )
    val_set = HRSC2016Dataset(
        root,
        split="val",
        image_size=image_size,
        single_class=single_class,
        hflip_prob=0.0,
        color_jitter=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=hrsc_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=hrsc_collate_fn,
    )
    return train_loader, val_loader, train_set.num_classes
