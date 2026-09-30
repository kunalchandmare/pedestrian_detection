import os
import json
import sys
from pathlib import Path

import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics.data.augment import LetterBox

CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck",
    "bus", "bike", "motor", "rider", "train",
]
NUM_CLASSES = len(CLASSES) + 1  # + background

CAT_TO_LABEL = {c: i + 1 for i, c in enumerate(CLASSES)}
LABEL_TO_CAT = {i + 1: c for i, c in enumerate(CLASSES)}

# Different BDD100K label releases use different category names. The det_v2
# detection format calls these classes "pedestrian"/"bicycle"/"motorcycle";
# map them onto our canonical names so any release works.
CATEGORY_ALIASES = {
    "pedestrian": "person",
    "bicycle": "bike",
    "motorcycle": "motor",
}


def category_to_label(category):
    """Integer label for a category name (resolving aliases), or None if unknown."""
    return CAT_TO_LABEL.get(CATEGORY_ALIASES.get(category, category))

# ============================================================================
# STEP 1: DATASET
# ============================================================================
class BDDDetectionDataset(Dataset):
    """Reads Student_Data (images/ + annotations.json) as a detection dataset.

    Each item is (image_tensor, target) where target has every annotated object
    in the image as boxes [M,4] (xyxy, pixel coords) and integer class labels.

    Parameters
    ----------
    augment : bool
        When True, applies random horizontal flip and light colour jitter at
        load time.  Set to True for the training split, False for validation.
    annotations_file : str
        Filename of the annotations JSON inside data_dir.  Defaults to
        ``annotations.json`` (the Student_Data layout).  Override to use a
        pre-split file such as ``annotations_train.json``.
    images_dir : Path | None
        Explicit path to the images folder.  When ``None`` (default) the
        loader uses ``data_dir / "images"`` — the original behaviour.
        Pass ``IMAGES_DIR`` to decouple annotation files from image storage.
    _samples : list | None
        Internal: pass a pre-built sample list to avoid re-parsing the JSON
        (used when constructing separate train / val dataset instances).
    """

    def __init__(self, data_dir: Path, max_images: int = 0):
        self.images_dir = Path(data_dir) / "images"
        with open(Path(data_dir) / "annotations.json") as f:
            entries = json.load(f).get("images", [])

        self.samples = []
        for e in entries:
            boxes, labels = [], []
            for o in e.get("objects", []):
                b = o.get("box2d")
                lbl = category_to_label(o.get("category"))
                if not b or lbl is None:
                    continue
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                labels.append(lbl)
            if boxes and (self.images_dir / e["filename"]).exists():
                self.samples.append((e["filename"], boxes, labels))
            if max_images and len(self.samples) >= max_images:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, boxes, labels = self.samples[idx]
        img = cv2.imread(str(self.images_dir / filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_loader(data_dir, batch_size=10, workers=0):

    data_path = Path(data_dir)

    data_set = BDDDetectionDataset(data_path)

    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate_fn)

    return train_loader


def preprocess_yolo_image(image: torch.Tensor) -> torch.Tensor:
    """CHW RGB float [0, 1] -> BCHW RGB float [0, 1], letterboxed to 640."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}")

    # Undo the dataset loader's /255 and RGB conversion for LetterBox.
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    bgr_u8 = np.ascontiguousarray(rgb_u8[:, :, ::-1])

    padded_bgr = LetterBox(
        new_shape=(640, 640),
        auto=False,
    )(image=bgr_u8)

    rgb_chw = np.ascontiguousarray(
        padded_bgr[:, :, ::-1].transpose(2, 0, 1)
    )
    return torch.from_numpy(rgb_chw).unsqueeze(0).float() / 255.0