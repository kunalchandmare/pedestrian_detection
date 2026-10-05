"""YOLO adapter to match the benchmark detection contract.

This module exposes a torch.nn.Module that accepts the benchmark input format:
    list[Tensor(3,H,W), RGB float in [0,1]]
and returns per-image dicts with:
    boxes [N,4] xyxy, labels [N] (1..10), scores [N]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor, nn
from ultralytics import YOLO


class YoloBenchmarkAdapter(nn.Module):
    """Wrap Ultralytics YOLO with the benchmark-compatible model interface."""

    def __init__(
        self,
        yolo_weights_path: str,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_det: int = 300,
        image_size: int = 640,
    ) -> None:
        super().__init__()
        self.yolo_weights_path = str(Path(yolo_weights_path).expanduser().resolve())
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.image_size = int(image_size)
        self._yolo = None

    def _ensure_yolo(self) -> YOLO:
        if self._yolo is None:
            self._yolo = YOLO(self.yolo_weights_path)
        return self._yolo

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_yolo"] = None
        return state

    def _predict_single(self, image: Tensor) -> Dict[str, Tensor]:
        yolo = self._ensure_yolo()

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected image tensor shape (3,H,W), got {tuple(image.shape)}"
            )

        img = image.detach().to(dtype=torch.float32, device="cpu")
        if img.min().item() < 0.0 or img.max().item() > 1.0:
            raise ValueError("Expected image values in [0,1].")

        # Convert CHW float [0,1] RGB tensor to HWC uint8 RGB image so
        # Ultralytics can apply its own resize/letterbox preprocessing.
        source_img = (
            image.permute(1, 2, 0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        result = yolo.predict(
            imgsz=self.image_size,
            source=source_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            agnostic_nms=False,
            rect=False,
            verbose=False,
            device="cpu",
            max_det=self.max_det,
        )[0]

        boxes_xyxy = result.boxes.xyxy.to(dtype=torch.float32, device="cpu")
        scores = result.boxes.conf.to(dtype=torch.float32, device="cpu")
        labels_1_to_10 = (result.boxes.cls.to(dtype=torch.int64, device="cpu") + 1)

        if boxes_xyxy.numel() == 0:
            boxes_xyxy = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)
            labels_1_to_10 = torch.empty((0,), dtype=torch.int64)

        return {
            "boxes": boxes_xyxy.reshape(-1, 4),
            "scores": scores.reshape(-1),
            "labels": labels_1_to_10.reshape(-1),
        }

    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:
        if not isinstance(images, list):
            raise TypeError("Expected input to be a list of image tensors.")
        return [self._predict_single(image) for image in images]

YoloBenchmarkAdapterPortable = YoloBenchmarkAdapter