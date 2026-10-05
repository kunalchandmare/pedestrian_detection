from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from shared.data_loader import preprocess_yolo_image


class OnnxYoloBenchmarkAdapter(nn.Module):
    """Benchmark-facing adapter for FP32 or INT8 Ultralytics ONNX models."""

    def __init__(
        self,
        model_path: Path,
        image_size: int = 640,
        num_classes: int = 10,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_det: int = 300,
    ) -> None:
        super().__init__()

        self.model_path = Path(model_path)
        self.image_size = image_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"Expected an ONNX file, got: {self.model_path}"
            )

        # Ultralytics handles ONNX Runtime, preprocessing, NMS,
        # and scaling boxes back to the original image.
        self.yolo = YOLO(str(self.model_path))

    @torch.inference_mode()
    def forward(
        self,
        images: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        if not isinstance(images, list):
            raise TypeError("Expected a list of CHW image tensors")

        results = []

        for image in images:
            if not isinstance(image, Tensor):
                raise TypeError(
                    f"Expected torch.Tensor, got {type(image).__name__}"
                )

            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    "Expected image shape (3,H,W), "
                    f"got {tuple(image.shape)}"
                )

            image = image.detach().cpu().float()

            if image.numel() == 0:
                raise ValueError("Image tensor is empty")

            if image.min().item() < 0.0 or image.max().item() > 1.0:
                raise ValueError("Expected image values in [0,1]")

            # Ultralytics can apply its own resize/letterbox preprocessing.
            source_img = (
                image.permute(1, 2, 0)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .numpy()
            )

            result = self.yolo.predict(
                #imgsz=self.image_size,
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

            results.append({
                "boxes": boxes_xyxy.reshape(-1, 4),
                "scores": scores.reshape(-1),
                "labels": labels_1_to_10.reshape(-1),
            })

        return results
