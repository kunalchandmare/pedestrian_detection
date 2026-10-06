from pathlib import Path

import onnx
import torch
from torch import Tensor, nn
from ultralytics import YOLO
import numpy as np
import onnxruntime as ort
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


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
        self.model_path = Path(model_path).expanduser().resolve()

        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"Expected an ONNX file, got: {self.model_path}"
            )

        self.yolo = YOLO(
            str(self.model_path),
            task="detect",
        )

    @torch.inference_mode()
    def forward(
            self,
            images: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        if not isinstance(images, list):
            raise TypeError("Expected a list of CHW image tensors")

        results = []

        for image in images:
            image = image.detach().cpu().float()

            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"Expected image shape (3,H,W), got {tuple(image.shape)}"
                )

            if image.numel() == 0:
                raise ValueError("Image tensor is empty")

            if not torch.isfinite(image).all():
                raise ValueError("Image contains NaN or Inf")

            if image.min().item() < 0.0 or image.max().item() > 1.0:
                raise ValueError("Expected image values in [0,1]")

            source_img = (
                image.permute(1, 2, 0)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .numpy()
            )

            result = self.yolo.predict(
                source=source_img,
                imgsz=self.image_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                agnostic_nms=False,
                rect=False,
                verbose=False,
                device="cpu",
                max_det=self.max_det,
            )[0]

            if result.boxes is None or len(result.boxes) == 0:
                results.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                })
                continue

            results.append({
                "boxes": result.boxes.xyxy.detach().cpu().float().reshape(-1, 4),
                "scores": result.boxes.conf.detach().cpu().float().reshape(-1),
                "labels": (
                        result.boxes.cls.detach().cpu().long() + 1
                ).reshape(-1),
            })

        return results

    def print_onnx_nodes(self) -> None:
        """
        Print all nodes in the ONNX graph.

        Useful for inspecting the FP32 or INT8 model and identifying
        detection-head nodes to exclude from quantization.
        """
        model = onnx.load(str(self.model_path))

        print(f"Model: {self.model_path}")
        print(f"Number of nodes: {len(model.graph.node)}")

        for index, node in enumerate(model.graph.node):
            print(
                f"{index:04d} | "
                f"name={node.name!r} | "
                f"type={node.op_type}"
            )
