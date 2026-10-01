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
    """Same benchmark-facing interface for both FP32 and INT8 ONNX."""

    def __init__(self, model_path: Path,
        image_size: int = 640,
        num_classes: int = 10,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    @torch.inference_mode()
    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        if not isinstance(images, list):
            raise TypeError("Expected a list of CHW image tensors")

        results = []
        for image in images:  # Exported ONNX model uses batch size 1.
            original_hw = tuple(image.shape[-2:])
            x = preprocess_yolo_image(image)

            if tuple(x.shape) != (1, 3, self.image_size, self.image_size):
                raise ValueError(f"Unexpected preprocessed shape: {tuple(x.shape)}")

            input_array = np.ascontiguousarray(
                x.detach().cpu().numpy(),
                dtype=np.float32,
            )
            raw_outputs = self.session.run(
                None,
                {self.input_name: input_array},
            )
            prediction = torch.from_numpy(raw_outputs[0])

            if prediction.ndim != 3 or prediction.shape[:2] != (
                1, 4 + self.num_classes
            ):
                raise ValueError(
                    f"Expected raw [1, {4 + self.num_classes}, N], "
                    f"got {tuple(prediction.shape)}. "
                    "Check the YOLO export output format."
                )

            detections = non_max_suppression(
                prediction,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                max_det=self.max_det,
                nc=self.num_classes,
                end2end=False,
            )[0]

            boxes = ops.scale_boxes(
                x.shape[2:],
                detections[:, :4].clone(),
                original_hw,
            )
            results.append({
                "boxes": boxes.float(),
                "scores": detections[:, 4].float(),
                "labels": detections[:, 5].long() + 1,
            })

        return results