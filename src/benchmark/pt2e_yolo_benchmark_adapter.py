"""Benchmark adapter for a saved PT2E-converted YOLO raw model.

Inputs: list of CHW RGB torch tensors (values in [0,1] or [0,255]), or
        list of HWC BGR uint8 numpy images. The list contract matches
        a benchmark that calls model(images) and expects per-image dicts.
Output: list[{'boxes': xyxy tensor, 'scores': tensor, 'labels': int64 tensor}].

The PT2E artifact is a Q/DQ graph, not an Inductor-compiled model or a
checkpoint loadable by ultralytics.YOLO(...). Requires ultralytics, torch,
torchao, numpy, and OpenCV. This code assumes the first raw model output
is a YOLO detection tensor [1, 4+nc, N] with xywh boxes; validate it
against the original FP32 model/predictor before reporting mAP.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from ultralytics.utils.nms import non_max_suppression

from shared.data_loader import preprocess_yolo_image


class PT2EYOLOBenchmarkAdapter(nn.Module):
    def __init__(
        self,
        pt2_path: str | Path,
        image_size: int = 640,
        num_classes: int = 10,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.artifact = Path(pt2_path).read_bytes()
        self.image_size = image_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        # Bypass nn.Module registration: calling adapter.eval() must not call
        # eval() on an exported GraphModule (which can reject train/eval).
        object.__setattr__(self, "_raw_model", None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_raw_model"] = None
        return state

    def _load_raw(self):
        if self._raw_model is None:
            import torchao.quantization.pt2e  # Register custom PT2E operators.

            model = torch.export.load(io.BytesIO(self.artifact)).module()
            object.__setattr__(self, "_raw_model", model)
        return self._raw_model

    @staticmethod
    def _preprocess(image):
        original_hw = tuple(image.shape[-2:])  # Your dataset supplies CHW tensors.
        x = preprocess_yolo_image(image)
        return x, original_hw

    @torch.inference_mode()
    def forward(self, images):
        if not isinstance(images, (list, tuple)):
            raise TypeError("Expected a list of images, as in the benchmark")
        raw_model = self._load_raw()
        results = []
        for img in images:  # exported graph specializes batch size 1
            x, original_hw = self._preprocess(image=img)
            out = raw_model(x)
            prediction = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
                raise RuntimeError("Expected raw predictions [1, 4+nc, N]")
            if prediction.shape[0] != 1 or prediction.shape[1] != self.num_classes + 4:
                raise RuntimeError(
                    f"Expected [1, {self.num_classes + 4}, N], got {tuple(prediction.shape)}"
                )
            detections = non_max_suppression(
                prediction,
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                max_det=self.max_det,
                nc=self.num_classes,
                end2end=False,
            )[0]
            boxes = ops.scale_boxes(x.shape[2:], detections[:, :4].clone(), original_hw)
            results.append({
                "boxes": boxes,
                "scores": detections[:, 4].clone(),
                "labels": detections[:, 5].long()+1,
            })
        return results


if __name__ == "__main__":
    # Run from the repository root. Adjust the PT2 artifact path as needed.
    adapter = PT2EYOLOBenchmarkAdapter("results/checkpoint/pt2e/torch_x86_int8.pt2",
                                       conf_threshold=0.5,
                                       iou_threshold=0.5)
    adapter.eval()
    sample = torch.zeros((3, 640, 640), dtype=torch.float32)
    output = adapter([sample])
    assert len(output) == 1 and set(output[0]) == {"boxes", "scores", "labels"}
    assert output[0]["boxes"].shape[1] == 4
    print({key: tuple(val.shape) for key, val in output[0].items()})
    torch.save(adapter, "results/benchmark/torch_x86_int8_benchmark.pt")
    # For your own trusted artifact, if your PyTorch version defaults to
    # weights_only=True: torch.load('model_int8_benchmark.pt', weights_only=False)
    print("Wrote model_int8_benchmark.pt")
