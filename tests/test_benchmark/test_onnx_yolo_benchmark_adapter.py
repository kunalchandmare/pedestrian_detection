# tests/test_benchmark/test_onnx_yolo_benchmark_adapter.py

import torch

from shared.model_helper import export_yolo_fp32_onnx
from tests.conftest import yolo_wt_path


def test_onnx_export(onnx_fp32_path,yolo_wt_path,):
    try:
        export_yolo_fp32_onnx(yolo_weight_path=yolo_wt_path,out_fp32_onnx=onnx_fp32_path,img_size=640)
    except Exception as e:
        assert False, f"Failed to export ONNX: {e}"


def test_adapter_loads_real_onnx_and_runs_640_image(onnx_fp32_adapter, image):
    """Proves the real ONNX file loads and runs through the adapter."""
    with torch.inference_mode():
        detections = onnx_fp32_adapter([image])

    assert detections is not None
    assert isinstance(detections, list)
    assert len(detections) == 1


def test_adapter_detection_format(onnx_fp32_adapter, image):
    """Proves output has the format expected by the benchmark."""
    with torch.inference_mode():
        detections = onnx_fp32_adapter([image])

    result = detections[0]
    assert isinstance(result, dict)
    assert {"boxes", "scores", "labels"} <= result.keys()

    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    assert isinstance(boxes, torch.Tensor)
    assert isinstance(scores, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 1
    assert labels.ndim == 1
    assert boxes.shape[0] == scores.shape[0] == labels.shape[0]
    assert boxes.shape[0] <= 300

    # No detections from random input is acceptable.
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()
    assert torch.isfinite(labels.float()).all()


def test_adapter_uses_one_based_class_labels(onnx_adapter, image):
    """Matches your benchmark's 1–10 class-ID convention."""
    with torch.inference_mode():
        result = onnx_adapter([image])[0]

    labels = result["labels"]
    if labels.numel():
        assert (labels >= 1).all()
        assert (labels <= 10).all()