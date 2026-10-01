# tests/test_benchmark/test_onnx_yolo_benchmark_adapter.py
import numpy as np
import torch
from torchvision.ops import box_iou

from shared.model_helper import export_yolo_fp32_onnx
from src.benchmark.yolo_adapter import YoloBenchmarkAdapterPortable
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

def test_fp32_onnx_matches_pytorch(
    yolo_model, onnx_fp32_adapter, example_batch_input
):
    x = example_batch_input[0]  # Tensor of shape (1, 3, 640, 640)

    with torch.inference_mode():
        pytorch_output = yolo_model(x)

    if isinstance(pytorch_output, (tuple, list)):
        pytorch_output = pytorch_output[0]

    onnx_output = onnx_fp32_adapter.session.run(
        None,
        {onnx_fp32_adapter.input_name: x.numpy()},
    )[0]

    torch.testing.assert_close(
        pytorch_output.cpu(),
        torch.from_numpy(onnx_output),
        rtol=1e-3,
        atol=1e-3,
    )

def test_onnx_vs_original_detections(
            yolo_wt_path,
            onnx_fp32_adapter,
            real_image,  # Fixture: ONE real CHW [0,1] benchmark image.
    ):
        original = YoloBenchmarkAdapterPortable(str(yolo_wt_path)).eval()

        with torch.inference_mode():
            pt = original([real_image])[0]
            ox = onnx_fp32_adapter([real_image])[0]

        print(f"\nOriginal detections: {len(pt['scores'])}")
        print(f"ONNX detections:     {len(ox['scores'])}")

        # Compare the highest-confidence detection from each adapter.
        assert len(pt["scores"]) > 0, "Choose an image with a detection"
        assert len(ox["scores"]) > 0, "ONNX found no detections"

        pt_top = pt["scores"].argmax().item()
        ox_top = ox["scores"].argmax().item()

        iou = box_iou(
            pt["boxes"][pt_top: pt_top + 1],
            ox["boxes"][ox_top: ox_top + 1],
        )[0, 0].item()

        print(f"Top-box IoU: {iou:.3f}")
        print(f"Top labels:  original={pt['labels'][pt_top].item()}, "
              f"ONNX={ox['labels'][ox_top].item()}")
        print(f"Top scores:  original={pt['scores'][pt_top].item():.3f}, "
              f"ONNX={ox['scores'][ox_top].item():.3f}")

        assert pt["labels"][pt_top] == ox["labels"][ox_top]
        assert iou > 0.9