# tests/test_benchmark/test_onnx_yolo_benchmark_adapter.py
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from shared.model_helper import export_yolo_fp32_onnx
from src.benchmark.yolo_adapter import YoloBenchmarkAdapterPortable
from tests.conftest import yolo_wt_path
import onnxruntime as ort


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

def compare_all_detections(
    original: dict[str, torch.Tensor],
    onnx: dict[str, torch.Tensor],
    match_iou: float = 0.5,
) -> None:
    """Compare detections returned as {'boxes', 'scores', 'labels'} dictionaries."""

    pt_boxes = original["boxes"].detach().cpu().float()
    pt_scores = original["scores"].detach().cpu().float()
    pt_labels = original["labels"].detach().cpu().long()

    ox_boxes = onnx["boxes"].detach().cpu().float()
    ox_scores = onnx["scores"].detach().cpu().float()
    ox_labels = onnx["labels"].detach().cpu().long()

    print(f"Original: {len(pt_boxes)}, ONNX: {len(ox_boxes)}")

    if not len(pt_boxes) or not len(ox_boxes):
        print("No detections to compare.")
        return

    ious = box_iou(pt_boxes, ox_boxes)
    used_onnx = set()

    for pt_index in range(len(pt_boxes)):
        candidates = [
            ox_index
            for ox_index in range(len(ox_boxes))
            if ox_index not in used_onnx
            and pt_labels[pt_index] == ox_labels[ox_index]
            and ious[pt_index, ox_index] >= match_iou
        ]

        if not candidates:
            print(
                f"Original-only: class={pt_labels[pt_index].item()}, "
                f"score={pt_scores[pt_index].item():.3f}, "
                f"box={pt_boxes[pt_index].tolist()}"
            )
            continue

        ox_index = max(
            candidates,
            key=lambda index: ious[pt_index, index].item(),
        )
        used_onnx.add(ox_index)

        print(
            f"Match: IoU={ious[pt_index, ox_index].item():.3f}, "
            f"class={pt_labels[pt_index].item()}/"
            f"{ox_labels[ox_index].item()}, "
            f"score={pt_scores[pt_index].item():.3f}/"
            f"{ox_scores[ox_index].item():.3f}, "
            f"boxes={pt_boxes[pt_index].tolist()}/"
            f"{ox_boxes[ox_index].tolist()}"
        )

    for ox_index in range(len(ox_boxes)):
        if ox_index not in used_onnx:
            print(
                f"ONNX-only: class={ox_labels[ox_index].item()}, "
                f"score={ox_scores[ox_index].item():.3f}, "
                f"box={ox_boxes[ox_index].tolist()}"
            )

def test_onnx_vs_original_detections(
            yolo_wt_path,
            onnx_fp32_adapter,
            real_image,  # Fixture: ONE real CHW [0,1] benchmark image.
            predict_args,
    ):
        original = YoloBenchmarkAdapterPortable(image_size=predict_args["imgsz"],
                                                conf_threshold=predict_args["conf"],
                                                iou_threshold=predict_args["iou"],
                                                max_det=predict_args["max_det"],
                                                yolo_weights_path=str(yolo_wt_path),).eval()

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

        compare_all_detections(pt, ox)



def print_prediction_diff(pt_result, onnx_result, match_iou=0.5):
    pt_boxes = pt_result.boxes.xyxy.detach().cpu().float()
    pt_scores = pt_result.boxes.conf.detach().cpu().float()
    pt_labels = pt_result.boxes.cls.detach().cpu().long()

    ox_boxes = onnx_result.boxes.xyxy.detach().cpu().float()
    ox_scores = onnx_result.boxes.conf.detach().cpu().float()
    ox_labels = onnx_result.boxes.cls.detach().cpu().long()

    print(f"PT detections:   {len(pt_boxes)}")
    print(f"ONNX detections: {len(ox_boxes)}")

    if not len(pt_boxes) or not len(ox_boxes):
        return

    ious = box_iou(pt_boxes, ox_boxes)
    used_onnx = set()

    for pt_index in range(len(pt_boxes)):
        candidates = [
            ox_index
            for ox_index in range(len(ox_boxes))
            if ox_index not in used_onnx
            and pt_labels[pt_index] == ox_labels[ox_index]
            and ious[pt_index, ox_index] >= match_iou
        ]

        if not candidates:
            print(
                f"PT-only: "
                f"class={pt_labels[pt_index].item() + 1}, "
                f"score={pt_scores[pt_index].item():.4f}, "
                f"box={pt_boxes[pt_index].tolist()}"
            )
            continue

        ox_index = max(
            candidates,
            key=lambda j: ious[pt_index, j].item(),
        )
        used_onnx.add(ox_index)

        score_delta = (
            ox_scores[ox_index] - pt_scores[pt_index]
        ).item()

        box_delta = (
            ox_boxes[ox_index] - pt_boxes[pt_index]
        ).abs().max().item()

        print(
            f"MATCH: "
            f"class={pt_labels[pt_index].item() + 1}/"
            f"{ox_labels[ox_index].item() + 1}, "
            f"IoU={ious[pt_index, ox_index].item():.4f}, "
            f"score="
            f"{pt_scores[pt_index].item():.4f}/"
            f"{ox_scores[ox_index].item():.4f}, "
            f"score_delta={score_delta:+.4f}, "
            f"max_box_delta={box_delta:.2f}px"
        )

    for ox_index in range(len(ox_boxes)):
        if ox_index not in used_onnx:
            print(
                f"ONNX-only: "
                f"class={ox_labels[ox_index].item() + 1}, "
                f"score={ox_scores[ox_index].item():.4f}, "
                f"box={ox_boxes[ox_index].tolist()}"
            )


def test_raw_onnx_vs_original_detections(
        yolo_wt_path,
        onnx_fp32_path,
        real_image,          # Fixture: ONE real CHW [0,1] benchmark image
        predict_args=None,
):
    original  = YOLO(yolo_wt_path)
    onnx_fp32 = YOLO(onnx_fp32_path)

    args = {
        # "imgsz": 640,          # keep commented to match “no imgsz” behaviour
        "conf": 0.1,
        "iou": 0.5,
        "max_det": 300,
        "agnostic_nms": False,
        "rect": False,
        "verbose": False,
        "device": "cpu",
    }

    # Convert CHW float [0,1] → HWC uint8 (what Ultralytics expects)
    source_img = (
        real_image.detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )

    # ------------------------------------------------------------------
    # 1. Normal post-NMS comparison
    # ------------------------------------------------------------------
    pt_result   = original.predict(source=source_img, **args)[0]
    onnx_result = onnx_fp32.predict(source=source_img, **args)[0]

    # ------------------------------------------------------------------
    # 2. RAW network output comparison
    # ------------------------------------------------------------------
    # Reproduce the exact letter-box that Ultralytics just used
    # ------------------------------------------------------------------
    # RAW comparison – completely bypass Ultralytics for ONNX
    # ------------------------------------------------------------------
    letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)
    im = letterbox(image=source_img)
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im_tensor = torch.from_numpy(im).float() / 255.0
    im_np = im_tensor.unsqueeze(0).numpy()  # (1,3,H,W)

    with torch.no_grad():
        # PyTorch raw
        raw_pt = original.model(im_tensor.unsqueeze(0) if im_tensor.ndim == 3 else im_tensor)
        if isinstance(raw_pt, (list, tuple)):
            raw_pt = raw_pt[0]
        raw_pt = raw_pt.cpu().numpy()

    # Pure ONNX Runtime – no Ultralytics wrapper
    sess = ort.InferenceSession(str(onnx_fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    raw_onnx = sess.run(None, {input_name: im_np})[0]

    print("\n===== RAW NETWORK OUTPUT =====")
    print(f"Shape PT     : {raw_pt.shape}")
    print(f"Shape ONNX   : {raw_onnx.shape}")
    print(f"Max  PT      : {raw_pt.max():.6f}")
    print(f"Max  ONNX    : {raw_onnx.max():.6f}")
    print(f"Mean PT      : {raw_pt.mean():.6f}")
    print(f"Mean ONNX    : {raw_onnx.mean():.6f}")
    if raw_pt.shape == raw_onnx.shape:
        print(f"Max abs diff : {np.abs(raw_pt - raw_onnx).max():.6f}")
        print(f"Mean abs diff: {np.abs(raw_pt - raw_onnx).mean():.6f}")
    else:
        print("WARNING: shapes differ – ONNX file still contains NMS")
        print("Output names:", [o.name for o in sess.get_outputs()])
        print("Output shapes:", [o.shape for o in sess.get_outputs()])

    # ------------------------------------------------------------------
    # 3. Your existing post-NMS diff printer
    # ------------------------------------------------------------------
    print_prediction_diff(
        pt_result,
        onnx_result,
        match_iou=0.5,
    )