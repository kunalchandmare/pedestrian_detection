"""
Shared detection utilities (PyTorch)
====================================

Common pieces used by both the training script and the self-check benchmark:
  - the BDD100K class list and the pedestrian sub-set,
  - the model builder (a fine-tunable Faster R-CNN / MobileNetV3 detector),
  - model loading,
  - real object-detection metrics (per-class AP, mAP@0.5, precision/recall),
  - the 0-100 Accuracy / Robustness / Efficiency scoring.

The model is a proper multi-object, multi-class detector. For one image it
returns a variable number of detections, each with a box, a class label and a
confidence score - not a single fixed box.
"""

from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from src.benchmark.onnx_yolo_benchmark_adapter import (
    OnnxYoloBenchmarkAdapter,
)
from src.benchmark.pt2e_yolo_benchmark_adapter import (
    PT2EYOLOBenchmarkAdapter,
)
from src.benchmark.yolo_adapter import YoloBenchmarkAdapter, YoloBenchmarkAdapterPortable

# ============================================================================
# CLASSES
# ============================================================================
# BDD100K object classes. Label 0 is reserved for the detector's "background",
# so each class maps to index+1.
CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck",
    "bus", "bike", "motor", "rider", "train",
]
NUM_CLASSES = len(CLASSES) + 1  # + background

CAT_TO_LABEL = {c: i + 1 for i, c in enumerate(CLASSES)}
LABEL_TO_CAT = {i + 1: c for i, c in enumerate(CLASSES)}

# "Pedestrian" classes for the pedestrians-only evaluation toggle.
PEDESTRIAN_CATEGORIES = {"person", "rider", "pedestrian"}
PEDESTRIAN_LABELS = {CAT_TO_LABEL[c] for c in ("person", "rider")}

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


# Default thresholds for the reported precision/recall.
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5


# ============================================================================
# MODEL
# ============================================================================
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Faster R-CNN with a MobileNetV3-Large FPN backbone, retargeted to
    ``num_classes`` (COCO-pretrained backbone, fresh detection head).

    Small ``min_size``/``max_size`` keep it light enough to train on CPU.
    """
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, min_size=320, max_size=640
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detection_model(model_path):
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()

    if suffix == ".pt":
        return YoloBenchmarkAdapter(yolo_weights_path=str(model_path)).eval()

    if suffix == ".onnx":
        return OnnxYoloBenchmarkAdapter(model_path).eval()

    if suffix == ".pt2":
        return PT2EYOLOBenchmarkAdapter(
            pt2_path=str(model_path),
            image_size=640,
            num_classes=len(CLASSES),
        ).eval()

    raise ValueError(f"Unsupported model format: {suffix}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# METRICS  (standard IoU-matched detection metrics)
# ============================================================================
def _average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """VOC-style all-point-interpolation AP from per-prediction TP/FP flags."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detections(predictions, targets, class_filter=None,
                        iou_threshold: float = IOU_THRESHOLD,
                        score_threshold: float = SCORE_THRESHOLD) -> dict:
    """Compute per-class AP, mAP@0.5 and precision/recall over a dataset.

    ``predictions`` / ``targets`` are lists (one entry per image) of dicts with
    numpy/torch arrays:
      pred = {"boxes": [P,4] xyxy, "scores": [P], "labels": [P]}
      gt   = {"boxes": [G,4] xyxy, "labels": [G]}
    Boxes must share the same (pixel) coordinate space.

    ``class_filter`` (a set of label ids, e.g. the pedestrian labels) restricts
    the evaluation to those classes only.
    """
    labels = sorted(class_filter) if class_filter else list(range(1, NUM_CLASSES))

    per_class = {}
    for label in labels:
        scores, is_tp, n_gt = [], [], 0
        tp_at_thr = fp_at_thr = 0

        for pred, gt in zip(predictions, targets):
            p_boxes = np.asarray(pred["boxes"], dtype=np.float32).reshape(-1, 4)
            p_scores = np.asarray(pred["scores"], dtype=np.float32).reshape(-1)
            p_labels = np.asarray(pred["labels"]).reshape(-1)
            g_boxes = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            g_labels = np.asarray(gt["labels"]).reshape(-1)

            pmask = p_labels == label
            gmask = g_labels == label
            pb, ps = p_boxes[pmask], p_scores[pmask]
            gb = g_boxes[gmask]
            n_gt += len(gb)

            if len(pb) == 0:
                continue

            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]

            if len(gb) == 0:
                ious = np.zeros((len(pb), 0), dtype=np.float32)
            else:
                ious = box_iou(torch.from_numpy(pb), torch.from_numpy(gb)).numpy()

            matched = np.zeros(len(gb), dtype=bool)
            for i in range(len(pb)):
                tp = False
                if ious.shape[1] > 0:
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= iou_threshold and not matched[j]:
                        matched[j] = True
                        tp = True
                scores.append(ps[i])
                is_tp.append(tp)
                if ps[i] >= score_threshold:
                    tp_at_thr += int(tp)
                    fp_at_thr += int(not tp)

        ap = _average_precision(np.asarray(scores), np.asarray(is_tp, dtype=bool), n_gt)
        recall = tp_at_thr / n_gt if n_gt else float("nan")
        precision = tp_at_thr / (tp_at_thr + fp_at_thr) if (tp_at_thr + fp_at_thr) else 0.0
        per_class[LABEL_TO_CAT[label]] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "n_gt": int(n_gt),
            "tp": int(tp_at_thr),
            "fp": int(fp_at_thr),
        }

    valid_aps = [m["ap"] for m in per_class.values() if not np.isnan(m["ap"])]
    present = {c: m for c, m in per_class.items() if m["n_gt"] > 0}
    tp_sum = sum(m["tp"] for m in per_class.values())
    fp_sum = sum(m["fp"] for m in per_class.values())
    gt_sum = sum(m["n_gt"] for m in per_class.values())

    return {
        "map50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "precision": tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0,
        "recall": tp_sum / gt_sum if gt_sum else 0.0,
        "per_class": per_class,
        "classes_present": len(present),
        "total_gt": int(gt_sum),
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
    }


def calculate_scores(results: dict, model) -> dict:
    """Combine detection metrics into Accuracy / Robustness / Efficiency (0-100)."""
    # Accuracy: localisation + classification quality (mAP@0.5).
    accuracy = results["map50"] * 100

    # Robustness: recall at the reporting threshold, tempered by how evenly the
    # model performs across classes (consistency of per-class AP).
    aps = [m["ap"] for m in results["per_class"].values()
           if m["n_gt"] > 0 and not np.isnan(m["ap"])]
    consistency = 1.0 - float(np.std(aps)) if aps else 0.5
    robustness = (results["recall"] + min(1.0, max(0.0, consistency))) * 100 / 2

    # Efficiency: smaller models score higher.
    try:
        size_mb = count_params(model) * 4 / (1024 * 1024)
        if size_mb < 10:
            efficiency = 100
        elif size_mb < 50:
            efficiency = 100 - (size_mb - 10) * 1.5
        else:
            efficiency = max(20, 100 - (size_mb - 10))
    except Exception:
        efficiency = 50

    clip = lambda v: round(min(100, max(0, v)), 1)
    accuracy, robustness, efficiency = clip(accuracy), clip(robustness), clip(efficiency)
    return {
        "accuracy": accuracy,
        "robustness": robustness,
        "efficiency": efficiency,
        "overall": round((accuracy + robustness + efficiency) / 3, 1),
    }
