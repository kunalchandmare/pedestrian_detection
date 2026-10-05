"""
Student Self-Check Benchmark (PyTorch)
======================================

Confirms that your trained detector:
  1. loads correctly,
  2. follows the detection contract (list of images -> list of
     {boxes, labels, scores} dicts),
  3. runs through the full evaluation pipeline without errors,

and shows indicative detection metrics (mAP@0.5, precision, recall). It
evaluates against the local Student_Data subset, so absolute scores are only a
sanity check - the Moderator runs the official benchmark on held-out data.

Run from anywhere with:
    streamlit run Students/testbenchmark/benchmark_app.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
# this file lives in  Students/testbenchmark/  ->  parent.parent is  Students/
STUDENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDENT_DIR))
from shared.detection_utils import (  # noqa: E402
    load_detection_model, evaluate_detections, calculate_scores, count_params,
    category_to_label, PEDESTRIAN_LABELS,
)

DATA_DIR = Path("data/calibration")
MODELS_DIR = Path("results")

RANDOM_SEED = 42
BENCHMARK_EVAL_SIZE = 200      # self-check sample (smaller than the moderator)
INFER_BATCH = 4

def predict_args():
    return {
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.5,
        "max_det": 300,
        "agnostic_nms": False,
        "rect": False,
        "verbose": False,
        "device": "cpu",
    }
# ============================================================================
# DATASET
# ============================================================================
def load_dataset(num_samples: int, seed: int, log=print):
    """Load a deterministic sample of Student_Data as (image_tensor, gt) pairs."""
    annotations_path = DATA_DIR / "annotations.json"
    images_dir = DATA_DIR / "images"
    if not annotations_path.exists() or not images_dir.exists():
        log(f"Dataset not found at {DATA_DIR}")
        return [], []

    with open(annotations_path) as f:
        entries = list(json.load(f).get("images", []))
    np.random.default_rng(seed).shuffle(entries)

    images, targets = [], []
    for e in entries:
        if len(images) >= num_samples:
            break
        img_path = images_dir / e.get("filename", "")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for o in e.get("objects", []):
            b = o.get("box2d")
            lbl = category_to_label(o.get("category"))
            if b and lbl is not None:
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                labels.append(lbl)

        images.append(tensor)
        targets.append({
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, dtype=np.int64).reshape(-1),
        })
        if len(images) % 50 == 0:
            log(f"  loaded {len(images)}/{num_samples} images...")

    log(f"Loaded {len(images)} images from Student_Data")
    return images, targets

def _is_ultralytics_onnx(model):
    """Return True for an Ultralytics YOLO wrapper loaded from ONNX."""
    yolo = getattr(model, "yolo", model)
    model_path = getattr(yolo, "ckpt_path", None)

    return (
        isinstance(model_path, str)
        and model_path.lower().endswith(".onnx")
    )

def run_inference(model, images, log=print):
    """Run the detector and return predictions as numpy dicts."""
    preds = []
    # PyTorch adapters need eval(); Ultralytics ONNX models do not.
    if hasattr(model, "eval") and not _is_ultralytics_onnx(model):
        model.eval()
    with torch.no_grad():
        for start in range(0, len(images), INFER_BATCH):
            batch = images[start:start + INFER_BATCH]
            for out in model(batch):
                preds.append({
                    "boxes": out["boxes"].cpu().numpy().reshape(-1, 4),
                    "scores": out["scores"].cpu().numpy().reshape(-1),
                    "labels": out["labels"].cpu().numpy().reshape(-1),
                })
            if (start + INFER_BATCH) % 50 < INFER_BATCH:
                log(f"  inferred {min(start + INFER_BATCH, len(images))}/{len(images)}...")
    return preds


# ============================================================================
# MODEL DISCOVERY
# ============================================================================
def find_models(base_dir: Path) -> dict:
    models = {}
    if base_dir.exists():
        for root, _dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".pt"):
                    p = Path(root) / f
                    models[os.path.relpath(p, base_dir)] = str(p)
    return models


# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(page_title="Student Self-Check Benchmark", layout="wide")
    st.title("Pedestrian / Object Detection - Student Self-Check")
    st.caption(
        "Confirms your detector loads and runs through the benchmark pipeline. "
        "Scores here are indicative only; the Moderator runs the official benchmark."
    )

    st.sidebar.header("Configuration")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_models = find_models(MODELS_DIR)

    st.sidebar.subheader("1. Select your model")
    if all_models:
        selected = st.sidebar.selectbox("Models found in Students/Models", sorted(all_models))
        model_path = all_models[selected]
    else:
        st.sidebar.warning(f"No .pt models found in {MODELS_DIR}")
        st.sidebar.info("Train a model first (python Students/train_model.py) "
                        "or enter a path below.")
        model_path = st.sidebar.text_input("Model path")

    uploaded = st.sidebar.file_uploader("Or upload a model", type=["pt","pt2", "onnx"])
    if uploaded:
        model_path = str(MODELS_DIR / uploaded.name)
        with open(model_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"Saved {uploaded.name}")

    st.sidebar.subheader("2. Settings")
    pedestrians_only = st.sidebar.checkbox("Evaluate pedestrians only", value=False)
    n_samples = st.sidebar.slider("Images to evaluate", 50, 500, BENCHMARK_EVAL_SIZE, 50)

    run = st.sidebar.button("Run benchmark", use_container_width=True)

    if not run:
        st.info("Select a model in the sidebar and click **Run benchmark**.")
        _show_help()
        return

    if not model_path or not os.path.exists(model_path):
        st.error(f"Model not found: {model_path or '(no path given)'}")
        return

    with st.status("Running benchmark...", expanded=True) as status:
        status.update(label="Loading model...")
        st.write(f"Loading model from `{model_path}`")
        try:
            model = load_detection_model(model_path,predict_args())
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Could not load the model: {e}")
            return
        st.write(f"Model loaded - {count_params(model):,} parameters")

        status.update(label="Loading dataset...")
        images, targets = load_dataset(n_samples, RANDOM_SEED, log=st.write)
        if not images:
            status.update(label="Failed", state="error")
            st.error(f"No images loaded. Make sure Student_Data is at:\n{DATA_DIR}")
            return

        status.update(label="Running inference...")
        try:
            predictions = run_inference(model, images, log=st.write)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error("Benchmark failed")
            st.exception(e)
            return

        status.update(label="Scoring...")
        class_filter = PEDESTRIAN_LABELS if pedestrians_only else None
        results = evaluate_detections(predictions, targets, class_filter=class_filter)
        scores = calculate_scores(results, model)
        status.update(label="Benchmark complete", state="complete")

    st.success("Benchmark complete - your model is compatible with the pipeline.")
    _render_results(results, scores, len(images))


def _render_results(results, scores, n_images):
    st.header("Detection metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mAP@0.5", f"{results['map50']*100:.2f}%")
    c2.metric("Precision", f"{results['precision']*100:.2f}%")
    c3.metric("Recall", f"{results['recall']*100:.2f}%")
    c4.metric("Classes present", results["classes_present"])

    st.header("Scores")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Accuracy", f"{scores['accuracy']:.1f}/100")
    s2.metric("Robustness", f"{scores['robustness']:.1f}/100")
    s3.metric("Efficiency", f"{scores['efficiency']:.1f}/100")
    s4.metric("Overall", f"{scores['overall']:.1f}/100")

    st.header("Per-class results")
    rows = [
        {
            "Class": name,
            "AP@0.5": "n/a" if np.isnan(m["ap"]) else f"{m['ap']*100:.2f}%",
            "Precision": "n/a" if np.isnan(m["recall"]) else f"{m['precision']*100:.2f}%",
            "Recall": "n/a" if np.isnan(m["recall"]) else f"{m['recall']*100:.2f}%",
            "GT instances": m["n_gt"],
            "TP": m["tp"], "FP": m["fp"],
        }
        for name, m in results["per_class"].items() if m["n_gt"] > 0
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No labelled objects in the sampled data for the selected classes.")

    st.header("Dataset")
    st.write(f"Evaluated **{n_images}** images from `{DATA_DIR}` "
             f"(IoU {results['iou_threshold']}, score >= {results['score_threshold']})")

    st.download_button(
        "Download results (JSON)",
        data=json.dumps(results, indent=2, default=float),
        file_name=f"benchmark_results_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )


def _show_help():
    with st.expander("Model requirements"):
        st.markdown(
            """
            A detector that follows the standard torchvision contract:
            - **Input:** a list of image tensors `(3, H, W)`, RGB, float in `[0, 1]`
            - **Output:** a list (one per image) of dicts
              `{"boxes": [M,4] xyxy, "labels": [M], "scores": [M]}`
            - **Saved as:** `model.pt` (the training script saves the whole model)

            **Object classes (BDD100K):** car, traffic sign, traffic light,
            person, truck, bus, bike, motor, rider, train.
            """
        )


if __name__ == "__main__":
    main()
