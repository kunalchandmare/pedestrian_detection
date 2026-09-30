import torch


def test_adapter_loads_real_pt2_and_runs_640_image(pt2e_adapter, image):
    """Proves the adapter works end-to-end with the saved model."""
    with torch.inference_mode():
        detections = pt2e_adapter([image])

    assert detections is not None
    assert isinstance(detections, list)
    assert len(detections) == 1  # One output per image in the batch.


def test_adapter_detection_format(pt2e_adapter, image):
    """Proves output is usable by the benchmark, not merely non-empty."""
    with torch.inference_mode():
        detections = pt2e_adapter([image])

    result = detections[0]
    assert isinstance(result, dict)

    # Adjust these keys ONLY if your benchmark specifies different names.
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

    # Random noise may produce zero detections; that's okay.
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()