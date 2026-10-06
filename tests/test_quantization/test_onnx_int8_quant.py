from pathlib import Path

import numpy as np
import onnx
import torch
import onnxruntime as ort
from onnx import numpy_helper

from shared.data_loader import preprocess_yolo_image
from shared.model_helper import export_yolo_fp32_onnx
from src.quantization.onnx_int8_quant import (
    YoloCalibrationReader,
    calibrate,
    quantize_onnx,
)


def test_onnx_export(yolo_wt_path,onnx_fp32_path):

    export_yolo_fp32_onnx(yolo_wt_path,onnx_fp32_path,img_size=[640, 640])

    assert onnx_fp32_path.exists()
    assert onnx_fp32_path.is_file()

def test_calibrate(calibration_data_path):
    images = list(calibrate(calibration_data_path))

    assert len(images) == 200

    for image in images:
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.shape[0] == 3
        assert image.dtype == torch.float32
        assert torch.isfinite(image).all()
        assert image.min().item() >= 0.0
        assert image.max().item() <= 1.0


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.input = FakeInput()

    def get_inputs(self):
        return [self.input]

def test_yolo_calibration_reader(
    onnx_fp32_path,
    calibration_data_path,
):
    images = list(
        calibrate(
            calibration_data_path,
            max_images=3,
        )
    )

    reader = YoloCalibrationReader(
        img_size=640,
        model_path=onnx_fp32_path,
        images=images,
    )

    assert reader.img_size == 640
    assert reader.input_name == "images"

    for _ in range(3):
        result = reader.get_next()

        assert result is not None
        assert set(result) == {"images"}

        batch = result["images"]

        assert isinstance(batch, np.ndarray)
        assert batch.dtype == np.float32
        assert batch.shape == (1, 3, 640, 640)
        assert batch.flags["C_CONTIGUOUS"]
        assert np.isfinite(batch).all()
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    assert reader.get_next() is None



def test_quantize_onnx(
    onnx_fp32_path,
    onnx_int8_path,
    calibration_data_path,
    nodes_to_exclude,
    ops_to_exclude,
):
    int8_path = quantize_onnx(
        onnx_fp32_path,
        onnx_int8_path,
        calibration_data_path,
        exclude_nodes=nodes_to_exclude,
        exclude_ops=ops_to_exclude,
    )

    assert int8_path.resolve() == onnx_int8_path.resolve()
    assert int8_path.exists()
    assert int8_path.is_file()
    assert int8_path.stat().st_size > 0


def test_excluded_detection_nodes_are_not_quantized(
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
nodes_to_exclude):
    fp32_model = onnx.load(str(onnx_fp32_path))
    int8_model = onnx.load(str(onnx_int8_path))

    fp32_nodes = {
        node.name: node
        for node in fp32_model.graph.node
    }

    int8_nodes = {
        node.name: node
        for node in int8_model.graph.node
    }

    initializers = {
        item.name: item
        for item in int8_model.graph.initializer
    }

    for name in nodes_to_exclude:
        assert name in fp32_nodes
        assert name in int8_nodes

        expected_type = fp32_nodes[name].op_type
        actual_type = int8_nodes[name].op_type

        print(
            f"{name}: expected={expected_type}, "
            f"actual={actual_type}"
        )

        assert actual_type == expected_type, (
            f"{name} changed from {expected_type} "
            f"to {actual_type}"
        )

        if expected_type == "Conv":
            weight_name = int8_nodes[name].input[1]
            assert weight_name in initializers

            weight = numpy_helper.to_array(
                initializers[weight_name]
            )

            assert weight.dtype in (np.float16, np.float32), (
                f"{name} weight dtype is {weight.dtype}"
            )

def test_excluded_ops_are_not_quantized(
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    ops_to_exclude
):
    fp32_model = onnx.load(str(onnx_fp32_path))
    int8_model = onnx.load(str(onnx_int8_path))

    fp32_nodes = list(fp32_model.graph.node)
    int8_nodes = list(int8_model.graph.node)

    assert fp32_nodes, "FP32 graph contains no nodes"
    assert int8_nodes, "INT8 graph contains no nodes"

    fp32_op_counts = {}

    for node in fp32_nodes:
        fp32_op_counts[node.op_type] = (
                fp32_op_counts.get(node.op_type, 0) + 1
        )

    int8_op_counts = {}

    for node in int8_nodes:
        int8_op_counts[node.op_type] = (
                int8_op_counts.get(node.op_type, 0) + 1
        )

    quantize_count = int8_op_counts.get(
        "QuantizeLinear",
        0,
    )

    dequantize_count = int8_op_counts.get(
        "DequantizeLinear",
        0,
    )

    print("QuantizeLinear:", quantize_count)
    print("DequantizeLinear:", dequantize_count)

    assert quantize_count > 0
    assert dequantize_count > 0

    for op_type in ops_to_exclude:
        fp32_count = fp32_op_counts.get(op_type, 0)
        int8_count = int8_op_counts.get(op_type, 0)

        print(
            f"{op_type}: "
            f"FP32={fp32_count}, "
            f"INT8={int8_count}"
        )

        assert fp32_count > 0, (
            f"{op_type} does not exist in the FP32 graph"
        )

        assert int8_count == fp32_count, (
            f"{op_type} count changed: "
            f"FP32={fp32_count}, "
            f"INT8={int8_count}"
        )

def test_quantized_int8_model(onnx_int8_path):

    session = ort.InferenceSession(
        str(onnx_int8_path),
        providers=["CPUExecutionProvider"],
    )

    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) >= 1

    print(session.get_inputs()[0].name)
    print(session.get_outputs()[0].shape)

def test_int8_raw_output(
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    real_image: torch.Tensor,
):
    fp32_session = ort.InferenceSession(
        str(onnx_fp32_path),
        providers=["CPUExecutionProvider"],
    )

    int8_session = ort.InferenceSession(
        str(onnx_int8_path),
        providers=["CPUExecutionProvider"],
    )

    fp32_input = fp32_session.get_inputs()[0]
    int8_input = int8_session.get_inputs()[0]

    assert fp32_input.name == int8_input.name
    assert fp32_input.type == int8_input.type

    x = preprocess_yolo_image(
        real_image,
    )

    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert x.min().item() >= 0.0
    assert x.max().item() <= 1.0

    x_np = np.ascontiguousarray(
        x.detach().cpu().numpy(),
        dtype=np.float32,
    )

    fp32_raw = fp32_session.run(
        None,
        {fp32_input.name: x_np},
    )[0]

    int8_raw = int8_session.run(
        None,
        {int8_input.name: x_np},
    )[0]

    assert isinstance(fp32_raw, np.ndarray)
    assert isinstance(int8_raw, np.ndarray)
    assert fp32_raw.shape == int8_raw.shape
    assert fp32_raw.ndim == 3
    assert fp32_raw.shape[1] == 14

    assert np.isfinite(fp32_raw).all()
    assert np.isfinite(int8_raw).all()

    mae = float(np.mean(np.abs(fp32_raw - int8_raw)))
    max_error = float(np.max(np.abs(fp32_raw - int8_raw)))

    print(f"FP32 raw range: {fp32_raw.min()} .. {fp32_raw.max()}")
    print(f"INT8 raw range: {int8_raw.min()} .. {int8_raw.max()}")
    print(f"Raw MAE: {mae}")
    print(f"Raw max error: {max_error}")

    # Prevent a completely collapsed INT8 model.
    assert np.ptp(int8_raw) > 0.0
    assert np.max(np.abs(int8_raw)) > 1e-6

def test_int8_output_error_by_channel(
    onnx_fp32_path,
    onnx_int8_path,
    real_image,
):

    fp32_session = ort.InferenceSession(
        str(onnx_fp32_path),
        providers=["CPUExecutionProvider"],
    )
    int8_session = ort.InferenceSession(
        str(onnx_int8_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = fp32_session.get_inputs()[0].name

    x = preprocess_yolo_image(
        real_image
    ).detach().cpu().numpy().astype(np.float32)

    fp32 = fp32_session.run(None, {input_name: x})[0]
    int8 = int8_session.run(None, {input_name: x})[0]

    error = np.abs(fp32 - int8)

    channel_mae = error.mean(axis=(0, 2))
    channel_max = error.max(axis=(0, 2))

    print("Channel MAE:", channel_mae)
    print("Channel max error:", channel_max)

    assert fp32.shape == int8.shape
    assert fp32.shape[1] == 14

def test_int8_decoded_predictions(
    onnx_fp32_adapter,
    onnx_int8_adapter,
    real_image
):
    fp32 = onnx_fp32_adapter([real_image])[0]
    int8 = onnx_int8_adapter([real_image])[0]

    print("FP32 detections:", len(fp32["boxes"]))
    print("INT8 detections:", len(int8["boxes"]))

    #onnx_fp32_adapter.print_onnx_nodes()

    if len(fp32["scores"]):
        print("FP32 top scores:", fp32["scores"][:10])

    if len(int8["scores"]):
        print("INT8 top scores:", int8["scores"][:10])

    assert len(int8["boxes"]) == len(int8["scores"])
    assert len(int8["boxes"]) == len(int8["labels"])

import torch


def test_yolo_predict_fp32_vs_int8(
    onnx_fp32_adapter,
    onnx_int8_adapter,
    real_image,
):
    source_img = (
        real_image.detach()
        .cpu()
        .float()
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )

    predict_args = {
        "imgsz": onnx_fp32_adapter.image_size,
        "source": source_img,
        "conf": onnx_fp32_adapter.conf_threshold,
        "iou": onnx_fp32_adapter.iou_threshold,
        "agnostic_nms": False,
        "rect": False,
        "verbose": False,
        "device": "cpu",
        "max_det": onnx_fp32_adapter.max_det,
    }

    fp32_result = onnx_fp32_adapter.yolo.predict(
        **predict_args,
    )[0]

    int8_result = onnx_int8_adapter.yolo.predict(
        **predict_args,
    )[0]

    fp32_count = (
        0
        if fp32_result.boxes is None
        else len(fp32_result.boxes)
    )

    int8_count = (
        0
        if int8_result.boxes is None
        else len(int8_result.boxes)
    )

    print("FP32 detections:", fp32_count)
    print("INT8 detections:", int8_count)

    if fp32_count:
        print(
            "FP32 top scores:",
            fp32_result.boxes.conf.detach().cpu(),
        )
        print(
            "FP32 labels:",
            fp32_result.boxes.cls.detach().cpu(),
        )
        print(
            "FP32 boxes:",
            fp32_result.boxes.xyxy.detach().cpu(),
        )

    if int8_count:
        print(
            "INT8 top scores:",
            int8_result.boxes.conf.detach().cpu(),
        )
        print(
            "INT8 labels:",
            int8_result.boxes.cls.detach().cpu(),
        )
        print(
            "INT8 boxes:",
            int8_result.boxes.xyxy.detach().cpu(),
        )

    assert fp32_count > 0, "FP32 produced no detections"
    assert int8_count > 0, (
        "INT8 produced no detections with adapter.yolo.predict"
    )