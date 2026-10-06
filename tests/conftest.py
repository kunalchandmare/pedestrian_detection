from pathlib import Path

import pytest
import torch

from shared.model_helper import load_model
from shared.data_loader import get_loader, BDDDetectionDataset
from src.benchmark.onnx_yolo_benchmark_adapter import OnnxYoloBenchmarkAdapter
from src.benchmark.pt2e_yolo_benchmark_adapter import PT2EYOLOBenchmarkAdapter


@pytest.fixture(scope="module")
def yolo_wt_path():
    m_path = "results/checkpoint/yolo/epoch52_main.pt"
    return Path(m_path)


@pytest.fixture(scope="module")
def data_dir_path():
    m_path = "data/calibration"
    return Path(m_path)


@pytest.fixture(scope="module")
def yolo_model(yolo_wt_path):
    return load_model(str(yolo_wt_path))


@pytest.fixture(scope="module")
def image():
    return torch.rand(3, 640, 640, dtype=torch.float32)


@pytest.fixture(scope="module")
def real_image(data_dir_path):
    dataset = BDDDetectionDataset(data_dir_path, max_images=1)

    assert len(dataset) == 1, "No annotated image found in BDD_DATA_DIR"

    img, _target = dataset[0]
    assert img.dtype == torch.float32
    assert img.ndim == 3 and img.shape[0] == 3  # CHW
    assert 0.0 <= img.min().item() <= img.max().item() <= 1.0

    return img


@pytest.fixture(scope="module")
def example_batch_input():
    return (
        torch.zeros(
            (1, 3, 640, 640),
            dtype=torch.float32,
        ),
    )


@pytest.fixture(scope="session")
def calibration_data_path():
    path = Path("data/calibration")
    return path


@pytest.fixture(scope="module")
def calibration_loader(calibration_data_path):
    return get_loader(data_dir=calibration_data_path, batch_size=10)


@pytest.fixture(scope="module")
def predict_args():
    return {
        "imgsz": 640,
        "conf": 0.1,
        "iou": 0.5,
        "max_det": 300,
        "verbose": False,
        "device": "cpu",
    }


@pytest.fixture(scope="module")
def pt2e_adapter():
    model_int8_dir = "results/checkpoint/pt2e/torch_x86_int8.pt2"
    adapter = PT2EYOLOBenchmarkAdapter(pt2_path=model_int8_dir)
    return adapter


@pytest.fixture(scope="session")
def onnx_fp32_path():
    path = Path("results/onnx/yolo_fp32.onnx")
    return path


@pytest.fixture(scope="session")
def onnx_int8_path():
    path = Path("results/onnx/yolo_int8.onnx")
    return path


@pytest.fixture(scope="module")
def onnx_fp32_adapter(onnx_fp32_path, predict_args):
    assert onnx_fp32_path.is_file(), (
        f"ONNX model not found: {onnx_fp32_path.resolve()}"
    )
    return OnnxYoloBenchmarkAdapter(onnx_fp32_path,
                                    conf_threshold=predict_args["conf"],
                                    iou_threshold=predict_args["iou"],
                                    image_size=predict_args["imgsz"])


@pytest.fixture(scope="module")
def onnx_int8_adapter(onnx_int8_path, predict_args):
    assert onnx_int8_path.is_file(), (
        f"ONNX model not found: {onnx_int8_path.resolve()}"
    )
    return OnnxYoloBenchmarkAdapter(onnx_int8_path,
                                    conf_threshold=predict_args["conf"],
                                    iou_threshold=predict_args["iou"],
                                    image_size=predict_args["imgsz"])


@pytest.fixture(scope="session")
def nodes_to_exclude():
    return [
        "node_Conv_2080",
        "node_Split_1712",
        "node_matmul",
        "node_softmax",
        "node_matmul_1",
        "node_Conv_2159",
        "node_Split_1855",
        "node_matmul_2",
        "node_softmax_1",
        "node_matmul_3",
        "node_Conv_2131",
        "node_Conv_2174",
        "node_Conv_2189",
        "node_Conv_2216",
        "node_Conv_2231",
    ]

@pytest.fixture(scope="session")
def ops_to_exclude():
    return [
    "Conv",
    "Sigmoid",
    "Softmax",
    "MatMul",
    "Split",
    "Reshape",
    "Concat",
    "Mul",
    "Add",
    "Sub",
    "Div",
]
