from pathlib import Path

import pytest
import torch

from shared.model_helper import load_model
from shared.data_loader import get_loader, BDDDetectionDataset
from src.benchmark.onnx_yolo_benchmark_adapter import OnnxYoloBenchmarkAdapter
from src.benchmark.pt2e_yolo_benchmark_adapter import PT2EYOLOBenchmarkAdapter
from src.benchmark.yolo_adapter import YoloBenchmarkAdapter


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

@pytest.fixture(scope="module")
def calibration_loader():
    calibration_dir = "data/calibration"
    return get_loader(data_dir=calibration_dir, batch_size=10)

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
def onnx_fp32_adapter(onnx_fp32_path):
    assert onnx_fp32_path.is_file(), (
        f"ONNX model not found: {onnx_fp32_path.resolve()}"
    )
    return OnnxYoloBenchmarkAdapter(onnx_fp32_path).eval()