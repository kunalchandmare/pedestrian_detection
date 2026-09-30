import pytest
import torch

from shared.model_helper import load_model
from shared.data_loader import get_loader
from src.benchmark.pt2e_yolo_benchmark_adapter import PT2EYOLOBenchmarkAdapter


@pytest.fixture(scope="module")
def yolo_model():
    m_path = "results/checkpoint/yolo/epoch52_main.pt"
    return load_model(m_path)

@pytest.fixture(scope="module")
def image():
    return torch.rand(3, 640, 640, dtype=torch.float32)

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