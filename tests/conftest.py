import pytest
import torch

from shared.model_helper import load_model
from shared.data_loader import get_loader


@pytest.fixture(scope="module")
def yolo_model():
    m_path = "results/checkpoint/yolo/epoch52_main.pt"
    return load_model(m_path)

@pytest.fixture(scope="module")
def example_input():
    example_inputs = (
        torch.zeros(
            (1, 3, 640, 640),
            dtype=torch.float32,
        ),
    )

@pytest.fixture(scope="module")
def calibration_loader():
    calibration_dir = "data/calibration"
    return get_loader(data_dir=calibration_dir, batch_size=1)