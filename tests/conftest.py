import pytest
import torch

from shared.yolo_helper import load_model


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