import pytest

from src.analysis.model_stats import params_stats


def test_param_stats(yolo_model):
    model_stats = params_stats(yolo_model, (1,3,640,640))

    assert len(model_stats)==9
    assert isinstance(model_stats[0], tuple)