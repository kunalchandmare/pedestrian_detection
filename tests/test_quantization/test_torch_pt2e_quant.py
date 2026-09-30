import torch

from src.analysis.model_stats import quant_inspect_model, inference_time, check_outputs, inference_raw_out
from src.quantization.torch_pt2e_quant import calibrate, export_yolo_raw_graph, quantize_yolo_int8


def test_export_yolo_raw_graph(yolo_model):
    try:
        export_yolo_raw_graph(yolo_model)
    except Exception as e:
        assert False, f"Failed to export yolo raw graph: {e}"

def test_calibrate_one_batch(yolo_model, calibration_loader):
    try:
        calibrate(yolo_model,calibration_loader,max_batches=1)
    except Exception as e:
        assert False, f"Failed to Calibrate using Calibration Data: {e}"


def test_quantize_yolo_int8(yolo_model, calibration_loader):
    try:
         model_int8,fp32_model, example_inputs = quantize_yolo_int8(yolo_model, calibration_loader,batch_size=1, max_calibration_batches=1,image_size=640)
    except Exception as e:
        assert False, f"Failed to Quantize model to INT8 : {e}"

    quant_inspect_model("FP32 model", fp32_model)
    quant_inspect_model("Converted PT2E model", model_int8)

    assert isinstance(example_inputs, tuple)
    assert isinstance(example_inputs[0], torch.Tensor)
    assert tuple(example_inputs[0].shape) == (1, 3, 640, 640)

    x = example_inputs[0]

    raw_fp32 = inference_raw_out(fp32_model, x)
    raw_int8 = inference_raw_out(model_int8, x)

    assert isinstance(raw_fp32, tuple)
    assert isinstance(raw_int8, tuple)

    # Checking raw output box, score prediction tensor shape
    for branch in ("one2many", "one2one"):
        for item in ("boxes", "scores"):
            fp32_shape = raw_fp32[1][branch][item].shape
            int8_shape = raw_int8[1][branch][item].shape

            assert fp32_shape == int8_shape, (
                f"{branch} {item}: {fp32_shape} != {int8_shape}"
            )
            print(f"{branch} {item}: {tuple(fp32_shape)} ✓")
