from pathlib import Path

import torch
from ultralytics import YOLO
import onnxruntime as ort

from src.benchmark.pt2e_yolo_benchmark_adapter import PT2EYOLOBenchmarkAdapter


def load_model(model_dir:str):
    """
    Loads a YOLO model from the specified directory. The function checks if the
    provided path points to a valid file and raises an error if the file is not
    found. If the model file exists, it initializes the YOLO model, converts it
    to float type, moves it to the CPU, and sets it to evaluation mode before
    returning the model.

    Args:
        model_dir: A string representing the directory or file path to the YOLO
                   model weights.

    Raises:
        FileNotFoundError: If the provided model path does not point to a valid file.

    Returns:
        The YOLO model in evaluation mode, loaded with the weights found at the
        specified path.
    """

    model_path = Path(model_dir)
    if model_path.is_file():
        yolo_model = YOLO(model_dir)
    else:
        raise FileNotFoundError(f"YOLO weights not found: {model_dir}")

    raw_model = yolo_model.model.float().cpu().eval()
    return raw_model

def save_pt2e_model(model_int8, example_inputs, out_path):
    """
    model_int8 + example input (1, 3, 640, 640)
                  │
                  ▼
    torch.export.export(...)
                  │
                  ▼
    ExportedProgram: graph + state + input constraints

    Saves a PyTorch model in PT2E export format to a specified path. It exports
    the provided quantized model using the Torch.Export module. The function
    ensures that the target directory exists before saving.

    Parameters:
    model_int8 : torch.nn.Module
        The quantized PyTorch model to be exported.
    example_inputs : tuple or torch.Tensor
        Example inputs to use for tracing the model during the export process.
        If a single tensor is provided, it will be automatically converted
        into a tuple.
    path : str
        The file path where the exported model will be saved.

    Returns:
    str : The path where the model has been saved.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    #The example tensor is not your calibration dataset, and exporting does not recalculate the Q/DQ scales.
    # Those were determined earlier when you ran calibration and convert_pt2e().
    # Here you are re-exporting the already converted model so torch.export.save() can serialize it
    inputs = (
        example_inputs
        if isinstance(example_inputs, tuple)
        else (example_inputs,)
    )

    exported_program = torch.export.export(model_int8, inputs)
    torch.export.save(exported_program, str(out_path))
    return out_path

def save_pt_yolo_adaptor(pt2_dir,pt_ot_dir):
    adapter = PT2EYOLOBenchmarkAdapter(
        pt2_dir,
        image_size=640,
        num_classes=10,
    ).eval()

    print(PT2EYOLOBenchmarkAdapter.__module__)
    # Must print: benchmark.pt2e_yolo_benchmark_adapter
    # Must NOT print: __main__ or main

    torch.save(adapter, pt_ot_dir)

WEIGHTS = Path("results/checkpoint/yolo/model.pt")
FP32_ONNX = Path("results/onnx/yolo_fp32.onnx")
INT8_ONNX = Path("results/onnx/yolo_int8.onnx")
IMAGE_SIZE = 640
NUM_CLASSES = 10

def export_yolo_fp32_onnx(yolo_weight_path, out_fp32_onnx, img_size=640) -> Path:
    """Stage 2a: export the original raw YOLO model, without embedded NMS."""
    out_fp32_onnx.parent.mkdir(parents=True, exist_ok=True)

    exported_path = Path(
        YOLO(str(yolo_weight_path)).export(
            format="onnx",
            imgsz=img_size,
            batch=1,
            dynamic=False,
            nms=None,
        )
    )

    if exported_path.resolve() != out_fp32_onnx.resolve():
        out_fp32_onnx.write_bytes(exported_path.read_bytes())

    return out_fp32_onnx



