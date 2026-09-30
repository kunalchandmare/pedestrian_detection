import torch
import torch.nn.functional as F
import torchao
import copy
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torchao.quantization.pt2e import move_exported_model_to_eval

from shared.data_loader import preprocess_yolo_image

'''
This script is adapted from the official PyTorch documentation:
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html'''

print(torch.__version__)
print(torchao.__version__)
print("cuda:", torch.cuda.is_available())

# ---------------- Configuration ----------------
WEIGHTS = "epoch52_main.pt"       # Or your trained: "runs/detect/train/weights/best.pt"
IMAGE_SIZE = 640
NUM_CALIBRATION_BATCHES = 100
DEVICE = "cpu"

@torch.inference_mode()
def calibrate(prepared_model, calibration_loader, max_batches):
    images_seen = 0
    total = len(calibration_loader)
    device = next(prepared_model.parameters()).device
    for batch_index, batch in enumerate(calibration_loader): #, total=total, desc="Calibrating")):

        if max_batches is not None and batch_index >= max_batches:
            break

        # Supports loaders yielding images only or (images, targets, ...)
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        if not all(isinstance(image, torch.Tensor) for image in images):
            raise TypeError("Expected a tuple of image tensors")

        for image in images:
            x = preprocess_yolo_image(image).to(device)

            assert x.shape == (1, 3, 640, 640), x.shape
            prepared_model(x)
            images_seen += 1

        if images_seen == 0:
            raise ValueError("No images were used for calibration")

    print(f"Calibrated on {images_seen} images")



def export_yolo_raw_graph(
    yolo_model,
    image_size: int = 640,
    batch_size: int = 1,
):
    """
    Export the raw neural network inside an already-loaded Ultralytics YOLO object.

    Args:
        yolo_model: An already-loaded ultralytics.YOLO object.
        image_size: Fixed height and width for the exported graph.
        batch_size: Fixed batch size for the first export.

    Returns:
        exported_program: The captured torch.export graph.
        graph_model: Callable nn.Module reconstructed from that graph.
        fp32_model: The original raw YOLO nn.Module in eval mode.
        example_inputs: Tuple containing input tensor [B, 3, H, W].
    """
    fp32_model = copy.deepcopy(yolo_model)

    example_inputs = (
        torch.zeros(
            (batch_size, 3, image_size, image_size),
            dtype=torch.float32,
        ),
    )

    exported_program = torch.export.export(
        fp32_model,
        example_inputs,
    )

    graph_model = exported_program.module()

    return exported_program, graph_model, fp32_model, example_inputs

def quantize_yolo_int8(
    yolo_model,
    calibration_loader,
    image_size: int = 640,
    batch_size: int = 1,
    max_calibration_batches: int = 100,
):
    """
    Create a uniform static INT8 PTQ version of a loaded Ultralytics YOLO model.

    Args:
        yolo_model: Loaded ultralytics.YOLO object.
        calibration_loader: Representative data for your existing calibrate function.
        image_size: Fixed model input height and width.
        batch_size: Fixed exported batch size.
        max_calibration_batches: Number of calibration batches to use.

    Returns:
        model_int8: Quantized raw YOLO graph model.
        fp32_model: Original raw FP32 YOLO neural network.
        example_inputs: Example input tuple used during export.
    """
    # 1. Extract raw YOLO nn.Module and export its operator graph.
    _, graph_model, fp32_model, example_inputs = export_yolo_raw_graph(
        yolo_model=yolo_model,
        image_size=image_size,
        batch_size=batch_size,
    )

    # 2. Specify the uniform x86 static-INT8 recipe.
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config()
    )

    # 3. Insert observers and fold eligible Conv-BN patterns.
    prepared_model = prepare_pt2e(graph_model, quantizer)
    move_exported_model_to_eval(prepared_model)

    # 4. Your calibration loop forwards representative image batches
    #    through prepared_model, allowing observers to collect ranges.
    calibrate(
        prepared_model=prepared_model,
        calibration_loader=calibration_loader,
        max_batches=max_calibration_batches,
    )

    # 5. Replace observers with quantize/dequantize operations.
    model_int8 = convert_pt2e(prepared_model)

    return model_int8, fp32_model, example_inputs