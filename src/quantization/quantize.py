import torch
import torchao
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

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
    prepared_model.eval()

    for batch_index, batch in enumerate(calibration_loader):
        if batch_index >= max_batches:
            break

        # Supports loaders yielding images only or (images, targets, ...)
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(DEVICE, dtype=torch.float32)

        # Ensure loader preprocessing already produces [B, 3, 640, 640].
        prepared_model(images)

    print(f"Calibrated using {min(batch_index + 1, max_batches)} batches.")