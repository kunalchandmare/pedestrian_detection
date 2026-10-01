from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from onnxruntime.quantization import QuantType, QuantFormat, quantize_static, CalibrationDataReader
import onnxruntime as ort
from torch import Tensor

from shared.data_loader import get_loader, preprocess_yolo_image
from shared.model_helper import export_yolo_fp32_onnx


class YoloCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        img_size:int,
        model_path: Path,
        images: Iterable[Tensor],
    ):
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.img_size = img_size
        self.input_name = session.get_inputs()[0].name
        self.images = iter(images)

    def get_next(self):
        image = next(self.images, None)
        if image is None:
            return None

        x = preprocess_yolo_image(image)
        if tuple(x.shape) != (1, 3, self.img_size, self.img_size):
            raise ValueError(f"Bad calibration shape: {tuple(x.shape)}")

        return {
            self.input_name: np.ascontiguousarray(
                x.detach().cpu().numpy(),
                dtype=np.float32,
            )
        }


def calibration_images(
    data_dir: str | Path,
    max_images: int = 200,
) -> Iterable[Tensor]:
    loader = get_loader(
        data_dir=data_dir,
        batch_size=10,
        workers=0,
    )

    yielded = 0

    for batch in loader:
        images, _targets = batch  # Adjust if your collate_fn returns another shape.

        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError(
                    f"Expected batched NCHW images, got {tuple(images.shape)}"
                )
            individual_images = images.unbind(0)

        elif isinstance(images, (list, tuple)):
            individual_images = images

        else:
            raise TypeError(f"Unexpected images type: {type(images)!r}")

        for image in individual_images:
            if not isinstance(image, torch.Tensor) or image.ndim != 3:
                raise ValueError(
                    "Each calibration image must be a CHW torch.Tensor"
                )

            yield image  # The reader applies preprocess_yolo_image(image).
            yielded += 1

            if yielded >= max_images:
                return

def quantize_onnx(
    fp32_path: Path,
    int8_path: Path,
    calib_img_path: Path,
    img_size=640,
) -> Path:
    """Stage 3: static INT8 PTQ with representative images."""
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    calibrate_imgs = calibration_images(calib_img_path)

    reader = YoloCalibrationReader(model_path=fp32_path, img_size=img_size, images=calibrate_imgs)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return int8_path

if __name__ == "__main__":

    yolo_pt = Path("results/checkpoint/yolo/epoch52_main.pt")
    onnx_out = Path("results/onnx/yolo_fp32.onnx")
    onnx_int8 = Path("results/onnx/yolo_int8.onnx")
    calib_img = Path("data/calibration/images")

    print("2. Export and validate FP32 ONNX")
    fp32_path = export_yolo_fp32_onnx(img_size=640,yolo_weight_path=yolo_pt,out_fp32_onnx=onnx_out,)

    print("3. Calibrate and quantize ONNX")
    int8_path = quantize_onnx(
        fp32_path,
        onnx_int8,
        calib_img
    )
