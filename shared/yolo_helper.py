from pathlib import Path

from ultralytics import YOLO

def load_model(model_dir:str):

    model_path = Path(model_dir)

    if model_path.is_file():
        yolo_model = YOLO(model_dir)
    else:
        raise FileNotFoundError(f"YOLO weights not found: {model_dir}")

    raw_model = yolo_model.model.float().cpu().eval()
    return raw_model