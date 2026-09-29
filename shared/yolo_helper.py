from pathlib import Path

from ultralytics import YOLO


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