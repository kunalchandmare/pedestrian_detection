from src.quantization.quantize import calibrate, export_yolo_raw_graph


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
    pass