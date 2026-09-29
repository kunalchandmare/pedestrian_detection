from src.quantization.quantize import calibrate, export_yolo_raw_graph, quantize_yolo_int8


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
        quantize_yolo_int8(yolo_model, calibration_loader,batch_size=1, max_calibration_batches=1,image_size=640)
    except Exception as e:
        assert False, f"Failed to Quantize model to INT8 : {e}"
