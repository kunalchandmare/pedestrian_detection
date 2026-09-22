from src.quantization.quantize import calibrate, export_yolo_raw_graph


def test_export_yolo_raw_graph(yolo_model):
    try:
        export_yolo_raw_graph(yolo_model)
    except Exception as e:
        assert False, f"Failed to export yolo raw graph: {e}"