from shared.model_helper import export_yolo_fp32_onnx


def test_onnx_export(yolo_wt_path,onnx_fp32_path):

    export_yolo_fp32_onnx(yolo_wt_path,onnx_fp32_path,img_size=[640, 640])

    assert onnx_fp32_path.exists()
    assert onnx_fp32_path.is_file()