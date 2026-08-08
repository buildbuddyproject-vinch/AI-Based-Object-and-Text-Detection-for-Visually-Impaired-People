"""
Export YOLOv8 to ONNX for faster CPU inference.
==================================================
Usage:
    python export_onnx.py

Exports weights/yolov8n.pt -> weights/yolov8n.onnx using Ultralytics'
built-in exporter. Ultralytics' YOLO class auto-detects the model
format from its file extension, so switching to the ONNX build is just
a one-line config change afterwards - no code changes needed in
modules/object_detector.py:

    # config.py
    YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "yolov8n.onnx")

ONNX Runtime typically runs meaningfully faster than PyTorch on CPU for
this kind of small detection model, at the cost of losing GPU/CUDA
acceleration if you ever move to a GPU host (stick with the .pt weights
there instead).
"""
import os

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(BASE_DIR, "weights", "yolov8n.pt")


def main():
    if not os.path.exists(PT_PATH):
        print(f"'{PT_PATH}' not found - run `python download_weights.py` first.")
        return

    print(f"Exporting {PT_PATH} to ONNX...")
    model = YOLO(PT_PATH)
    onnx_path = model.export(format="onnx")
    print(f"Done: {onnx_path}")
    print("\nTo use it, set in config.py:")
    print(f'    YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "{os.path.basename(onnx_path)}")')


if __name__ == "__main__":
    main()
