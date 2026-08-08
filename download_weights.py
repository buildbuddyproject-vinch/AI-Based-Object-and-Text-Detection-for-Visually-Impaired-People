"""
Convenience script to download the YOLOv8n pretrained weights into the
weights/ directory (~6 MB).

Usage:
    python download_weights.py

If this script fails (e.g. no internet access from the runtime
environment), download yolov8n.pt manually from:
    https://github.com/ultralytics/assets/releases
and place it at:  weights/yolov8n.pt
"""
import os
import shutil

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
TARGET_PATH = os.path.join(WEIGHTS_DIR, "yolov8n.pt")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if os.path.exists(TARGET_PATH):
        print(f"Weights already present at {TARGET_PATH}")
        return

    print("Downloading yolov8n.pt via Ultralytics (this triggers an "
          "automatic download on first load)...")
    model = YOLO("yolov8n.pt")
    downloaded_path = getattr(model, "ckpt_path", None) or "yolov8n.pt"

    if os.path.exists(downloaded_path) and os.path.abspath(downloaded_path) != os.path.abspath(TARGET_PATH):
        shutil.move(downloaded_path, TARGET_PATH)
        print(f"Moved weights to {TARGET_PATH}")
    elif os.path.exists(TARGET_PATH):
        print(f"Weights already at {TARGET_PATH}")
    else:
        print("Download finished but the file location is unclear - check "
              "the current directory for 'yolov8n.pt' and move it into "
              f"'{WEIGHTS_DIR}' manually.")


if __name__ == "__main__":
    main()
