# dataset/

Optional data used to extend the bonus features or retrain models.

- `currency/` - reference images for the heuristic currency detector
  (see `dataset/currency/README.md`). Not required for the core
  object detection / OCR / navigation features.
- If you want to fine-tune YOLOv8 on custom classes (e.g. specific
  indoor obstacles, braille signage, etc.), organize your labeled data
  here following the standard Ultralytics YOLO dataset format
  (`images/`, `labels/`, and a `data.yaml`), then train with:

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

and point `YOLO_WEIGHTS_PATH` in `config.py` at the resulting
`runs/detect/train/weights/best.pt`.
