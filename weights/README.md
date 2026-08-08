# weights/

Place YOLOv8 model weights here. The app expects:

```
weights/yolov8n.pt
```

## How to get it

**Option 1 - automatic script (recommended)**
```bash
python download_weights.py
```

**Option 2 - let the app auto-download it**
Just run `python app.py`. If `weights/yolov8n.pt` is missing, the
`ObjectDetector` falls back to asking Ultralytics to auto-download
`yolov8n.pt` into the project root on first use. Move it into `weights/`
afterwards to keep the folder structure clean.

**Option 3 - manual download**
Download directly from the official Ultralytics release assets:
https://github.com/ultralytics/assets/releases

Look for `yolov8n.pt` (nano - fastest, ~6 MB) and save it as
`weights/yolov8n.pt`.

## Using a bigger/more accurate model

Ultralytics ships several sizes: `yolov8n.pt` (nano), `yolov8s.pt`
(small), `yolov8m.pt` (medium), `yolov8l.pt` (large), `yolov8x.pt`
(extra-large). Larger models are more accurate but slower - `yolov8n.pt`
is recommended for real-time CPU inference on a typical laptop. To
switch, download the desired file into this folder and update
`YOLO_WEIGHTS_PATH` in `config.py`.
