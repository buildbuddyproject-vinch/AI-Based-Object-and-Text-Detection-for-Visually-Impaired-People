"""
Object Detector Module
========================
Wraps a YOLOv8 (Ultralytics) model to perform real-time object detection
on video frames and draw bounding boxes with confidence labels.
"""
import os

import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="weights/yolov8n.pt", conf_threshold=0.45, device="cpu"):
        if not os.path.exists(model_path):
            # Fall back to the bare filename so Ultralytics can
            # auto-download the pretrained checkpoint on first use
            # (see download_weights.py for a cleaner alternative).
            model_path = os.path.basename(model_path)

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = self.model.names

    def _label_for(self, cls_id):
        if isinstance(self.class_names, dict):
            return self.class_names.get(cls_id, str(cls_id))
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    def detect(self, frame):
        """Run inference on a single BGR frame.

        Returns a list of dicts:
            {"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "label": self._label_for(cls_id),
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                })
        return detections

    @staticmethod
    def draw_boxes(frame, detections):
        """Draw bounding boxes + label/confidence text onto `frame`
        in-place and return it. Static because it's pure drawing logic
        (no model weights involved) - callers can use it without
        needing any particular detector loaded, e.g. auto-assistance
        mode drawing boxes from whichever domain model is active
        without also having to load the generic COCO model."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (66, 135, 245)  # accessible blue, BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = det["label"]
            if det.get("color"):
                label_text = f"{det['color']} {label_text}"
            text = f"{label_text} {det['confidence'] * 100:.0f}%"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(0, y1 - th - 10)
            cv2.rectangle(frame, (x1, label_y), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
