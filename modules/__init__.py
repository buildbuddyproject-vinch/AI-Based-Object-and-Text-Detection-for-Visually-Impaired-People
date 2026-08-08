"""AI/vision/speech modules for the AI Blind Assistant.

Each module is a self-contained, independently testable unit:
    - object_detector : YOLOv8 real-time object detection
    - ocr_reader       : EasyOCR printed-text extraction
    - speaker          : offline pyttsx3 text-to-speech queue
    - navigation       : left/center/right obstacle guidance heuristics
    - voice_commands   : microphone-driven voice command listener
    - color_detector   : dominant color estimation (bonus)
    - currency_detector: heuristic currency note recognition (bonus)
    - qr_reader        : QR code detection/decoding (bonus)
"""
