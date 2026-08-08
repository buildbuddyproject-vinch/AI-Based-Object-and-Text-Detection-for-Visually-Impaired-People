"""
Navigation Module
===================
Turns raw object detections into simple spatial guidance: which third of
the frame an obstacle is in (Left / Center / Right) and a rough
"near / far" estimate from bounding-box size, then composes a short
spoken instruction such as "Person ahead. Move slightly right."

NOTE: This is a heuristic, monocular approach (no true depth sensor) - it
is meant as an accessible academic prototype, not a certified mobility
aid. For production-grade distance estimates, pair this with a stereo
camera or a depth sensor (e.g. Intel RealSense, LiDAR).
"""


class NavigationAssistant:
    def __init__(self, near_area_ratio=0.18):
        # bbox_area / frame_area above this ratio is treated as "close".
        self.near_area_ratio = near_area_ratio

    @staticmethod
    def _zone(cx, frame_width):
        third = frame_width / 3
        if cx < third:
            return "left"
        if cx > 2 * third:
            return "right"
        return "center"

    def analyze(self, detections, frame_width, frame_height):
        """Return (instruction: str, zone_map: dict) describing the most
        relevant (largest / closest) obstacle currently in view."""
        if not detections:
            return "Path looks clear.", {"left": [], "center": [], "right": []}

        frame_area = max(1, frame_width * frame_height)
        enriched = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            area = max(0, x2 - x1) * max(0, y2 - y1)
            enriched.append({
                **det,
                "zone": self._zone(cx, frame_width),
                "area_ratio": area / frame_area,
            })

        # The closest / most prominent obstacle is approximated by the
        # largest bounding-box area (bigger on screen => nearer).
        primary = max(enriched, key=lambda d: d["area_ratio"])
        label = primary["label"]
        zone = primary["zone"]
        is_close = primary["area_ratio"] >= self.near_area_ratio

        if zone == "center":
            instruction = f"{label.capitalize()} ahead."
            if is_close:
                left_count = sum(1 for d in enriched if d["zone"] == "left")
                right_count = sum(1 for d in enriched if d["zone"] == "right")
                move_dir = "left" if left_count <= right_count else "right"
                instruction = f"{label.capitalize()} ahead. Move slightly {move_dir}."
        elif zone == "left":
            instruction = f"{label.capitalize()} on your left."
        else:
            instruction = f"{label.capitalize()} on your right."

        zone_map = {"left": [], "center": [], "right": []}
        for d in enriched:
            zone_map[d["zone"]].append(d["label"])

        return instruction, zone_map
