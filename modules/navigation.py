"""
Navigation Module
===================
Turns raw object detections into simple spatial guidance: which third of
the frame an obstacle is in (Left / Center / Right) and a rough
"near / far" estimate, then composes a short spoken instruction such as
"Person ahead. Move slightly right."

Distance estimation has two modes:
  - Default: a bounding-box-size heuristic (bigger on screen => nearer).
  - Enhanced: if a MiDaS relative-depth map is supplied (see
    modules/depth_estimator.py), each object's average depth is ranked
    against the *rest of the current frame's* depth distribution to
    produce a "very close" / "nearby" / "farther away" label.

NOTE: This is a heuristic, monocular approach (no true depth sensor, and
no camera calibration) - it is meant as an accessible academic
prototype, not a certified mobility aid. Even with the depth map, this
deliberately never states a distance in metres: MiDaS produces
*relative* depth (useful for ranking "nearer vs farther" within one
frame), not a calibrated measurement, and reporting a fake number would
be actively misleading for a blind user. For production-grade distance
estimates, pair this with a stereo camera or a depth sensor (e.g. Intel
RealSense, LiDAR).
"""
import numpy as np


class NavigationAssistant:
    def __init__(self, near_area_ratio=0.18):
        # bbox_area / frame_area above this ratio is treated as "close"
        # when no depth map is available.
        self.near_area_ratio = near_area_ratio

    @staticmethod
    def _zone(cx, frame_width):
        third = frame_width / 3
        if cx < third:
            return "left"
        if cx > 2 * third:
            return "right"
        return "center"

    @staticmethod
    def _depth_in_bbox(depth_map, bbox):
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = depth_map[y1:y2, x1:x2]
        return float(region.mean()) if region.size else None

    def analyze(self, detections, frame_width, frame_height, depth_map=None):
        """Return (instruction: str, zone_map: dict) describing the most
        relevant (largest / closest) obstacle currently in view.

        `depth_map`, if given, is a MiDaS-style relative-depth array
        (higher value = nearer) the same size as the frame - see
        modules/depth_estimator.py. Optional; distance falls back to
        the bounding-box-size heuristic without it.
        """
        if not detections:
            return "Path looks clear.", {"left": [], "center": [], "right": []}

        frame_area = max(1, frame_width * frame_height)
        # Depth values are only meaningful relative to the *current*
        # frame (MiDaS has no fixed scale), so thresholds are computed
        # per-call from this frame's own depth distribution, never as
        # fixed absolute numbers.
        near_threshold = far_threshold = None
        if depth_map is not None:
            near_threshold = float(np.percentile(depth_map, 85))
            far_threshold = float(np.percentile(depth_map, 50))

        enriched = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            area = max(0, x2 - x1) * max(0, y2 - y1)
            entry = {
                **det,
                "zone": self._zone(cx, frame_width),
                "area_ratio": area / frame_area,
            }
            if depth_map is not None:
                entry["depth_value"] = self._depth_in_bbox(depth_map, det["bbox"])
            enriched.append(entry)

        # The closest/most prominent obstacle: ranked by depth when
        # available (a real per-pixel signal), otherwise by bounding-box
        # size (bigger on screen => nearer, a much cruder proxy).
        if depth_map is not None and any(d.get("depth_value") is not None for d in enriched):
            primary = max(enriched, key=lambda d: d.get("depth_value") or -1)
        else:
            primary = max(enriched, key=lambda d: d["area_ratio"])

        label = primary["label"]
        zone = primary["zone"]

        distance_phrase = ""
        if primary.get("depth_value") is not None:
            depth_value = primary["depth_value"]
            if depth_value >= near_threshold:
                is_close = True
                distance_phrase = " It's very close."
            elif depth_value >= far_threshold:
                is_close = False
            else:
                is_close = False
                distance_phrase = " It's a bit farther away."
        else:
            is_close = primary["area_ratio"] >= self.near_area_ratio

        if zone == "center":
            instruction = f"{label.capitalize()} ahead.{distance_phrase}"
            if is_close:
                left_count = sum(1 for d in enriched if d["zone"] == "left")
                right_count = sum(1 for d in enriched if d["zone"] == "right")
                move_dir = "left" if left_count <= right_count else "right"
                instruction = f"{label.capitalize()} ahead.{distance_phrase} Move slightly {move_dir}."
        elif zone == "left":
            instruction = f"{label.capitalize()} on your left.{distance_phrase}"
        else:
            instruction = f"{label.capitalize()} on your right.{distance_phrase}"

        zone_map = {"left": [], "center": [], "right": []}
        for d in enriched:
            zone_map[d["zone"]].append(d["label"])

        return instruction, zone_map
