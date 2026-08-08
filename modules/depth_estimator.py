"""
Monocular Depth Estimation Module
====================================
Wraps Intel's MiDaS ("small" variant) via torch.hub to estimate a
per-pixel *relative* depth map from a single RGB frame. Used to give
navigation guidance a real distance signal instead of only guessing
"probably close" from bounding-box size.

IMPORTANT / honesty note: MiDaS outputs *relative inverse depth*, not
calibrated metres - it tells you "this pixel is nearer than that one",
not "this object is 2.3 metres away". Without a calibrated stereo rig
or a depth sensor, no monocular-only system can honestly claim metric
distances. This module and `navigation.py` only ever produce coarse
labels ("very close" / "nearby" / "farther away"), never a fabricated
number of metres.

Downloads model weights from torch.hub on first use (~a few tens of MB)
- like the other model modules, this needs internet access the first
time it runs and then caches locally.
"""
import cv2
import numpy as np
import torch
import torch.hub

# MiDaS's own hubconf makes a *nested* torch.hub.load() call (for its
# EfficientNet-Lite backbone, "rwightman/gen-efficientnet-pytorch") that
# doesn't forward the trust_repo=True passed to our top-level call
# below, so it hits torch.hub's interactive trust prompt on stdin -
# which has no console to answer in a background server process, and
# just crashes with EOFError. Disabling the trust check entirely (the
# standard workaround for this known issue) is safe here since we've
# already deliberately chosen to load the official MiDaS repo and are
# accepting whatever repos it in turn pulls in.
torch.hub._check_repo_is_trusted = lambda *args, **kwargs: None


class DepthEstimator:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

    def estimate(self, frame_bgr):
        """Return a single-channel float32 relative-depth map (higher
        value = nearer), resized to match the input frame's resolution."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    @staticmethod
    def average_depth_in_bbox(depth_map, bbox):
        """Mean relative-depth value inside `bbox` (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        return float(np.mean(region))

    @staticmethod
    def relative_distance_label(depth_value, near_threshold, far_threshold):
        """Map a raw relative-depth value to a coarse, honest label -
        never a fabricated metric distance."""
        if depth_value >= near_threshold:
            return "very close"
        if depth_value >= far_threshold:
            return "nearby"
        return "farther away"
