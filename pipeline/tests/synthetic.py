"""Shared synthetic fixtures: a broadcast camera behind the near baseline and people standing on the court."""

import numpy as np

from horizon.camera import PinholeCamera
from horizon.tracking import Detection

W, H = 1280, 720


def broadcast_camera() -> PinholeCamera:
    return PinholeCamera.look_at(eye=(0.5, -24.0, 9.0), target=(0.0, -3.0, 0.0), vertical_fov_deg=40.0, width=W, height=H)


def person_detection(camera: PinholeCamera, track_id: int, x: float, y: float, stature: float = 1.85, conf: float = 0.9) -> Detection:
    """Box whose bottom-centre is the exact projection of the feet and whose top row is the projected head."""
    uv, _ = camera.project(np.array([[x, y, 0.0], [x, y, stature]]))
    height_px = uv[0, 1] - uv[1, 1]
    half_w = 0.2 * height_px
    cx = uv[0, 0]
    return Detection(track_id=track_id, conf=conf, bbox=(cx - half_w, uv[1, 1], cx + half_w, uv[0, 1]))
