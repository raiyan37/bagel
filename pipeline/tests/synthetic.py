"""Shared synthetic fixtures: a broadcast camera behind the near baseline and people standing on the court."""

import cv2
import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import HALF_DOUBLES, HALF_LENGTH, HALF_SINGLES
from horizon.depth import pixel_grid
from horizon.identify import Identity, PlayerIdentity
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


WALL_Y, WALL_HEIGHT = 22.0, 12.0
COURT_BLUE, SURROUND_GREEN, LINE_WHITE = (40, 80, 170), (40, 120, 70), (240, 240, 240)
WALL_RED, PLAYER_MAGENTA = (170, 40, 40), (255, 0, 255)
TEST_EXTENT = (-40.0, 40.0, -30.0, 25.0)  # covers every ground pixel the synthetic camera sees


def ground_colors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    colors = np.empty((len(x), 3), np.uint8)
    colors[:] = SURROUND_GREEN
    court = (np.abs(x) <= HALF_DOUBLES) & (np.abs(y) <= HALF_LENGTH)
    colors[court] = COURT_BLUE
    line = court & (
        (np.abs(np.abs(y) - HALF_LENGTH) < 0.05)
        | (np.abs(np.abs(x) - HALF_DOUBLES) < 0.05)
        | (np.abs(np.abs(x) - HALF_SINGLES) < 0.05)
    )
    colors[line] = LINE_WHITE
    return colors


def render_synthetic_frame(camera: PinholeCamera, people=()) -> tuple[np.ndarray, np.ndarray]:
    """Court ground + a red wall behind the far baseline + magenta billboards (x, y, stature). Returns (rgb, depth)."""
    us, vs = pixel_grid(camera.width, camera.height)
    u, v = us.ravel(), vs.ravel()
    ground_pts, ground_depth = camera.ground_intersection(u, v)
    rays = camera.ray_directions(u, v)
    c = camera.center
    with np.errstate(divide="ignore", invalid="ignore"):
        wall_t = (WALL_Y - c[1]) / rays[:, 1]
    wall_z = c[2] + rays[:, 2] * wall_t
    wall_hit = (wall_t > 0) & (wall_z >= 0) & (wall_z <= WALL_HEIGHT) & (wall_t < ground_depth)
    depth = np.where(wall_hit, wall_t, ground_depth).reshape(camera.height, camera.width)
    rgb = ground_colors(ground_pts[:, 0], ground_pts[:, 1])
    rgb[wall_hit] = WALL_RED
    rgb = rgb.reshape(camera.height, camera.width, 3)
    for x, y, stature in people:
        quad = np.array([[x - 0.3, y, 0.0], [x + 0.3, y, 0.0], [x + 0.3, y, stature], [x - 0.3, y, stature]])
        uv, z = camera.project(quad)
        mask = np.zeros((camera.height, camera.width), np.uint8)
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], 1)
        mask = mask.astype(bool)
        rgb[mask] = PLAYER_MAGENTA
        depth[mask] = float(np.mean(z))
    return rgb, depth


def billboard_detection(camera: PinholeCamera, track_id: int, x: float, y: float, stature: float) -> Detection:
    quad = np.array([[x - 0.3, y, 0.0], [x + 0.3, y, 0.0], [x + 0.3, y, stature], [x - 0.3, y, stature]])
    uv, _ = camera.project(quad)
    bbox = (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))
    return Detection(track_id=track_id, conf=0.9, bbox=bbox, polygon=tuple((float(a), float(b)) for a, b in uv))


def make_identity(near_id: int | None = 5, far_id: int | None = 9) -> Identity:
    return Identity(
        players={
            "near": PlayerIdentity("near", "Near Player", "white shirt", near_id, "gemini"),
            "far": PlayerIdentity("far", "Far Player", "navy shirt", far_id, "gemini"),
        },
        score="",
        summary="",
    )


class FakeEstimator:
    """Returns exact disparity for the synthetic background; magenta pixels get one constant 'player' disparity."""

    def __init__(self, background_depth: np.ndarray, scale: float = 0.37, shift: float = -0.004, player_depth: float = 30.0):
        self.base = (1.0 / background_depth - shift) / scale
        self.player = (1.0 / player_depth - shift) / scale

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        disparity = self.base.copy()
        magenta = (rgb[..., 0] > 180) & (rgb[..., 1] < 90) & (rgb[..., 2] > 180)
        disparity[magenta] = self.player
        return disparity.astype(np.float32)
