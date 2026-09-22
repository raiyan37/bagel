"""Solve the static broadcast camera from clicked court keypoints.

For a candidate focal length, planar PnP (IPPE) gives up to two poses. We keep the one with the camera
above the ground and the lowest reprojection error, and search the focal length that minimises it.
The principal point is fixed at the image centre and lens distortion is ignored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from horizon.camera import PinholeCamera, camera_from_dict, camera_to_dict, intrinsics
from horizon.court import KEYPOINTS, court_lines, net_quad

MAX_ACCEPTABLE_RMS_PX = 4.0


@dataclass(frozen=True)
class Calibration:
    camera: PinholeCamera
    keypoints: dict[str, tuple[float, float]]
    rms_px: float


def _best_pose(world: np.ndarray, image: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(world, image, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best = None
    for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        if (-R.T @ t)[2] <= 0:  # camera below the court: mirror solution
            continue
        projected, _ = cv2.projectPoints(world, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image) ** 2, axis=1))))
        if best is None or rms < best[2]:
            best = (R, t, rms)
    return best


def _golden_section(fn: Callable[[float], float], lo: float, hi: float, iterations: int = 60) -> float:
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def solve_calibration(keypoints: dict[str, tuple[float, float]], width: int, height: int) -> Calibration:
    names = [name for name in KEYPOINTS if name in keypoints]
    if len(names) < 4:
        raise ValueError(f"Need at least 4 court keypoints, got {len(names)}")
    world = np.array([[*KEYPOINTS[n], 0.0] for n in names], dtype=np.float64)
    image = np.array([keypoints[n] for n in names], dtype=np.float64)

    def rms_for(focal: float) -> float:
        best = _best_pose(world, image, intrinsics(focal, width, height))
        return np.inf if best is None else best[2]

    grid = np.geomspace(0.3 * width, 8.0 * width, 240)
    errors = np.array([rms_for(f) for f in grid])
    if not np.isfinite(errors).any():
        raise ValueError("No camera pose above the court explains these keypoints; check the click order")
    i = int(np.argmin(errors))
    focal = _golden_section(rms_for, grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)])
    best = _best_pose(world, image, intrinsics(focal, width, height))
    if best is None or best[2] > errors[i]:
        focal = float(grid[i])
        best = _best_pose(world, image, intrinsics(focal, width, height))
    R, t, rms = best
    camera = PinholeCamera(K=intrinsics(focal, width, height), R=R, t=t, width=width, height=height)
    clicked = {n: (float(keypoints[n][0]), float(keypoints[n][1])) for n in names}
    return Calibration(camera=camera, keypoints=clicked, rms_px=rms)


def save_calibration(calibration: Calibration, path: Path) -> None:
    data = {
        "camera": camera_to_dict(calibration.camera),
        "keypoints": {k: list(v) for k, v in calibration.keypoints.items()},
        "rms_px": calibration.rms_px,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration(path: Path) -> Calibration:
    data = json.loads(Path(path).read_text())
    return Calibration(
        camera=camera_from_dict(data["camera"]),
        keypoints={k: (float(v[0]), float(v[1])) for k, v in data["keypoints"].items()},
        rms_px=float(data["rms_px"]),
    )


def _pt(uv: np.ndarray) -> tuple[int, int]:
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def draw_court_overlay(frame_rgb: np.ndarray, camera: PinholeCamera, keypoints: dict | None = None) -> np.ndarray:
    """Court lines (red), net (yellow) and clicked keypoints (green) drawn over a copy of the frame."""
    out = frame_rgb.copy()
    for (x0, y0), (x1, y1) in court_lines():
        uv, z = camera.project(np.array([[x0, y0, 0.0], [x1, y1, 0.0]]))
        if (z > 0).all():
            cv2.line(out, _pt(uv[0]), _pt(uv[1]), (255, 64, 64), 2, cv2.LINE_AA)
    uv, z = camera.project(net_quad())
    if (z > 0).all():
        cv2.polylines(out, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (255, 220, 0), 2, cv2.LINE_AA)
    for u, v in (keypoints or {}).values():
        cv2.circle(out, (int(round(u)), int(round(v))), 5, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_keypoint_legend(canvas_bgr: np.ndarray, current: str | None, size: tuple[int, int] = (120, 220)) -> None:
    """Top-down court diagram in the top-right corner; the keypoint to click next is red."""
    lw, lh = size
    x0 = canvas_bgr.shape[1] - lw - 12
    y0 = 40
    cv2.rectangle(canvas_bgr, (x0 - 6, y0 - 6), (x0 + lw + 6, y0 + lh + 18), (0, 0, 0), -1)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(x0 + (x + 6.5) / 13.0 * lw)), int(round(y0 + (13.0 - y) / 26.0 * lh))

    for (xa, ya), (xb, yb) in court_lines():
        cv2.line(canvas_bgr, to_px(xa, ya), to_px(xb, yb), (200, 200, 200), 1, cv2.LINE_AA)
    for name, (x, y) in KEYPOINTS.items():
        color, radius = ((0, 0, 255), 5) if name == current else ((160, 160, 160), 2)
        cv2.circle(canvas_bgr, to_px(x, y), radius, color, -1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "camera side", (x0 + 18, y0 + lh + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
