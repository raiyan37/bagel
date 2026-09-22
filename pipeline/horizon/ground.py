"""Top-down texture (orthophoto) of the ground plane, sampled from the clean plate through the broadcast camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from horizon.camera import PinholeCamera
from horizon.court import HALF_DOUBLES, HALF_LENGTH

DEFAULT_EXTENT = (-10.0, 10.0, -20.0, 20.0)  # x_min, x_max, y_min, y_max in metres
DEFAULT_FILL = (40, 90, 60)


@dataclass(frozen=True, eq=False)
class GroundTexture:
    image: np.ndarray  # (rows, cols, 3) uint8; row 0 is y = y_max (far end), column 0 is x = x_min
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    fill: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    @property
    def resolution(self) -> float:
        return (self.x_max - self.x_min) / self.image.shape[1]

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Nearest-texel colours for ground points; the fill colour outside the texture."""
        rows, cols = self.shape
        ci = np.rint((np.asarray(x) - self.x_min) / self.resolution - 0.5).astype(np.int64)
        ri = np.rint((self.y_max - np.asarray(y)) / self.resolution - 0.5).astype(np.int64)
        inside = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
        out = np.empty((len(ci), 3), np.uint8)
        out[:] = self.fill
        out[inside] = self.image[ri[inside], ci[inside]]
        return out

    def world_points(self, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.mgrid[0 : self.shape[0] : stride, 0 : self.shape[1] : stride]
        x = self.x_min + (cols + 0.5) * self.resolution
        y = self.y_max - (rows + 0.5) * self.resolution
        points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
        return points.astype(np.float32), self.image[rows.ravel(), cols.ravel()]


def build_ground_texture(
    plate_rgb: np.ndarray,
    camera: PinholeCamera,
    ground_mask: np.ndarray,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    resolution: float = 0.025,
) -> GroundTexture:
    x_min, x_max, y_min, y_max = extent
    cols = int(round((x_max - x_min) / resolution))
    rows = int(round((y_max - y_min) / resolution))
    r, c = np.mgrid[0:rows, 0:cols]
    x = x_min + (c + 0.5) * resolution
    y = y_max - (r + 0.5) * resolution
    uv, z = camera.project(np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1))
    unusable = (z <= 0) | ~np.isfinite(uv).all(axis=1)
    uv[unusable] = -1.0
    map_x = uv[:, 0].reshape(rows, cols).astype(np.float32)
    map_y = uv[:, 1].reshape(rows, cols).astype(np.float32)
    texture = cv2.remap(plate_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.remap(
        ground_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    valid &= ~unusable.reshape(rows, cols)
    ring = valid & ((np.abs(x) > HALF_DOUBLES + 0.5) | (np.abs(y) > HALF_LENGTH + 0.5))
    if ring.sum() >= 100:
        fill = tuple(int(v) for v in np.median(texture[ring], axis=0))
    elif valid.any():
        fill = tuple(int(v) for v in np.median(texture[valid], axis=0))
    else:
        fill = DEFAULT_FILL
    texture[~valid] = fill
    return GroundTexture(texture, x_min, x_max, y_min, y_max, fill)
