"""Pinhole camera in the OpenCV convention (+X right, +Y down, +Z forward) and pose helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def intrinsics(focal: float, width: int, height: int) -> np.ndarray:
    return np.array([[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def vertical_fov_from_horizontal(hfov_deg: float, width: int, height: int) -> float:
    """Vertical field of view in degrees that matches a horizontal FOV at this aspect ratio."""
    half = np.radians(hfov_deg) / 2.0
    return float(np.degrees(2.0 * np.arctan(np.tan(half) * height / width)))


@dataclass(frozen=True, eq=False)
class PinholeCamera:
    K: np.ndarray  # (3, 3) intrinsics
    R: np.ndarray  # (3, 3) world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        return self.R[2].copy()

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view in radians."""
        return float(2.0 * np.arctan2(self.height / 2.0, self.fy))

    @property
    def projection_matrix(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.R.T + self.t

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixel coordinates (N,2) and camera-space depth z (N,)."""
        pc = self.world_to_camera(points)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / z + self.cx
            v = self.fy * pc[:, 1] / z + self.cy
        return np.stack([u, v], axis=1), z

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels plus camera-space depth -> world points (N,3)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        pc = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=1)
        return (pc - self.t) @ self.R

    def ray_directions(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """World-space ray directions scaled so that their camera-space z component is exactly 1."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        dc = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=1)
        return dc @ self.R

    def ground_intersection(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with the plane z=0. Returns world points and camera depth (inf = miss)."""
        d = self.ray_directions(u, v)
        c = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = -c[2] / d[:, 2]
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, np.inf)
        points = c[None, :] + d * np.where(valid, depth, 0.0)[:, None]
        return points, depth

    def height_above_ground(self, foot_xy: tuple[float, float], v_top: float) -> float:
        """Height of the point on the vertical line through foot_xy that projects onto image row v_top."""
        P = self.projection_matrix
        x, y = foot_xy
        a = P[1, 0] * x + P[1, 1] * y + P[1, 3]
        b = P[1, 2]
        c = P[2, 0] * x + P[2, 1] * y + P[2, 3]
        d = P[2, 2]
        return float((a - v_top * c) / (v_top * d - b))

    @classmethod
    def look_at(
        cls,
        eye,
        target,
        vertical_fov_deg: float,
        width: int,
        height: int,
        up=(0.0, 0.0, 1.0),
    ) -> "PinholeCamera":
        eye = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        if np.linalg.norm(right) < 1e-9:  # looking straight up or down
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        focal = (height / 2.0) / np.tan(np.radians(vertical_fov_deg) / 2.0)
        return cls(K=intrinsics(focal, width, height), R=R, t=-R @ eye, width=width, height=height)


def rotation_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def wxyz_to_rotation(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_pose(cam: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
    """Scene-node pose (wxyz, position) of a camera, as used by Viser frustums and gizmos."""
    return rotation_to_wxyz(cam.R.T), cam.center


def camera_from_pose(wxyz, position, vertical_fov_deg: float, width: int, height: int) -> PinholeCamera:
    R = wxyz_to_rotation(wxyz).T
    position = np.asarray(position, dtype=np.float64)
    focal = (height / 2.0) / np.tan(np.radians(vertical_fov_deg) / 2.0)
    return PinholeCamera(K=intrinsics(focal, width, height), R=R, t=-R @ position, width=width, height=height)


def camera_to_dict(cam: PinholeCamera) -> dict:
    return {
        "K": cam.K.tolist(),
        "R": cam.R.tolist(),
        "t": cam.t.tolist(),
        "width": cam.width,
        "height": cam.height,
    }


def camera_from_dict(data: dict) -> PinholeCamera:
    return PinholeCamera(
        K=np.array(data["K"], dtype=np.float64),
        R=np.array(data["R"], dtype=np.float64),
        t=np.array(data["t"], dtype=np.float64),
        width=int(data["width"]),
        height=int(data["height"]),
    )
