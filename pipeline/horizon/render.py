"""Render the reconstructed scene from any pinhole camera.

Pass 1 intersects every pixel ray with the ground plane and samples the ground texture (sky gradient above the
horizon). Pass 2 projects points, splats each over ceil(2 * radius * f / z) pixels (clamped to 1..max_splat) and
keeps the nearest point per pixel that is in front of the ground.
"""

from __future__ import annotations

import numpy as np

from horizon.ball import BALL_RADIUS_M
from horizon.camera import PinholeCamera
from horizon.depth import pixel_grid
from horizon.ground import GroundTexture

SKY_TOP = (10, 12, 18)
SKY_HORIZON = (38, 44, 58)
GROUND_EPSILON = 0.05  # metres; points must be this much in front of the ground to win
BALL_COLOR = (214, 232, 78)
BALL_POINTS = 256


def sky_gradient(width: int, height: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    column = (1 - t) * np.array(SKY_TOP, float) + t * np.array(SKY_HORIZON, float)
    return np.broadcast_to(np.round(column).astype(np.uint8), (height, width, 3)).copy()


def render_view(
    camera: PinholeCamera,
    ground: GroundTexture | None,
    points: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    max_splat: int = 7,
    near: float = 0.3,
) -> np.ndarray:
    width, height = camera.width, camera.height
    image = sky_gradient(width, height)
    flat = image.reshape(-1, 3)
    zbuf = np.full(width * height, np.inf)

    if ground is not None:
        us, vs = pixel_grid(width, height)
        hits, depth = camera.ground_intersection(us.ravel(), vs.ravel())
        valid = np.isfinite(depth) & (depth > near)
        flat[valid] = ground.sample(hits[valid, 0], hits[valid, 1])
        zbuf[valid] = depth[valid]

    if len(points):
        uv, z = camera.project(np.asarray(points, dtype=np.float64))
        ok = (z > near) & np.isfinite(uv).all(axis=1)
        uv, z = uv[ok], z[ok]
        colors, radii = np.asarray(colors)[ok], np.asarray(radii, dtype=np.float64)[ok]
        ui = np.rint(uv[:, 0]).astype(np.int64)
        vi = np.rint(uv[:, 1]).astype(np.int64)
        sizes = np.clip(np.ceil(2.0 * radii * camera.fy / z), 1, max_splat).astype(np.int64)
        lin_parts, src_parts = [], []
        for k in np.unique(sizes):
            sel = np.nonzero(sizes == k)[0]
            offsets = np.arange(k) - (k - 1) // 2
            dx, dy = np.meshgrid(offsets, offsets)
            uu = (ui[sel][:, None] + dx.ravel()[None, :]).ravel()
            vv = (vi[sel][:, None] + dy.ravel()[None, :]).ravel()
            src = np.repeat(sel, k * k)
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            lin_parts.append(vv[inside] * width + uu[inside])
            src_parts.append(src[inside])
        if lin_parts:
            lin = np.concatenate(lin_parts)
            src = np.concatenate(src_parts)
            zz = z[src]
            front = zz < zbuf[lin] - GROUND_EPSILON
            lin, src, zz = lin[front], src[front], zz[front]
            order = np.argsort(zz, kind="stable")
            lin, src = lin[order], src[order]
            unique, first = np.unique(lin, return_index=True)
            flat[unique] = colors[src[first]]
    return image


def _unit_sphere(count: int) -> np.ndarray:
    """`count` roughly equal-area directions on the unit sphere (Fibonacci lattice)."""
    i = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / count
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def ball_cloud(center, radius: float = BALL_RADIUS_M, count: int = BALL_POINTS):
    """The ball as splat points. The per-point radius is set so the splats just merge at any distance.

    render_view sizes a splat as 2 * radius * f / z, so giving each point 2 * R / sqrt(count) of world
    footprint keeps the sphere solid whether it is 1 m or 20 m away, with no special case for either.
    """
    directions = _unit_sphere(count)
    points = (np.asarray(center, dtype=np.float64)[None, :] + directions * radius).astype(np.float32)
    shade = 0.6 + 0.4 * (directions[:, 2] + 1.0) / 2.0  # a cheap overhead light, so it reads as a sphere
    colors = np.clip(np.array(BALL_COLOR, dtype=np.float64)[None, :] * shade[:, None], 0, 255).astype(np.uint8)
    radii = np.full(count, 2.0 * radius / np.sqrt(count), dtype=np.float32)
    return points, colors, radii


def render_frame(
    scene,
    frame: int,
    camera: PinholeCamera,
    exclude_role: str | None = None,
    max_splat: int = 7,
    ball_xyz=None,
) -> np.ndarray:
    player_points, player_colors, player_radii = scene.players_at(frame, exclude_role=exclude_role)
    parts = [
        (scene.background_points, scene.background_colors, scene.background_radii),
        (player_points, player_colors, player_radii),
    ]
    if ball_xyz is not None:
        parts.append(ball_cloud(ball_xyz))
    points = np.concatenate([p for p, _, _ in parts])
    colors = np.concatenate([c for _, c, _ in parts])
    radii = np.concatenate([r for _, _, r in parts])
    return render_view(camera, scene.ground, points, colors, radii, max_splat=max_splat)
