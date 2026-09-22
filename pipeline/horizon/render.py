"""Render the reconstructed scene from any pinhole camera.

Pass 1 intersects every pixel ray with the ground plane and samples the ground texture (sky gradient above the
horizon). Pass 2 projects points, splats each over ceil(2 * radius * f / z) pixels (clamped to 1..max_splat) and
keeps the nearest point per pixel that is in front of the ground.
"""

from __future__ import annotations

import numpy as np

from horizon.camera import PinholeCamera
from horizon.depth import pixel_grid
from horizon.ground import GroundTexture

SKY_TOP = (10, 12, 18)
SKY_HORIZON = (38, 44, 58)
GROUND_EPSILON = 0.05  # metres; points must be this much in front of the ground to win


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


def render_frame(scene, frame: int, camera: PinholeCamera, exclude_role: str | None = None, max_splat: int = 7) -> np.ndarray:
    player_points, player_colors, player_radii = scene.players_at(frame, exclude_role=exclude_role)
    points = np.concatenate([scene.background_points, player_points])
    colors = np.concatenate([scene.background_colors, player_colors])
    radii = np.concatenate([scene.background_radii, player_radii])
    return render_view(camera, scene.ground, points, colors, radii, max_splat=max_splat)
