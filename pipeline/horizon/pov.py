"""First-person cameras for both players and clip rendering."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from horizon.camera import PinholeCamera, vertical_fov_from_horizontal
from horizon.players import Players
from horizon.render import render_frame
from horizon.video import H264Writer

POV_WIDTH, POV_HEIGHT = 1024, 576
POV_HFOV_DEG = 75.0
TARGET_HEIGHT = 1.0  # look at the opponent's chest


def pov_camera(
    players: Players, role: str, frame: int, hfov_deg: float = POV_HFOV_DEG, width: int = POV_WIDTH, height: int = POV_HEIGHT
) -> PinholeCamera:
    eye = players.tracks[role].eye(frame)
    ox, oy = players.opponent(role).foot_xy[frame]
    target = np.array([ox, oy, TARGET_HEIGHT])
    if np.hypot(target[0] - eye[0], target[1] - eye[1]) < 0.5:
        target = eye + np.array([0.0, 1.0 if role == "near" else -1.0, 0.0])  # face the net
    return PinholeCamera.look_at(eye, target, vertical_fov_from_horizontal(hfov_deg, width, height), width, height)


def render_clip(
    scene,
    camera_for_frame: Callable[[int], PinholeCamera],
    out_path: Path,
    exclude_role: str | None = None,
    frame_count: int | None = None,
    ball=None,
    log: Callable[[str], None] | None = None,
) -> int:
    total = scene.frame_count if frame_count is None else min(frame_count, scene.frame_count)
    first = camera_for_frame(0)
    with H264Writer(out_path, first.width, first.height, scene.fps) as writer:
        for frame in range(total):
            camera = first if frame == 0 else camera_for_frame(frame)
            ball_xyz = None if ball is None else ball.position(frame)
            writer.write(render_frame(scene, frame, camera, exclude_role=exclude_role, ball_xyz=ball_xyz))
            if log and frame % 50 == 0:
                log(f"  {Path(out_path).name}: frame {frame}/{total}")
        return writer.frames_written


def render_pov_clip(
    scene,
    players: Players,
    role: str,
    out_path: Path,
    hfov_deg: float = POV_HFOV_DEG,
    width: int = POV_WIDTH,
    height: int = POV_HEIGHT,
    ball=None,
    log: Callable[[str], None] | None = None,
) -> int:
    return render_clip(
        scene,
        lambda frame: pov_camera(players, role, frame, hfov_deg, width, height),
        out_path,
        exclude_role=role,
        frame_count=players.frame_count,
        ball=ball,
        log=log,
    )
