import numpy as np
import pytest
from synthetic import broadcast_camera

from horizon.ground import GroundTexture
from horizon.players import EYE_HEIGHT_RATIO, Players, PlayerTrack
from horizon.pov import pov_camera, render_pov_clip
from horizon.reconstruct import Scene
from horizon.render import sky_gradient
from horizon.video import iter_frames, probe


def static_players(near_xy, far_xy, frames=3, stature=1.85) -> Players:
    def track(role, xy):
        foot = np.tile(np.array(xy, dtype=float), (frames, 1))
        return PlayerTrack(role, role.title(), "", [1] * frames, [(0.0, 0.0, 1.0, 1.0)] * frames, foot, stature, np.zeros(frames), np.zeros(frames))

    return Players(25.0, frames, {"near": track("near", near_xy), "far": track("far", far_xy)})


def empty_scene(frames: int) -> Scene:
    ground = np.zeros((400, 200, 3), np.uint8)  # 0.1 m texels over 20 m x 40 m
    ground[:] = (40, 120, 70)
    return Scene(
        background_points=np.zeros((0, 3), np.float32),
        background_colors=np.zeros((0, 3), np.uint8),
        background_radii=np.zeros(0, np.float32),
        player_points=np.zeros((0, 3), np.float32),
        player_colors=np.zeros((0, 3), np.uint8),
        player_radii=np.zeros(0, np.float32),
        player_roles=np.zeros(0, np.uint8),
        frame_offsets=np.zeros(frames + 1, np.int64),
        ground=GroundTexture(ground, -10.0, 10.0, -20.0, 20.0, (40, 120, 70)),
        camera=broadcast_camera(),
        fps=25.0,
    )


def test_pov_camera_sits_at_the_eye_and_faces_the_opponent():
    players = static_players((1.0, -10.0), (-2.0, 10.0))
    cam = pov_camera(players, "near", 1)
    eye = np.array([1.0, -10.0, EYE_HEIGHT_RATIO * 1.85])
    target = np.array([-2.0, 10.0, 1.0])
    assert np.allclose(cam.center, eye)
    assert np.allclose(cam.forward, (target - eye) / np.linalg.norm(target - eye))
    uv, _ = cam.project(target[None, :])
    assert uv[0] == pytest.approx([cam.cx, cam.cy])
    assert (cam.width, cam.height) == (1024, 576)
    assert cam.fx == pytest.approx(512 / np.tan(np.radians(37.5)), rel=1e-6)


def test_pov_camera_faces_the_net_when_players_overlap():
    players = static_players((0.0, 3.0), (0.0, 3.2))
    assert pov_camera(players, "near", 0).forward[1] > 0.99
    assert pov_camera(players, "far", 0).forward[1] < -0.99


def test_render_pov_clip_writes_a_frame_aligned_video(tmp_path):
    players = static_players((1.0, -10.0), (-2.0, 10.0), frames=4)
    out = tmp_path / "pov_near.mp4"
    written = render_pov_clip(empty_scene(4), players, "near", out, width=160, height=96)
    assert written == 4
    info = probe(out)
    assert (info.width, info.height) == (160, 96) and info.fps == pytest.approx(25.0)
    frames = [f for _, f in iter_frames(out)]
    assert len(frames) == 4
    sky = sky_gradient(160, 96)
    assert np.abs(frames[0][-10:].astype(int) - sky[-10:].astype(int)).mean() > 20  # the ground fills the bottom
