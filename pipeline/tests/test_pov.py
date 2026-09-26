import numpy as np
import pytest
from synthetic import broadcast_camera

from horizon.ball import BallTrack
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


def ball_in_front_of(players, role, frames, distance=1.0):
    """A stationary ball hanging a short way along the player's view direction."""
    eye = players.tracks[role].eye(0)
    point = eye + pov_camera(players, role, 0).forward * distance
    return BallTrack(
        fps=25.0,
        frame_count=frames,
        xyz=np.tile(point, (frames, 1)),
        reprojection_rms_px=0.0,
        segments=[(0, frames - 1)],
    )


def ball_pixel_count(image):
    """Pixels carrying a shaded tennis-ball colour. empty_scene's ground is a dark green whose red channel
    is far too low to qualify, and the sky is darker still, so only the ball is counted."""
    pixels = image.astype(int)
    return int(((pixels[..., 0] > 110) & (pixels[..., 1] > 120) & (pixels[..., 2] < 150)).sum())


def test_render_pov_clip_without_a_ball_is_unchanged(tmp_path):
    """Review Focus 2: a match with no ball.json renders exactly as before."""
    players = static_players((1.0, -10.0), (-2.0, 10.0), frames=3)
    plain, explicit = tmp_path / "plain.mp4", tmp_path / "explicit.mp4"
    render_pov_clip(empty_scene(3), players, "near", plain, width=160, height=96)
    render_pov_clip(empty_scene(3), players, "near", explicit, width=160, height=96, ball=None)
    assert plain.read_bytes() == explicit.read_bytes()


def test_render_pov_clip_draws_the_ball(tmp_path):
    players = static_players((1.0, -10.0), (-2.0, 10.0), frames=3)
    out = tmp_path / "pov_near.mp4"
    ball = ball_in_front_of(players, "near", 3)
    render_pov_clip(empty_scene(3), players, "near", out, width=160, height=96, ball=ball)
    frames = [f for _, f in iter_frames(out)]
    assert len(frames) == 3
    assert ball_pixel_count(frames[0]) > 10


def test_render_pov_clip_hides_the_ball_on_frames_it_has_no_position_for(tmp_path):
    players = static_players((1.0, -10.0), (-2.0, 10.0), frames=3)
    ball = ball_in_front_of(players, "near", 3)
    ball.xyz[2] = np.nan
    out = tmp_path / "pov_near.mp4"
    render_pov_clip(empty_scene(3), players, "near", out, width=160, height=96, ball=ball)
    frames = [f for _, f in iter_frames(out)]
    assert ball_pixel_count(frames[0]) > 10
    assert ball_pixel_count(frames[2]) < 5  # not == 0: H.264 can ghost a few pixels from the previous frame


def test_render_pov_clip_tolerates_a_ball_shorter_than_the_clip(tmp_path):
    """Review Focus 3, at the render boundary: a two-frame ball.json against a four-frame clip."""
    players = static_players((1.0, -10.0), (-2.0, 10.0), frames=4)
    out = tmp_path / "pov_near.mp4"
    ball = ball_in_front_of(players, "near", 2)
    assert render_pov_clip(empty_scene(4), players, "near", out, width=160, height=96, ball=ball) == 4
