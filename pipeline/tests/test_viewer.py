import numpy as np
import pytest

from horizon.ground import GroundTexture
from horizon.players import Players, PlayerTrack
from horizon.pov import pov_camera
from horizon.viewer import (
    CHASE_HEIGHT,
    downsample,
    excluded_role,
    export_camera,
    free_camera,
    ground_cloud,
    look_target,
    parse_follow,
    snap_pose,
)


def moving_players(frames: int = 5) -> Players:
    def track(role, start, step):
        foot = np.array([[start[0] + step * i, start[1]] for i in range(frames)], dtype=float)
        return PlayerTrack(role, role, "", [1] * frames, [None] * frames, foot, 1.85, np.zeros(frames), np.zeros(frames))

    return Players(25.0, frames, {"near": track("near", (0.0, -10.0), 0.2), "far": track("far", (1.0, 11.0), -0.1)})


def test_snap_pose_matches_the_player_pov():
    players = moving_players()
    wxyz, position = snap_pose(players, "near", 3, 75.0)
    cam = free_camera(wxyz, position, 75.0, 1024, 576)
    pov = pov_camera(players, "near", 3, 75.0, 1024, 576)
    assert np.allclose(cam.center, pov.center)
    target = np.array([[1.0 - 0.3, 11.0, 1.0]])
    assert np.allclose(cam.project(target)[0], pov.project(target)[0], atol=1e-6)


def test_look_target_is_straight_ahead():
    cam = pov_camera(moving_players(), "far", 0)
    assert np.allclose(look_target(cam, 5.0) - cam.center, 5.0 * cam.forward)


def test_follow_options():
    assert parse_follow("off") is None
    assert parse_follow("near chase") == ("near", "chase")
    assert excluded_role("far eyes") == "far"
    assert excluded_role("far chase") is None and excluded_role("off") is None
    with pytest.raises(ValueError):
        parse_follow("sideways")


def test_export_camera_follows_players_or_stays_static():
    players = moving_players()
    eyes = [export_camera("near eyes", players, f, None, 75.0, 320, 180).center for f in range(5)]
    assert np.allclose([c[0] for c in eyes], [0.0, 0.2, 0.4, 0.6, 0.8])
    chase = export_camera("far chase", players, 2, None, 75.0, 320, 180)
    far_eye = players.tracks["far"].eye(2)
    assert chase.center[1] > far_eye[1] + 2.0  # behind the far player (further from the net)
    assert chase.center[2] == pytest.approx(far_eye[2] + CHASE_HEIGHT)
    assert chase.forward[1] < 0  # looking back towards the near end
    static_pose = snap_pose(players, "near", 0, 60.0)
    frames = [export_camera("off", players, f, static_pose, 60.0, 320, 180).center for f in range(5)]
    assert np.allclose(frames, frames[0])


def test_downsample_and_ground_cloud():
    points = np.random.default_rng(1).normal(size=(10_000, 3)).astype(np.float32)
    colors = np.zeros((10_000, 3), np.uint8)
    small, small_colors = downsample(points, colors, 1000)
    again, _ = downsample(points, colors, 1000)
    assert small.shape == (1000, 3) and small_colors.shape == (1000, 3)
    assert np.array_equal(small, again)
    untouched, _ = downsample(points[:10], colors[:10], 1000)
    assert np.array_equal(untouched, points[:10])
    texture = GroundTexture(np.zeros((1600, 800, 3), np.uint8), -10.0, 10.0, -20.0, 20.0, (0, 0, 0))
    ground_points, ground_colors = ground_cloud(texture, spacing=0.1)
    assert len(ground_points) == 400 * 200 == len(ground_colors)
