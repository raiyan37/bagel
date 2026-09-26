import numpy as np
import pytest
from synthetic import H, W, broadcast_camera, person_detection

from horizon.identify import Identity, PlayerIdentity
from horizon.players import EYE_HEIGHT_RATIO, Players, associate, build_players, fill_gaps, kinematics, smooth
from horizon.tracking import Detections


def identity(near_id=5, far_id=9) -> Identity:
    return Identity(
        players={
            "near": PlayerIdentity("near", "A", "white shirt", near_id, "gemini"),
            "far": PlayerIdentity("far", "B", "navy shirt", far_id, "gemini"),
        },
        score="",
        summary="",
    )


def test_associate_follows_id_switch_skips_gaps_and_ignores_ball_kid():
    cam = broadcast_camera()
    frames = []
    for i in range(10):
        frame = [person_detection(cam, 2, 6.6, -12.5, stature=0.9)]
        if i != 3:
            frame.append(person_detection(cam, 5 if i < 5 else 11, 0.2 * i, -10.0))
        frames.append(frame)
    dets = Detections(W, H, 25.0, frames)
    ids, chosen, xys = associate(dets, cam, "near", start_track_id=5)
    assert ids == [5, 5, 5, None, 5, 11, 11, 11, 11, 11]
    assert chosen[3] is None and xys[3] is None
    assert xys[6] == pytest.approx((1.2, -10.0), abs=1e-6)


def test_associate_reacquires_the_identified_player_after_a_ball_kid_takes_over():
    cam = broadcast_camera()
    frames = []
    for i in range(12):
        frame = [person_detection(cam, 2, 1.0, -10.0, stature=0.9)]  # ball kid about 1 m from the player
        if i < 3 or i >= 8:
            frame.append(person_detection(cam, 5, 0.0, -10.0))
        frames.append(frame)
    ids, _, _ = associate(Detections(W, H, 25.0, frames), cam, "near", start_track_id=5)
    assert ids[:3] == [5, 5, 5]
    assert ids[3:8] == [2] * 5  # while the player is missing the kid is the only person within reach
    assert ids[8:] == [5] * 4  # the identified track wins as soon as it is back


def test_fill_gaps_and_smoothing():
    filled = fill_gaps([None, (0.0, 0.0), None, (2.0, 2.0), None])
    assert filled.tolist() == [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
    ramp = np.stack([np.arange(20.0), np.full(20, 3.0)], axis=1)
    smoothed = smooth(ramp, 7)
    assert np.allclose(smoothed[3:-3], ramp[3:-3])
    assert np.allclose(smoothed[:, 1], 3.0)
    assert np.allclose(smooth(ramp[:2], 7), ramp[:2])
    with pytest.raises(ValueError):
        fill_gaps([None, None])


def test_kinematics_constant_speed():
    xy = np.stack([np.arange(26) * 0.2, np.zeros(26)], axis=1)  # 0.2 m per frame at 25 fps = 5 m/s
    speed, distance = kinematics(xy, 25.0)
    assert np.allclose(speed, 18.0)
    assert distance[0] == 0.0 and distance[-1] == pytest.approx(5.0)
    single_speed, single_distance = kinematics(xy[:1], 25.0)
    assert single_speed.tolist() == [0.0] and single_distance.tolist() == [0.0]


def test_build_players_end_to_end(tmp_path):
    cam = broadcast_camera()
    frames = [[person_detection(cam, 5, 1.0, -10.0, 1.85), person_detection(cam, 9, -2.0, 10.0, 1.90)] for _ in range(12)]
    players = build_players(Detections(W, H, 25.0, frames), cam, identity())
    near, far = players.tracks["near"], players.tracks["far"]
    assert near.stature_m == pytest.approx(1.85, abs=1e-3)
    assert far.eye_height_m == pytest.approx(EYE_HEIGHT_RATIO * 1.90, abs=1e-3)
    assert np.allclose(near.foot_xy, [1.0, -10.0], atol=1e-6)
    assert np.allclose(near.speed_kmh, 0.0) and near.distance_m[-1] == pytest.approx(0.0)
    assert near.eye(4) == pytest.approx([1.0, -10.0, EYE_HEIGHT_RATIO * 1.85], abs=1e-3)
    assert players.opponent("near") is far
    assert all(near.visible) and near.name == "A"
    players.save(tmp_path / "players.json")
    again = Players.load(tmp_path / "players.json")
    assert again.frame_count == 12 and again.fps == 25.0
    assert np.allclose(again.tracks["far"].foot_xy, far.foot_xy, atol=1e-4)
    assert again.tracks["near"].track_ids == [5] * 12


def test_build_players_requires_identified_tracks():
    cam = broadcast_camera()
    with pytest.raises(ValueError, match="near"):
        build_players(Detections(W, H, 25.0, [[]]), cam, identity(near_id=None))
