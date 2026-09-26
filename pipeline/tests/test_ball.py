import numpy as np
import pytest
from synthetic import broadcast_camera

from horizon.ball import (
    GRAVITY,
    BallTrack,
    ball_candidates,
    ballistic_positions,
    build_ball_track,
    detect_candidates,
    fit_ballistic,
    is_plausible,
    link_tracks,
    reprojection_residuals,
    split_flights,
)
from horizon.paths import MatchPaths
from horizon.players import PlayerTrack, Players
from horizon.tracking import Detections
from horizon.video import H264Writer


def court_plate(height=120, width=160):
    """A blue hard court with a white line, as uint8 RGB."""
    plate = np.zeros((height, width, 3), np.uint8)
    plate[:] = (40, 80, 170)
    plate[:, 70:76] = (240, 240, 240)
    return plate


def with_ball(plate, u, v, radius=3, color=(205, 220, 60)):
    frame = plate.copy()
    rows, cols = np.ogrid[: frame.shape[0], : frame.shape[1]]
    frame[(cols - u) ** 2 + (rows - v) ** 2 <= radius**2] = color
    return frame


def no_exclusion(plate):
    return np.zeros(plate.shape[:2], bool)


def test_detect_candidates_finds_a_yellow_blob_on_the_blue_court():
    plate = court_plate()
    found = detect_candidates(with_ball(plate, 100, 40), plate, no_exclusion(plate))
    assert len(found) == 1
    assert all(isinstance(c, float) for c in found[0])
    assert abs(found[0][0] - 100) < 1.0 and abs(found[0][1] - 40) < 1.0


def test_detect_candidates_ignores_the_white_line_and_unchanged_court():
    plate = court_plate()
    assert detect_candidates(plate, plate, no_exclusion(plate)) == []


def test_detect_candidates_ignores_blobs_that_are_not_yellow():
    plate = court_plate()
    white = with_ball(plate, 100, 40, color=(230, 230, 235))  # G - B is negative
    assert detect_candidates(white, plate, no_exclusion(plate)) == []


def test_detect_candidates_drops_blobs_that_are_too_big():
    plate = court_plate()
    huge = with_ball(plate, 80, 60, radius=30)  # ~2800 px, far over max_area
    assert detect_candidates(huge, plate, no_exclusion(plate)) == []


def test_detect_candidates_respects_the_exclusion_mask():
    plate = court_plate()
    exclude = no_exclusion(plate)
    exclude[30:50, 90:110] = True
    assert detect_candidates(with_ball(plate, 100, 40), plate, exclude) == []


def straight_candidates(frames=10, start=(20.0, 30.0), step=(12.0, 5.0), decoys=0):
    """A ball at constant pixel velocity, plus `decoys` distractors far away from it."""
    out = []
    for i in range(frames):
        frame = [(start[0] + step[0] * i, start[1] + step[1] * i)]
        for k in range(decoys):
            frame.append((600.0 + 30 * k, 400.0 - 20 * k * (i % 2)))
        out.append(frame)
    return out


def test_link_tracks_follows_a_constant_velocity_ball():
    frames, uv = link_tracks(straight_candidates(10))[0]
    assert frames.tolist() == list(range(10))
    assert uv[3] == pytest.approx((56.0, 45.0))


def test_link_tracks_ignores_decoys_that_do_not_move_consistently():
    frames, uv = link_tracks(straight_candidates(10, decoys=2))[0]
    assert frames.tolist() == list(range(10))
    assert uv[9] == pytest.approx((128.0, 75.0))


def test_link_tracks_bridges_short_gaps():
    candidates = straight_candidates(10)
    candidates[4] = []
    candidates[5] = []
    frames, _uv = link_tracks(candidates)[0]
    assert frames.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]


def test_link_tracks_stops_at_a_gap_longer_than_max_gap():
    candidates = straight_candidates(14)
    for i in (5, 6, 7, 8):  # four missing frames, one more than max_gap
        candidates[i] = []
    frames, _uv = link_tracks(candidates, max_gap=3)[0]
    assert frames.tolist() == [0, 1, 2, 3, 4]


def test_link_tracks_returns_nothing_when_nothing_is_long_enough():
    assert link_tracks([[(10.0, 10.0)], [], [], [], [], []]) == []


def test_link_tracks_returns_every_tracklet_longest_first():
    """build_ball_track needs them all, because the longest is not necessarily the ball."""
    candidates = [[c] for c in noisy_cluster(30)]
    for i in range(5, 17):
        candidates[i].append((200.0 + 14.0 * (i - 5), 500.0 - 6.0 * (i - 5)))
    tracks = link_tracks(candidates)
    assert len(tracks) >= 2
    assert [len(f) for f, _ in tracks] == sorted((len(f) for f, _ in tracks), reverse=True)
    assert any(f.tolist() == list(range(5, 17)) for f, _ in tracks), "the short straight run must survive"


def flight(a=(2.0, -8.0, 1.2), v=(-1.5, 16.0, 4.0), frames=range(14), fps=25.0):
    """Ground-truth projectile sampled on `frames`, with its exact projections in the broadcast camera."""
    camera = broadcast_camera()
    index = np.array(list(frames), dtype=np.int64)
    t = (index - index[0]) / fps
    uv, _z = camera.project(ballistic_positions(np.array(a), np.array(v), t))
    return camera, index, uv, t


def test_ballistic_positions_follow_gravity():
    points = ballistic_positions(np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0]))
    assert points[0] == pytest.approx([0.0, 0.0, 2.0])
    assert points[1] == pytest.approx([1.0, 0.0, 2.0 - 0.5 * 9.81])
    assert GRAVITY == pytest.approx([0.0, 0.0, -9.81])


def test_fit_ballistic_recovers_the_exact_trajectory_from_clean_observations():
    camera, index, uv, _t = flight()
    a, v = fit_ballistic(camera, index, uv, 25.0)
    assert a == pytest.approx([2.0, -8.0, 1.2], abs=1e-6)
    assert v == pytest.approx([-1.5, 16.0, 4.0], abs=1e-6)


def test_fit_ballistic_survives_sub_pixel_detection_noise():
    """Bounds are loose on purpose: the point is that half-pixel noise does not produce garbage."""
    camera, index, uv, _t = flight()
    rng = np.random.default_rng(0)
    a, v = fit_ballistic(camera, index, uv + rng.normal(0.0, 0.5, uv.shape), 25.0)
    assert a == pytest.approx([2.0, -8.0, 1.2], abs=0.6)
    assert v == pytest.approx([-1.5, 16.0, 4.0], abs=2.0)


def test_fit_ballistic_needs_three_observations():
    camera, index, uv, _t = flight(frames=range(2))
    with pytest.raises(ValueError, match="at least 3"):
        fit_ballistic(camera, index, uv, 25.0)


def test_fit_ballistic_refuses_a_degenerate_flight_down_the_camera_axis():
    """Review Focus 4: with every observation on one pixel the trajectory is not determined at all.

    lstsq would happily return its minimum-norm solution, which reprojects perfectly but sits at an
    arbitrary depth. Refusing is the only honest answer.
    """
    camera = broadcast_camera()
    index = np.arange(8)
    depths = 12.0 + 18.0 * index / 25.0
    points = camera.center[None, :] + camera.forward[None, :] * depths[:, None]
    uv, _z = camera.project(points)
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_ballistic(camera, index, uv, 25.0)


def test_reprojection_residuals_are_zero_for_the_true_trajectory():
    camera, index, uv, t = flight()
    residuals = reprojection_residuals(camera, np.array([2.0, -8.0, 1.2]), np.array([-1.5, 16.0, 4.0]), t, uv)
    assert residuals.shape == (len(index),)
    assert residuals.max() < 1e-6


def test_is_plausible_accepts_a_real_rally_shot():
    camera, _index, _uv, t = flight()
    assert is_plausible(camera, np.array([2.0, -8.0, 1.2]), np.array([-1.5, 16.0, 4.0]), t)


def test_is_plausible_rejects_a_trajectory_under_the_court():
    t = np.linspace(0.0, 0.5, 8)
    assert not is_plausible(broadcast_camera(), np.array([0.0, 0.0, -3.0]), np.array([0.0, 5.0, 0.0]), t)


def test_is_plausible_rejects_an_impossibly_fast_ball():
    t = np.linspace(0.0, 0.5, 8)
    assert not is_plausible(broadcast_camera(), np.array([0.0, 0.0, 1.0]), np.array([0.0, 500.0, 0.0]), t)


def test_is_plausible_rejects_a_ball_that_leaves_the_court_box():
    t = np.linspace(0.0, 0.5, 8)
    assert not is_plausible(broadcast_camera(), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 60.0]), t)


def test_is_plausible_rejects_non_finite_parameters():
    t = np.linspace(0.0, 0.5, 8)
    assert not is_plausible(broadcast_camera(), np.array([np.nan, 0.0, 1.0]), np.array([0.0, 5.0, 0.0]), t)


def test_is_plausible_rejects_a_trajectory_sitting_on_top_of_the_camera():
    """A jittering blob can be fit by a tiny trajectory right at the lens: it reprojects perfectly and
    explains nothing, because near the camera any 3D wobble produces any pixel motion. Real balls are
    metres away, and without this bound the ao clip accepts three flights made of ball-kid jitter."""
    camera = broadcast_camera()
    t = np.linspace(0.0, 0.2, 6)
    assert not is_plausible(camera, camera.center + np.array([0.1, 0.1, 0.0]), np.array([0.5, 0.5, 0.0]), t)


FLIGHT_A, FLIGHT_V = np.array([2.0, -9.0, 1.0]), np.array([-0.6, 15.0, 4.2])


def two_flight_candidates(fps=25.0, span=12):
    """A ball struck again at frame `span`. A racket changes the velocity, not the position, so the 2D
    track stays continuous and one parabola still cannot explain the whole thing."""
    camera = broadcast_camera()
    t = np.arange(span) / fps
    first = ballistic_positions(FLIGHT_A, FLIGHT_V, t)
    contact = ballistic_positions(FLIGHT_A, FLIGHT_V, np.array([span / fps]))[0]
    second = ballistic_positions(contact, np.array([0.4, -14.0, 3.0]), t)
    uv, _z = camera.project(np.vstack([first, second]))
    return camera, [[(float(u), float(v))] for u, v in uv]


def test_split_flights_keeps_a_single_parabola_whole():
    camera, index, uv, _t = flight(frames=range(14))
    segments = split_flights(camera, index, uv, 25.0)
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (0, 13)


def test_split_flights_cuts_at_the_racket_hit():
    camera, candidates = two_flight_candidates()
    frames, uv = link_tracks(candidates)[0]
    assert len(frames) == 24, "the hit must not break the 2D tracklet"
    segments = split_flights(camera, frames, uv, 25.0)
    assert len(segments) >= 2
    assert not any(s.start <= 5 and s.end >= 18 for s in segments)  # nothing spans the hit at frame 12


def test_split_flights_drops_spans_that_are_too_short():
    camera, index, uv, _t = flight(frames=range(3))
    assert split_flights(camera, index, uv, 25.0, min_length=4) == []


def test_split_flights_drops_a_degenerate_span_instead_of_raising():
    camera = broadcast_camera()
    index = np.arange(8)
    depths = 12.0 + 18.0 * index / 25.0
    uv, _z = camera.project(camera.center[None, :] + camera.forward[None, :] * depths[:, None])
    assert split_flights(camera, index, uv, 25.0) == []


def test_build_ball_track_gives_a_world_position_per_covered_frame():
    camera, candidates = two_flight_candidates()
    track = build_ball_track(camera, candidates, fps=25.0, frame_count=len(candidates))
    assert track.frame_count == len(candidates)
    assert track.detected_frames >= 16
    assert track.reprojection_rms_px < 3.0
    expected = ballistic_positions(FLIGHT_A, FLIGHT_V, np.array([2 / 25.0]))[0]
    assert track.position(2) == pytest.approx(expected, abs=0.1)


def noisy_cluster(frames, center=(1050.0, 200.0), seed=7, jitter=9.0):
    """A persistent jittering blob, like the ball kids and umpire chair in real broadcast footage:
    long-lived, slow, and not on any parabola."""
    rng = np.random.default_rng(seed)
    return [(center[0] + rng.normal(0.0, jitter), center[1] + rng.normal(0.0, jitter)) for _ in range(frames)]


def test_build_ball_track_prefers_the_ballistic_tracklet_over_a_longer_noisy_one():
    """The real ao clip fails exactly here: the ball is on screen for ~20 frames while a cluster of
    green-shirted ball kids jitters in frame for all 105, so the longest tracklet is never the ball."""
    camera = broadcast_camera()
    total, first, last = 60, 20, 41
    noise = noisy_cluster(total)
    t = np.arange(last - first) / 25.0
    ball_uv, _z = camera.project(ballistic_positions(FLIGHT_A, FLIGHT_V, t))
    candidates = [[noise[i]] for i in range(total)]
    for k, i in enumerate(range(first, last)):
        candidates[i].append((float(ball_uv[k, 0]), float(ball_uv[k, 1])))

    track = build_ball_track(camera, candidates, fps=25.0, frame_count=total)
    assert track.detected_frames >= 15, "the ballistic tracklet must win over the longer noisy one"
    expected = ballistic_positions(FLIGHT_A, FLIGHT_V, np.array([5 / 25.0]))[0]
    assert track.position(first + 5) == pytest.approx(expected, abs=0.1)
    assert track.position(0) is None  # the noise cluster contributes nothing


def test_build_ball_track_with_no_candidates_is_empty_not_an_error():
    """Review Focus 1: a clip where the ball is never found still produces a usable artifact."""
    track = build_ball_track(broadcast_camera(), [[] for _ in range(20)], fps=25.0, frame_count=20)
    assert track.frame_count == 20
    assert track.detected_frames == 0
    assert track.segments == []
    assert track.position(0) is None
    assert track.reprojection_rms_px == 0.0


def test_ball_track_round_trips_through_json(tmp_path):
    camera, candidates = two_flight_candidates()
    track = build_ball_track(camera, candidates, fps=25.0, frame_count=len(candidates))
    path = tmp_path / "ball.json"
    track.save(path)
    loaded = BallTrack.load(path)
    assert loaded.fps == track.fps and loaded.frame_count == track.frame_count
    assert loaded.segments == track.segments
    assert loaded.reprojection_rms_px == pytest.approx(track.reprojection_rms_px, abs=1e-3)
    np.testing.assert_allclose(np.nan_to_num(loaded.xyz, nan=-999.0), np.nan_to_num(track.xyz, nan=-999.0), atol=1e-3)


def test_ball_track_position_is_none_outside_the_clip():
    """Review Focus 3: a ball.json shorter than the scene must not raise on the extra frames."""
    track = BallTrack(fps=25.0, frame_count=2, xyz=np.full((2, 3), np.nan), reprojection_rms_px=0.0, segments=[])
    assert track.position(5) is None
    assert track.position(-1) is None


def test_match_paths_expose_the_ball_artifact(tmp_path):
    assert MatchPaths.for_match("demo", tmp_path).ball == tmp_path / "demo" / "ball.json"


def test_ball_candidates_finds_the_ball_in_a_written_clip(tmp_path):
    plate = court_plate(96, 160)
    positions = [(30 + 8 * i, 20 + 3 * i) for i in range(12)]
    path = tmp_path / "clip.mp4"
    with H264Writer(path, 160, 96, 25.0) as writer:
        for u, v in positions:
            writer.write(with_ball(plate, u, v, radius=3))
    count = len(positions)
    detections = Detections(160, 96, 25.0, [[] for _ in positions])
    absent = PlayerTrack(
        "near", "Near", "", [None] * count, [None] * count,
        np.zeros((count, 2)), 1.85, np.zeros(count), np.zeros(count),
    )
    players = Players(25.0, count, {"near": absent})
    candidates = ball_candidates(path, count, detections, players)
    assert len(candidates) == count
    assert sum(1 for frame in candidates if frame) >= 9  # H.264 is lossy; a couple of frames may drop out
