import numpy as np
import pytest

from horizon.ball import detect_candidates, link_track


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


def test_link_track_follows_a_constant_velocity_ball():
    frames, uv = link_track(straight_candidates(10))
    assert frames.tolist() == list(range(10))
    assert uv[3] == pytest.approx((56.0, 45.0))


def test_link_track_ignores_decoys_that_do_not_move_consistently():
    frames, uv = link_track(straight_candidates(10, decoys=2))
    assert frames.tolist() == list(range(10))
    assert uv[9] == pytest.approx((128.0, 75.0))


def test_link_track_bridges_short_gaps():
    candidates = straight_candidates(10)
    candidates[4] = []
    candidates[5] = []
    frames, _uv = link_track(candidates)
    assert frames.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]


def test_link_track_stops_at_a_gap_longer_than_max_gap():
    candidates = straight_candidates(14)
    for i in (5, 6, 7, 8):  # four missing frames, one more than max_gap
        candidates[i] = []
    frames, _uv = link_track(candidates, max_gap=3)
    assert frames.tolist() == [0, 1, 2, 3, 4]


def test_link_track_returns_empty_when_nothing_is_long_enough():
    frames, uv = link_track([[(10.0, 10.0)], [], [], [], [], []])
    assert frames.shape == (0,) and uv.shape == (0, 2)
