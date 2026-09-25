import numpy as np
import pytest

from horizon.ball import detect_candidates


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
