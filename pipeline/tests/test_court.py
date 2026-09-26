import numpy as np
import pytest

from horizon.court import (
    HALF_DOUBLES,
    HALF_LENGTH,
    HALF_SINGLES,
    KEYPOINTS,
    court_lines,
    court_rect,
    keypoint_world,
    net_quad,
)


def test_itf_dimensions():
    assert HALF_LENGTH * 2 == pytest.approx(23.77)
    assert HALF_DOUBLES * 2 == pytest.approx(10.97)
    assert HALF_SINGLES * 2 == pytest.approx(8.23)


def test_keypoints_are_mirror_symmetric_and_on_the_ground():
    assert len(KEYPOINTS) == 14
    for name, (x, y) in KEYPOINTS.items():
        mirror = name.replace("left", "@").replace("right", "left").replace("@", "right")
        mx, my = KEYPOINTS[mirror]
        assert (mx, my) == pytest.approx((-x, y))
        assert keypoint_world(name)[2] == 0.0


def test_lines_net_and_rect():
    lines = court_lines()
    assert len(lines) == 9
    for (x0, y0), (x1, y1) in lines:
        for x, y in ((x0, y0), (x1, y1)):
            assert abs(x) <= HALF_DOUBLES + 1e-9 and abs(y) <= HALF_LENGTH + 1e-9
    net = net_quad()
    assert net.shape == (4, 3)
    assert np.allclose(net[:, 1], 0.0)
    assert net[:, 2].max() == pytest.approx(1.07)
    rect = court_rect(margin_x=1.0, margin_y=2.0)
    assert rect[2, 0] == pytest.approx(HALF_DOUBLES + 1.0)
    assert rect[2, 1] == pytest.approx(HALF_LENGTH + 2.0)
