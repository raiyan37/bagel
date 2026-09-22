"""ITF tennis court model in metres (see spec §5.1 for the world frame)."""

from __future__ import annotations

import numpy as np

COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_LINE_FROM_NET = 6.40
NET_HEIGHT_POSTS = 1.07

HALF_LENGTH = COURT_LENGTH / 2
HALF_DOUBLES = DOUBLES_WIDTH / 2
HALF_SINGLES = SINGLES_WIDTH / 2
NET_POST_X = HALF_DOUBLES + 0.914

# Named ground keypoints, in the order the calibration tool asks for them.
# "left"/"right" are as seen from the broadcast camera behind the near baseline.
KEYPOINTS: dict[str, tuple[float, float]] = {
    "near_doubles_left": (-HALF_DOUBLES, -HALF_LENGTH),
    "near_doubles_right": (HALF_DOUBLES, -HALF_LENGTH),
    "far_doubles_right": (HALF_DOUBLES, HALF_LENGTH),
    "far_doubles_left": (-HALF_DOUBLES, HALF_LENGTH),
    "near_singles_left": (-HALF_SINGLES, -HALF_LENGTH),
    "near_singles_right": (HALF_SINGLES, -HALF_LENGTH),
    "far_singles_right": (HALF_SINGLES, HALF_LENGTH),
    "far_singles_left": (-HALF_SINGLES, HALF_LENGTH),
    "near_service_left": (-HALF_SINGLES, -SERVICE_LINE_FROM_NET),
    "near_service_center": (0.0, -SERVICE_LINE_FROM_NET),
    "near_service_right": (HALF_SINGLES, -SERVICE_LINE_FROM_NET),
    "far_service_right": (HALF_SINGLES, SERVICE_LINE_FROM_NET),
    "far_service_center": (0.0, SERVICE_LINE_FROM_NET),
    "far_service_left": (-HALF_SINGLES, SERVICE_LINE_FROM_NET),
}


def keypoint_world(name: str) -> np.ndarray:
    x, y = KEYPOINTS[name]
    return np.array([x, y, 0.0])


def court_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Painted line segments on the ground as ((x0, y0), (x1, y1))."""
    L, D, S, SL = HALF_LENGTH, HALF_DOUBLES, HALF_SINGLES, SERVICE_LINE_FROM_NET
    return [
        ((-D, -L), (D, -L)),  # near baseline
        ((-D, L), (D, L)),  # far baseline
        ((-D, -L), (-D, L)),  # left doubles sideline
        ((D, -L), (D, L)),  # right doubles sideline
        ((-S, -L), (-S, L)),  # left singles sideline
        ((S, -L), (S, L)),  # right singles sideline
        ((-S, -SL), (S, -SL)),  # near service line
        ((-S, SL), (S, SL)),  # far service line
        ((0.0, -SL), (0.0, SL)),  # centre service line
    ]


def net_quad() -> np.ndarray:
    """Net corners (4x3): bottom-left, bottom-right, top-right, top-left."""
    return np.array(
        [
            [-NET_POST_X, 0.0, 0.0],
            [NET_POST_X, 0.0, 0.0],
            [NET_POST_X, 0.0, NET_HEIGHT_POSTS],
            [-NET_POST_X, 0.0, NET_HEIGHT_POSTS],
        ]
    )


def court_rect(margin_x: float = 0.0, margin_y: float = 0.0) -> np.ndarray:
    """Ground rectangle around the doubles court (4x3), counter-clockwise from near-left."""
    x, y = HALF_DOUBLES + margin_x, HALF_LENGTH + margin_y
    return np.array([[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]])
