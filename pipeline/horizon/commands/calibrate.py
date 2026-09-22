"""`horizon calibrate`: click court keypoints on a frame and solve the broadcast camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from horizon.calibration import (
    MAX_ACCEPTABLE_RMS_PX,
    draw_court_overlay,
    draw_keypoint_legend,
    save_calibration,
    solve_calibration,
)
from horizon.court import KEYPOINTS
from horizon.paths import MatchPaths
from horizon.video import read_frame

WINDOW = "horizon calibrate"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calibrate", help="Solve the broadcast camera from court keypoints")
    p.add_argument("--match-id", required=True)
    p.add_argument("--frame", type=int, default=0, help="Frame to click on")
    p.add_argument("--keypoints", type=Path, default=None, help="JSON {name: [u, v]} (or a calibration.json) instead of clicking")
    p.set_defaults(handler=run)


def _text(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def collect_keypoints(frame_rgb: np.ndarray) -> dict[str, tuple[float, float]]:
    """Interactive OpenCV window: click each named keypoint (S skip, U undo, Enter solve, Esc quit)."""
    names = list(KEYPOINTS)
    clicks: dict[str, tuple[float, float]] = {}
    state = {"index": 0}

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and state["index"] < len(names):
            clicks[names[state["index"]]] = (float(x), float(y))
            state["index"] += 1

    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            canvas = base.copy()
            for name, (x, y) in clicks.items():
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), 2, cv2.LINE_AA)
                _text(canvas, name, (int(x) + 6, int(y) - 6), 0.4)
            current = names[state["index"]] if state["index"] < len(names) else None
            draw_keypoint_legend(canvas, current)
            prompt = f"Click {current}" if current else "All points placed"
            _text(canvas, f"{prompt} | S skip  U undo  Enter solve ({len(clicks)}/4+)  Esc quit", (12, 24))
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(clicks) >= 4:
                return clicks
            if key in (ord("s"), ord("S")) and state["index"] < len(names):
                state["index"] += 1
            elif key in (ord("u"), ord("U")) and state["index"] > 0:
                state["index"] -= 1
                clicks.pop(names[state["index"]], None)
            elif key == 27:
                raise SystemExit("Calibration cancelled")
    finally:
        cv2.destroyWindow(WINDOW)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    frame = read_frame(paths.source_video, args.frame)
    height, width = frame.shape[:2]
    if args.keypoints:
        data = json.loads(Path(args.keypoints).read_text())
        keypoints = {k: (float(v[0]), float(v[1])) for k, v in data.get("keypoints", data).items()}
    else:
        keypoints = collect_keypoints(frame)
    calibration = solve_calibration(keypoints, width, height)
    save_calibration(calibration, paths.calibration)
    preview = draw_court_overlay(frame, calibration.camera, calibration.keypoints)
    cv2.imwrite(str(paths.calibration_preview), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    center = np.round(calibration.camera.center, 2).tolist()
    print(
        f"Saved {paths.calibration}: {len(calibration.keypoints)} keypoints, RMS {calibration.rms_px:.2f} px, "
        f"focal {calibration.camera.fy:.0f} px, camera at {center} m"
    )
    print(f"Check the overlay image: {paths.calibration_preview}")
    if calibration.rms_px > MAX_ACCEPTABLE_RMS_PX:
        print("WARNING: reprojection error above 4 px; re-run and click more carefully.")
        return 2
    return 0
