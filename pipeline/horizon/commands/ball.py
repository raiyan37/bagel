"""`horizon ball`: find the ball in 2D and fit its 3D trajectory."""

from __future__ import annotations

import argparse

from horizon.ball import ball_candidates, build_ball_track
from horizon.calibration import load_calibration
from horizon.paths import MatchPaths
from horizon.players import Players
from horizon.tracking import Detections


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("ball", help="Track the tennis ball and fit its 3D flight path")
    p.add_argument("--match-id", required=True)
    p.add_argument("--samples", type=int, default=24, help="Frames used to build the clean plate")
    p.add_argument("--diff", type=int, default=28, help="Sum-of-channels difference from the plate")
    p.add_argument("--yellow", type=int, default=25, help="Minimum G - B for a ball pixel")
    p.add_argument("--max-rms", type=float, default=3.0, help="Reprojection RMS (px) a flight must fit within")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    players = Players.load(paths.players)
    candidates = ball_candidates(
        paths.source_video,
        players.frame_count,
        Detections.load(paths.detections),
        players,
        samples=args.samples,
        diff_threshold=args.diff,
        yellow_threshold=args.yellow,
        log=print,
    )
    track = build_ball_track(
        load_calibration(paths.calibration).camera,
        candidates,
        fps=players.fps,
        frame_count=players.frame_count,
        max_rms_px=args.max_rms,
    )
    track.save(paths.ball)
    print(
        f"Wrote {paths.ball}: ball on {track.detected_frames}/{track.frame_count} frames "
        f"across {len(track.segments)} flights, reprojection RMS {track.reprojection_rms_px:.2f} px"
    )
    if track.detected_frames == 0:
        print("No ball found. Try lowering --yellow or --diff, or check that this is a blue hard court.")
    return 0
