"""`horizon all`: run every pipeline step for one clip."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizon.paths import MatchPaths


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("all", help="init -> calibrate -> track -> identify -> players -> reconstruct -> render -> export")
    p.add_argument("video", type=Path)
    p.add_argument("--match-id", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--orchestrator", choices=("backboard", "direct", "none"), default="backboard")
    p.add_argument("--recalibrate", action="store_true", help="Click the court keypoints again")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from horizon.cli import main as cli_main

    match = ["--match-id", args.match_id]
    init = ["init", str(args.video), *match, "--start", str(args.start)]
    if args.duration is not None:
        init += ["--duration", str(args.duration)]
    steps = [init]
    calibration = MatchPaths.for_match(args.match_id).calibration
    if args.recalibrate or not calibration.is_file():
        steps.append(["calibrate", *match])
    else:
        print(f"reusing {calibration} - pass --recalibrate if this is a different camera or clip")
    steps += [
        ["track", *match],
        ["identify", *match, "--orchestrator", args.orchestrator],
        ["players", *match],
        ["reconstruct", *match],
        ["render", *match],
        ["export", *match],
    ]
    for step in steps:
        print(f"== horizon {step[0]}")
        code = cli_main(step)
        if code != 0:
            print(f"Stopped: `horizon {step[0]}` exited with {code}")
            return code
    return 0
