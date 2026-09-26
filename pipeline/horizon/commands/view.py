"""`horizon view`: interactive 3D viewer with the free camera (Viser)."""

from __future__ import annotations

import argparse

from horizon.paths import MatchPaths


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("view", help="Open the 3D free-camera viewer (Viser)")
    p.add_argument("--match-id", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from horizon.viewer import run_viewer

    run_viewer(MatchPaths.for_match(args.match_id), host=args.host, port=args.port)
    return 0
