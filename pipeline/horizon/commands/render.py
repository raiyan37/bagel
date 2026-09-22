"""`horizon render`: first-person POV videos for the players."""

from __future__ import annotations

import argparse

from horizon.paths import MatchPaths
from horizon.players import Players
from horizon.pov import POV_HEIGHT, POV_HFOV_DEG, POV_WIDTH, render_pov_clip
from horizon.reconstruct import Scene


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("render", help="Render each player's first-person POV video")
    p.add_argument("--match-id", required=True)
    p.add_argument("--roles", nargs="+", choices=("near", "far"), default=["near", "far"])
    p.add_argument("--width", type=int, default=POV_WIDTH)
    p.add_argument("--height", type=int, default=POV_HEIGHT)
    p.add_argument("--fov", type=float, default=POV_HFOV_DEG, help="Horizontal field of view in degrees")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id)
    scene = Scene.load(paths.scene)
    players = Players.load(paths.players)
    for role in args.roles:
        out = paths.pov_video(role)
        count = render_pov_clip(scene, players, role, out, args.fov, args.width, args.height, log=print)
        print(f"Wrote {out} ({count} frames)")
    return 0
