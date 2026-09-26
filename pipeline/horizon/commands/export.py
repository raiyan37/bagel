"""`horizon export`: publish a processed match into public/matches for the web app."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizon.export import PUBLIC_MATCHES, VIEWER_URL, export_match
from horizon.paths import MatchPaths


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("export", help="Copy videos + manifest + tracks into public/matches/<id>")
    p.add_argument("--match-id", required=True)
    p.add_argument("--competition", default="Tennis")
    p.add_argument("--viewer-url", default=VIEWER_URL, help="Where `horizon view` serves the 3D free camera")
    p.add_argument("--public", type=Path, default=PUBLIC_MATCHES, help="public/matches directory of the web app")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    target = export_match(MatchPaths.for_match(args.match_id), args.public, args.competition, args.viewer_url)
    print(f"Exported to {target}; open http://localhost:5173 after `npm run dev`")
    return 0
