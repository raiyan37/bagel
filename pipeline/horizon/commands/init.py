"""`horizon init`: import a raw clip as data/<match-id>/source.mp4 + meta.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from horizon.paths import MatchPaths
from horizon.video import save_meta, transcode_clip

PEGASUS_INLINE_LIMIT = 25 * 1024 * 1024


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init", help="Import a raw broadcast tennis clip")
    p.add_argument("video", type=Path, help="Raw clip (single static broadcast shot)")
    p.add_argument("--match-id", required=True)
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="Length in seconds (default: to the end)")
    p.add_argument("--max-height", type=int, default=720)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    paths = MatchPaths.for_match(args.match_id).ensure()
    info = transcode_clip(
        args.video, paths.source_video, start_s=args.start, duration_s=args.duration, max_height=args.max_height
    )
    save_meta(
        paths.meta,
        info,
        source=str(Path(args.video).resolve()),
        start_s=args.start,
        duration_s=args.duration,
    )
    size = paths.source_video.stat().st_size
    print(
        f"Wrote {paths.source_video} ({info.width}x{info.height} @ {info.fps:.2f} fps, "
        f"{info.frame_count} frames, {size / 1e6:.1f} MB)"
    )
    if size > PEGASUS_INLINE_LIMIT:
        print("WARNING: source.mp4 is over 25 MB and Pegasus will reject it; use --duration or --max-height.")
    return 0
