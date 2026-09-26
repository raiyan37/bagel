"""`horizon doctor`: report tool availability and which secrets are configured (never their values)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from horizon.paths import resolve_data_root

SECRETS = (
    "AWS_REGION",
    "TWELVELABS_MODEL_ID",
    "OPENROUTER_API_KEY",
    "GEMINI_MODEL_ID",
    "BACKBOARD_API_KEY",
)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("doctor", help="Check the pipeline environment")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    import imageio_ffmpeg

    print(f"python: {sys.version.split()[0]}")
    print(f"ffmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}")
    for module in ("torch", "transformers", "ultralytics", "viser", "boto3"):
        status = "installed" if importlib.util.find_spec(module) else "MISSING"
        print(f"{module}: {status}")
    if importlib.util.find_spec("torch"):
        import torch

        print(f"cuda: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only'})")
    for name in SECRETS:
        print(f"{name}: {'set' if os.getenv(name) else 'missing'}")
    print(f"data root: {resolve_data_root()}")
    return 0
