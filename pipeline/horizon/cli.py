"""`horizon` command line. Sub-commands live in horizon/commands/*.py and are discovered automatically."""

from __future__ import annotations

import argparse
import importlib
import pkgutil

import horizon.commands as commands_pkg
from horizon.env import load_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horizon", description="Project Horizon Tennis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for info in sorted(pkgutil.iter_modules(commands_pkg.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"horizon.commands.{info.name}")
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)
