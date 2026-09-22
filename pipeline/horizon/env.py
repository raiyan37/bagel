"""Load pipeline/.env without overriding variables already set in the shell."""

from __future__ import annotations

from dotenv import load_dotenv

from horizon.paths import PIPELINE_ROOT


def load_env() -> None:
    load_dotenv(PIPELINE_ROOT / ".env", override=False)
