"""TwelveLabs Pegasus 1.2 video understanding via Amazon Bedrock, using structured JSON output."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horizon.llm.jsonutil import loads_lenient

MAX_INLINE_BYTES = 25 * 1024 * 1024  # Bedrock limit for mediaSource.base64String
DEFAULT_MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"

_PLAYER_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "appearance": {"type": "string"}},
    "required": ["name", "appearance"],
}

CLIP_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "near_player": _PLAYER_SCHEMA,
        "far_player": _PLAYER_SCHEMA,
        "score": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["near_player", "far_player", "score", "summary"],
}

CLIP_ANALYSIS_PROMPT = (
    "This is a broadcast tennis clip filmed from a fixed camera behind one baseline. "
    "The NEAR player is on the half of the court closest to the camera (lower part of the frame); "
    "the FAR player is on the other side of the net (upper part of the frame). "
    "For each player give their name if it appears on screen or is spoken by the commentators, otherwise 'Unknown', "
    "and describe their clothing (shirt, shorts or skirt, headwear, shoes) and hair precisely enough to pick them out "
    "in a single still frame. Ignore ball kids, line judges, the chair umpire and spectators. "
    "Report the score exactly as shown on any on-screen scoreboard (empty string if none) and summarise the point "
    "in at most two sentences."
)


@dataclass(frozen=True)
class PlayerDescription:
    name: str
    appearance: str


@dataclass(frozen=True)
class ClipAnalysis:
    near: PlayerDescription
    far: PlayerDescription
    score: str
    summary: str

    def to_json(self) -> dict:
        return {
            "near": {"name": self.near.name, "appearance": self.near.appearance},
            "far": {"name": self.far.name, "appearance": self.far.appearance},
            "score": self.score,
            "summary": self.summary,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ClipAnalysis":
        return cls(
            near=PlayerDescription(**data["near"]),
            far=PlayerDescription(**data["far"]),
            score=data["score"],
            summary=data["summary"],
        )


def build_request(video_bytes: bytes, prompt: str = CLIP_ANALYSIS_PROMPT) -> dict:
    if len(video_bytes) > MAX_INLINE_BYTES:
        raise ValueError(
            f"Clip is {len(video_bytes) / 1e6:.1f} MB; Pegasus accepts at most 25 MB inline. "
            "Re-run `horizon init` with a shorter --duration or smaller --max-height."
        )
    return {
        "inputPrompt": prompt,
        "mediaSource": {"base64String": base64.b64encode(video_bytes).decode("ascii")},
        "temperature": 0,
        "maxOutputTokens": 1024,
        "responseFormat": {"jsonSchema": CLIP_ANALYSIS_SCHEMA},
    }


def parse_response(body: dict) -> ClipAnalysis:
    message = body.get("message")
    if not isinstance(message, str):
        raise ValueError(f"Unexpected Pegasus response: {json.dumps(body)[:300]}")
    data = loads_lenient(message)

    def player(key: str) -> PlayerDescription:
        raw = data.get(key) or {}
        name = str(raw.get("name") or "").strip() or "Unknown"
        return PlayerDescription(name=name, appearance=str(raw.get("appearance") or "").strip())

    return ClipAnalysis(
        near=player("near_player"),
        far=player("far_player"),
        score=str(data.get("score") or "").strip(),
        summary=str(data.get("summary") or "").strip(),
    )


def analyze_clip(video_path: Path, region: str | None = None, model_id: str | None = None, client: Any = None) -> ClipAnalysis:
    model_id = model_id or os.getenv("TWELVELABS_MODEL_ID") or DEFAULT_MODEL_ID
    if client is None:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=region or os.getenv("AWS_REGION") or "us-east-1")
    request = build_request(Path(video_path).read_bytes())
    response = client.invoke_model(
        modelId=model_id, body=json.dumps(request), contentType="application/json", accept="application/json"
    )
    return parse_response(json.loads(response["body"].read()))
