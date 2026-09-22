"""Gemini (through OpenRouter) finds both players in one frame as normalised box_2d boxes.

Gemini's native box format is [ymin, xmin, ymax, xmax] on a 0-1000 grid. The soccer prototype asked for pixels
and got 0-1000 values back (y=721 in a 480-px-high frame), so we ask for the native format and convert.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import requests

from horizon.llm.jsonutil import loads_lenient

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.5-flash"

_BOX = {
    "type": "object",
    "properties": {"box_2d": {"type": "array", "items": {"type": "number"}}},
    "required": ["box_2d"],
    "additionalProperties": False,
}
BOXES_SCHEMA = {
    "type": "object",
    "properties": {"near_player": _BOX, "far_player": _BOX},
    "required": ["near_player", "far_player"],
    "additionalProperties": False,
}


def build_prompt(near_appearance: str, far_appearance: str) -> str:
    return (
        "Detect the two tennis players in this broadcast frame.\n"
        f"NEAR player (on the half of the court closest to the camera): {near_appearance or 'no description'}.\n"
        f"FAR player (on the other side of the net): {far_appearance or 'no description'}.\n"
        "Ignore ball kids, line judges, the chair umpire and spectators. "
        "Return one tight bounding box per player as box_2d = [ymin, xmin, ymax, xmax] normalised to 0-1000."
    )


def build_request(image_jpeg: bytes, near_appearance: str, far_appearance: str, model: str) -> dict:
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(near_appearance, far_appearance)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "player_boxes", "strict": True, "schema": BOXES_SCHEMA},
        },
    }


def box_2d_to_xyxy(box, width: int, height: int) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"box_2d must have 4 numbers, got {box!r}")
    ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) for v in box)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Degenerate box_2d {box!r}")
    return (xmin / 1000.0 * width, ymin / 1000.0 * height, xmax / 1000.0 * width, ymax / 1000.0 * height)


def parse_response(body: dict, width: int, height: int) -> dict[str, tuple[float, float, float, float]]:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    data = loads_lenient(content)
    return {
        "near": box_2d_to_xyxy(data["near_player"]["box_2d"], width, height),
        "far": box_2d_to_xyxy(data["far_player"]["box_2d"], width, height),
    }


def locate_players(
    frame_rgb: np.ndarray,
    near_appearance: str,
    far_appearance: str,
    model: str | None = None,
    session=None,
    api_key: str | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to pipeline/.env)")
    model = model or os.getenv("GEMINI_MODEL_ID") or DEFAULT_MODEL
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    session = session or requests.Session()
    response = session.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=build_request(jpeg.tobytes(), near_appearance, far_appearance, model),
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {response.text[:500]}")
    height, width = frame_rgb.shape[:2]
    return parse_response(response.json(), width, height)
