import io
import json

import numpy as np
import pytest

from horizon.llm import gemini, pegasus


class FakeBedrock:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": io.BytesIO(json.dumps(self.body).encode())}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def pegasus_message(near="Carlos Alcaraz", far="Unknown"):
    data = {
        "near_player": {"name": near, "appearance": "white shirt, black shorts"},
        "far_player": {"name": far, "appearance": "navy shirt, white cap"},
        "score": "6-4 2-1 30-15",
        "summary": "Long baseline rally won by the near player.",
    }
    return {"message": json.dumps(data), "finishReason": "stop"}


def openrouter_body(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_pegasus_request_uses_structured_output():
    request = pegasus.build_request(b"fake-video")
    assert request["mediaSource"]["base64String"] == "ZmFrZS12aWRlbw=="
    assert request["responseFormat"]["jsonSchema"]["required"] == ["near_player", "far_player", "score", "summary"]
    assert request["temperature"] == 0


def test_pegasus_rejects_clips_over_the_inline_limit():
    with pytest.raises(ValueError, match="25 MB"):
        pegasus.build_request(b"x" * (pegasus.MAX_INLINE_BYTES + 1))


def test_pegasus_parse_and_defaults():
    analysis = pegasus.parse_response(pegasus_message(far=""))
    assert analysis.near.name == "Carlos Alcaraz"
    assert analysis.far.name == "Unknown"
    assert analysis.far.appearance == "navy shirt, white cap"
    assert analysis.score == "6-4 2-1 30-15"
    with pytest.raises(ValueError):
        pegasus.parse_response({"finishReason": "stop"})


def test_analyze_clip_calls_bedrock(tmp_path):
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"video-bytes")
    fake = FakeBedrock(pegasus_message())
    analysis = pegasus.analyze_clip(clip, model_id="us.twelvelabs.pegasus-1-2-v1:0", client=fake)
    assert fake.calls[0]["modelId"] == "us.twelvelabs.pegasus-1-2-v1:0"
    assert json.loads(fake.calls[0]["body"])["inputPrompt"] == pegasus.CLIP_ANALYSIS_PROMPT
    assert analysis.to_json()["near"]["name"] == "Carlos Alcaraz"
    assert pegasus.ClipAnalysis.from_json(analysis.to_json()) == analysis


def test_box_2d_conversion_and_validation():
    assert gemini.box_2d_to_xyxy([100, 200, 500, 400], 1000, 500) == (200.0, 50.0, 400.0, 250.0)
    assert gemini.box_2d_to_xyxy([-5, 0, 1200, 1000], 100, 100) == (0.0, 0.0, 100.0, 100.0)
    with pytest.raises(ValueError):
        gemini.box_2d_to_xyxy([1, 2, 3], 10, 10)
    with pytest.raises(ValueError):
        gemini.box_2d_to_xyxy([500, 500, 400, 600], 10, 10)


def test_gemini_parse_handles_code_fences():
    data = {"near_player": {"box_2d": [600, 100, 900, 200]}, "far_player": {"box_2d": [200, 500, 300, 550]}}
    boxes = gemini.parse_response(openrouter_body("```json\n" + json.dumps(data) + "\n```"), 1000, 1000)
    assert boxes == {"near": (100.0, 600.0, 200.0, 900.0), "far": (500.0, 200.0, 550.0, 300.0)}


def test_locate_players_sends_image_and_schema():
    data = {"near_player": {"box_2d": [500, 100, 900, 200]}, "far_player": {"box_2d": [100, 600, 250, 650]}}
    session = FakeSession(FakeResponse(200, openrouter_body(json.dumps(data))))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = gemini.locate_players(frame, "white shirt", "navy shirt", model="google/gemini-3.5-flash", session=session, api_key="k")
    url, kwargs = session.calls[0]
    assert url == gemini.OPENROUTER_URL
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    payload = kwargs["json"]
    assert payload["model"] == "google/gemini-3.5-flash"
    assert payload["response_format"]["type"] == "json_schema"
    text_part, image_part = payload["messages"][0]["content"]
    assert "white shirt" in text_part["text"] and "navy shirt" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert boxes["near"] == (20.0, 50.0, 40.0, 90.0)


def test_locate_players_raises_on_http_error():
    session = FakeSession(FakeResponse(429, {"error": {"message": "rate limited"}}))
    with pytest.raises(RuntimeError, match="429"):
        gemini.locate_players(np.zeros((10, 10, 3), np.uint8), "", "", session=session, api_key="k")
