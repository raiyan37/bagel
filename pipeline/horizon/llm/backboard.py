"""Backboard.io REST client and the tool-calling loop that orchestrates Pegasus + Gemini."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

BASE_URL = "https://app.backboard.io/api"


class BackboardClient:
    def __init__(self, api_key: str | None = None, session=None, base_url: str = BASE_URL, timeout: float = 180):
        self.api_key = api_key or os.getenv("BACKBOARD_API_KEY")
        if not self.api_key:
            raise RuntimeError("BACKBOARD_API_KEY is not set (add it to pipeline/.env or use --orchestrator direct)")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        response = self.session.post(
            f"{self.base_url}{path}",
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Backboard {path} failed ({response.status_code}): {response.text[:500]}")
        return response.json()

    def create_assistant(self, name: str, system_prompt: str, tools: list[dict]) -> str:
        data = self._post("/assistants", {"name": name, "system_prompt": system_prompt, "tools": tools})
        return str(data.get("assistant_id") or data.get("id"))

    def create_thread(self, assistant_id: str) -> str:
        return str(self._post(f"/assistants/{assistant_id}/threads", {})["thread_id"])

    def send_message(
        self, thread_id: str, content: str, *, llm_provider: str | None = None, model_name: str | None = None, memory: str = "Auto"
    ) -> dict:
        payload: dict = {"content": content, "stream": False, "memory": memory}
        if llm_provider:
            payload["llm_provider"] = llm_provider
        if model_name:
            payload["model_name"] = model_name
        return self._post(f"/threads/{thread_id}/messages", payload)

    def submit_tool_outputs(self, thread_id: str, run_id: str, tool_outputs: list[dict]) -> dict:
        return self._post(f"/threads/{thread_id}/runs/{run_id}/submit-tool-outputs", {"tool_outputs": tool_outputs})


@dataclass
class ToolExecution:
    name: str
    arguments: dict
    output: dict | None = None
    error: str | None = None


@dataclass
class ToolLoopResult:
    content: str
    executions: list[ToolExecution] = field(default_factory=list)


def _parse_arguments(function: dict) -> dict:
    raw = function.get("parsed_arguments", function.get("arguments"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def run_tool_loop(
    client,
    thread_id: str,
    content: str,
    handlers: dict[str, Callable[[dict], dict]],
    *,
    llm_provider: str | None = None,
    model_name: str | None = None,
    max_rounds: int = 6,
) -> ToolLoopResult:
    """Send `content`, execute every requested tool locally, submit outputs, repeat until no tool calls remain."""
    response = client.send_message(thread_id, content, llm_provider=llm_provider, model_name=model_name)
    executions: list[ToolExecution] = []
    for _ in range(max_rounds):
        calls = response.get("tool_calls") or []
        if not calls:
            break
        run_id = response.get("run_id")
        if not run_id:
            raise RuntimeError("Backboard requested tool calls without a run_id")
        outputs = []
        for call in calls:
            function = call.get("function") or {}
            execution = ToolExecution(name=function.get("name", ""), arguments=_parse_arguments(function))
            handler = handlers.get(execution.name)
            if handler is None:
                execution.error = f"unknown tool {execution.name!r}"
            else:
                try:
                    execution.output = handler(execution.arguments)
                except Exception as exc:  # the failure is reported back to the assistant
                    execution.error = f"{type(exc).__name__}: {exc}"
            executions.append(execution)
            payload = execution.output if execution.error is None else {"error": execution.error}
            outputs.append({"tool_call_id": call.get("id"), "output": json.dumps(payload)})
        response = client.submit_tool_outputs(thread_id, run_id, outputs)
    return ToolLoopResult(content=str(response.get("content") or ""), executions=executions)
