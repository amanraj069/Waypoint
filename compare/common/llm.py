"""One chat interface for every rebuilt system, over whichever LLM is configured.

The three papers each talk to a different LLM stack (PEACE: Ollama with a JSON
schema; AutoTAMP: the OpenAI chat API with a fixed system message; DroneServer:
any tool-calling model). The rebuilds keep each paper's messages exactly and
send them through this one interface, so the model can be held constant across
systems, or set to the one a paper used.

    client = default_client()                 # OpenAI if OPENAI_API_KEY is set, else Gemini
    reply = client.chat(messages)             # reply.text
    reply = client.chat(messages, json_mode=True)
    reply = client.chat(messages, tools=[...])   # reply.tool_calls

Keys and model ladders come from Waypoint's own configuration
(`waypoint.planner.config`), and every call is added to Waypoint's token
accounting (`waypoint.planner.llm_planner.USAGE`), so a comparison batch
reports what each system spent the same way.

`ScriptedChat` replays canned replies; every test uses it, so the suite never
dials out.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 180.0
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_DEFAULT_ENDPOINT = "http://localhost:11434"

Message = dict[str, Any]


class LLMError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


@dataclass
class ToolCall:
    """One function call the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]
    #: The argument text exactly as the model wrote it (it may not be valid JSON).
    raw: str = ""
    #: Why `raw` did not parse, when it did not; `arguments` is then empty.
    error: str | None = None


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    finish_reason: str = ""

    def as_message(self) -> Message:
        """This reply as the assistant message to append to a conversation."""
        message: Message = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw or json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class ChatClient(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        schema: dict | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply: ...


# -- usage ---------------------------------------------------------------------

USAGE_FIELDS = ("calls", "input", "cached", "output", "reasoning")
#: Every call made through this module, per "provider:model".
USAGE: dict[str, dict[str, int]] = {}


def usage_snapshot() -> dict[str, dict[str, int]]:
    return {model: dict(counts) for model, counts in USAGE.items()}


def usage_since(before: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """What was spent since `before`, a `usage_snapshot()`: only models that were used."""
    spent = {}
    for model, counts in USAGE.items():
        then = before.get(model, {})
        delta = {k: counts.get(k, 0) - then.get(k, 0) for k in USAGE_FIELDS}
        if delta["calls"]:
            spent[model] = delta
    return spent


def usage_total(usage: dict[str, dict[str, int]]) -> dict[str, int]:
    total = dict.fromkeys(USAGE_FIELDS, 0)
    for counts in usage.values():
        for key in USAGE_FIELDS:
            total[key] += counts.get(key, 0)
    return total


def _count(label: str, usage: dict[str, int]) -> None:
    for book in _books():
        counts = book.setdefault(label, dict.fromkeys(USAGE_FIELDS, 0))
        counts["calls"] += 1
        for key in ("input", "cached", "output", "reasoning"):
            counts[key] += int(usage.get(key, 0) or 0)


def _books() -> list[dict]:
    """This module's ledger, and Waypoint's when it is importable, so one report covers both."""
    books = [USAGE]
    try:
        from waypoint.planner import llm_planner  # noqa: PLC0415

        books.append(llm_planner.USAGE)
    except Exception:  # noqa: BLE001 -- the rebuilds also run without Waypoint's planner
        pass
    return books


# -- OpenAI-compatible endpoints (OpenAI itself, Gemini, a local server) -------


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIChat:
    """Chat completions against OpenAI or any endpoint that speaks its API.

    `models` is a ladder: an overloaded, rate-limited, missing or unreachable
    model shifts down to the next, as Waypoint's planner does.
    """

    def __init__(
        self,
        models: Iterable[str],
        *,
        api_key: str,
        base_url: str | None = None,
        provider: str = "openai",
        reasoning_effort: str | None = None,
        client: Any = None,
    ):
        self.models = tuple(models)
        if not self.models:
            raise LLMError("no model given")
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        #: Sent with every request that offers tools (e.g. parallel_tool_calls, tool_choice); None sends nothing.
        self.tool_protocol: dict[str, Any] | None = None
        #: A schema strict mode cannot express is still sent, non-strict, instead of falling back to bare JSON mode.
        self.loose_schema = False
        self.name = f"{provider}:{self.models[0]}"
        if client is None:
            from openai import OpenAI  # noqa: PLC0415

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=1, timeout=REQUEST_TIMEOUT_S)
        self._client = client

    def chat(self, messages, *, json_mode=False, schema=None, tools=None, temperature=None, max_tokens=None) -> Reply:
        import openai  # noqa: PLC0415

        attempts = []
        for position, model in enumerate(self.models):
            request: dict[str, Any] = {"model": model, "messages": messages}
            reasoning = self.provider == "openai" and _is_reasoning_model(model)
            if reasoning:
                # Reasoning models take no sampling temperature, and count thinking against the output cap.
                if self.reasoning_effort:
                    request["reasoning_effort"] = self.reasoning_effort
            else:
                if temperature is not None:
                    request["temperature"] = temperature
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
            if tools:
                request["tools"] = tools
                request.update(self.tool_protocol or {})
            elif schema is not None and self.provider == "openai" and _strict_ok(schema):
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "strict": True, "schema": schema},
                }
            elif schema is not None and self.provider == "openai" and self.loose_schema:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "strict": False, "schema": schema},
                }
            elif json_mode or schema is not None:
                request["response_format"] = {"type": "json_object"}
            began = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**request)
            except openai.OpenAIError as exc:
                attempts.append(f"{model}: {type(exc).__name__}: {exc}")
                if position == len(self.models) - 1 or not _should_shift(exc):
                    raise LLMError(f"{self.provider} could not answer.\n  " + "\n  ".join(attempts)) from exc
                log.warning("%s unavailable (%s); shifting down to %s", model, type(exc).__name__, self.models[position + 1])
                continue
            return self._reply(completion, model, time.perf_counter() - began)
        raise LLMError("no model was tried")  # pragma: no cover

    def _reply(self, completion: Any, model: str, seconds: float) -> Reply:
        choice = completion.choices[0]
        message = choice.message
        if getattr(message, "refusal", None):
            raise LLMError(f"{model} refused: {message.refusal}")
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            raw = call.function.arguments or ""
            arguments, error = _parse_arguments(raw)
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments, raw=raw, error=error))
        usage = _usage_of(getattr(completion, "usage", None))
        _count(f"{self.provider}:{model}", usage)
        return Reply(
            text=message.content or "",
            tool_calls=calls,
            model=model,
            usage=usage,
            seconds=round(seconds, 3),
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )


def _strict_ok(schema: dict) -> bool:
    """Whether a schema meets OpenAI's strict mode (every object closed); free-form dicts do not."""

    def closed(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "object" and node.get("additionalProperties") is not False:
                return False
            return all(closed(value) for value in node.values())
        if isinstance(node, list):
            return all(closed(value) for value in node)
        return True

    return closed(schema)


def _parse_arguments(raw: str) -> tuple[dict, str | None]:
    if not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        return {}, f"arguments are not valid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, "arguments are not a JSON object"
    return value, None


def _usage_of(usage: Any) -> dict[str, int]:
    if usage is None:
        return dict.fromkeys(("input", "cached", "output", "reasoning"), 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    output_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "output": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning": int(getattr(output_details, "reasoning_tokens", 0) or 0),
    }


def _should_shift(exc: BaseException) -> bool:
    import openai  # noqa: PLC0415

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return getattr(exc, "status_code", None) in {404, 408, 429, 500, 502, 503, 504}


# -- Ollama, as PEACE calls it -------------------------------------------------


class OllamaChat:
    """Ollama's native /api/chat, with `format` set to a JSON schema as PEACE sends it."""

    def __init__(self, model: str, *, endpoint: str = OLLAMA_DEFAULT_ENDPOINT, api_key: str = "", timeout: float = 120.0):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def available_models(self) -> list[str]:
        import httpx  # noqa: PLC0415

        try:
            response = httpx.get(f"{self.endpoint}/api/tags", headers=self._headers, timeout=5.0)
            response.raise_for_status()
            return [m.get("model") or m.get("name") for m in response.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return []

    @property
    def is_ready(self) -> bool:
        return any(self.model in name for name in self.available_models())

    def chat(self, messages, *, json_mode=False, schema=None, tools=None, temperature=None, max_tokens=None) -> Reply:
        import httpx  # noqa: PLC0415

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if schema is not None:
            body["format"] = schema
        elif json_mode:
            body["format"] = "json"
        if tools:
            body["tools"] = tools
        began = time.perf_counter()
        try:
            response = httpx.post(f"{self.endpoint}/api/chat", json=body, headers=self._headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Ollama chat failed: {exc}") from exc
        message = payload.get("message") or {}
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            calls.append(
                ToolCall(id=f"call_{index}", name=function.get("name", ""), arguments=dict(arguments), raw=json.dumps(arguments))
            )
        usage = {
            "input": int(payload.get("prompt_eval_count") or 0),
            "cached": 0,
            "output": int(payload.get("eval_count") or 0),
            "reasoning": 0,
        }
        _count(self.name, usage)
        return Reply(
            text=message.get("content") or "",
            tool_calls=calls,
            model=self.model,
            usage=usage,
            seconds=round(time.perf_counter() - began, 3),
            finish_reason=payload.get("done_reason") or "",
        )


# -- scripted, for tests -------------------------------------------------------


class ScriptedChat:
    """Replies from a list, or from a function of the messages; records what it was asked."""

    name = "scripted"

    def __init__(self, replies: Iterable[str | Reply | Callable[[list[Message]], str | Reply]] | Callable[[list[Message]], str | Reply]):
        self._function = replies if callable(replies) else None
        self._replies = [] if callable(replies) else list(replies)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages, *, json_mode=False, schema=None, tools=None, temperature=None, max_tokens=None) -> Reply:
        self.requests.append(
            {
                "messages": [dict(m) for m in messages],
                "json_mode": json_mode,
                "schema": schema,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self._function is not None:
            answer = self._function(messages)
        else:
            if not self._replies:
                raise LLMError("the scripted model has no reply left")
            answer = self._replies.pop(0)
            if callable(answer):
                answer = answer(messages)
        reply = answer if isinstance(answer, Reply) else Reply(text=str(answer))
        reply.model = reply.model or "scripted"
        return reply

    @property
    def calls(self) -> int:
        return len(self.requests)


# -- the configured client -----------------------------------------------------


def default_client(model: str | None = None) -> ChatClient:
    """The LLM the rest of Waypoint is configured to use: OpenAI first, then Gemini.

    `model` picks one model instead of the configured ladder: a bare name is
    looked up on the first configured provider, and `ollama:<name>`,
    `openai:<name>` or `gemini:<name>` name the provider too.
    """
    from waypoint.planner import config  # noqa: PLC0415 -- also loads .env

    if model and model.startswith("ollama:"):
        import os  # noqa: PLC0415

        return OllamaChat(
            model.split(":", 1)[1],
            endpoint=os.environ.get("OLLAMA_ENDPOINT", OLLAMA_DEFAULT_ENDPOINT),
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
        )
    provider = None
    if model and model.split(":", 1)[0] in ("openai", "gemini") and ":" in model:
        provider, model = model.split(":", 1)
    providers = config.get_providers()
    if provider is None:
        if not providers:
            raise LLMError(
                f"No LLM is configured: set {config.OPENAI_KEY_VAR} or {config.API_KEY_VAR}1 in .env (see .env.example)."
            )
        provider = providers[0]
    if provider == "openai":
        key = config.get_openai_key()
        if not key:
            raise LLMError(f"{config.OPENAI_KEY_VAR} is not set")
        ladder = (model,) if model else config.get_openai_model_ladder()
        return OpenAIChat(ladder, api_key=key, provider="openai", reasoning_effort=config.get_reasoning_effort())
    key = config.get_api_key()
    if not key:
        raise LLMError(f"{config.API_KEY_VAR}1 is not set")
    ladder = (model,) if model else config.get_model_ladder()
    return OpenAIChat(ladder, api_key=key, base_url=GEMINI_OPENAI_BASE_URL, provider="gemini")
