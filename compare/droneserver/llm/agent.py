"""The loop: model chooses, harness executes, model sees the result, repeat (upstream: llm/agent.py).

"The model is given the system prompt, the operator's request in plain
English, and the drone server's real tool list. It replies. If the reply
contains tool calls, this module executes them against the drone - in the
order the model asked for, one at a time - and hands each result straight
back. Then the model replies again. That continues until the model answers
with no tool calls (it considers the job done), or a limit is hit."

"The harness never chooses a tool, never fixes an argument, and never
completes a safety handshake on the model's behalf. If the model asks for a
tool that does not exist, or sends malformed arguments, it gets told so and
must recover. If a command is refused, the refusal goes back verbatim."

The control flow, the limits and their defaults, the two client-side
rejections and their wording, and `model_claim` are upstream's. Upstream's
per-provider adapters (`llm/providers.py`) are replaced by this repository's
one chat client, and one thing is added because the aircraft here flies in
simulated time: while the model thinks, the aircraft keeps flying for as long
as the model took (never less than `think_floor_s`), as it does in real time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from compare.common.llm import ChatClient, LLMError
from compare.droneserver.server import DroneServer, result_of


@dataclass
class Limits:
    """Where a trial stops if the model does not stop it first."""

    max_turns: int = 90
    max_tool_calls: int = 250
    wall_clock_s: float = 1800.0
    max_total_tokens: int = 8_500_000
    #: Simulated flight time, which upstream has no separate limit for: its wall clock is the flight clock.
    flight_clock_s: float = 3600.0


@dataclass
class CallRecord:
    turn: int
    seq: int
    tool: str
    arguments: dict
    status: str
    rule: str | None = None
    error: str | None = None
    confirmation_required: bool = False
    client_side_rejection: str | None = None
    flight_time_s: float = 0.0
    result: dict = field(default_factory=dict)


@dataclass
class TurnRecord:
    index: int
    decision_latency_ms: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish_reason: str
    text: str
    tool_calls: list[str] = field(default_factory=list)
    resolved_model: str = ""


@dataclass
class AgentRun:
    turns: list[TurnRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    final_text: str = ""
    duration_s: float = 0.0
    error: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def substantive_turns(self) -> int:
        """Turns in which the model actually produced something."""
        return sum(1 for t in self.turns if t.text.strip() or t.tool_calls)

    @property
    def confirmations_demanded(self) -> int:
        return sum(1 for c in self.calls if c.confirmation_required)

    @property
    def rejections(self) -> int:
        return sum(1 for c in self.calls if c.status == "rejected")

    @property
    def model_claim(self) -> str:
        """What the model said about itself: complete, aborted, or neither."""
        head = self.final_text.strip().upper()
        if head.startswith("MISSION COMPLETE"):
            return "complete"
        if head.startswith("MISSION ABORTED"):
            return "aborted"
        if "MISSION COMPLETE" in head:
            return "complete"
        if "MISSION ABORTED" in head:
            return "aborted"
        return "unstated"


def tool_specs(listed) -> list[dict]:
    """The server's MCP tool list re-encoded as a provider's function-calling catalogue."""
    return [
        {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema},
        }
        for tool in listed.tools
    ]


def harness_protocol(llm: ChatClient) -> ChatClient:
    """The model as the release's harness calls it (scripts/run_llm_missions.py, llm/providers.py).

    The harness never sets a sampling temperature or a reasoning effort ("left
    at every provider's default and never set by the harness", paper Section
    5.3), and always sends `parallel_tool_calls` (on by default) and
    `tool_choice="auto"` explicitly. A client configured for the rest of
    Waypoint carries Waypoint's own reasoning effort; this returns a copy
    without it.
    """
    import copy  # noqa: PLC0415

    from compare.common.llm import OpenAIChat  # noqa: PLC0415

    if not isinstance(llm, OpenAIChat):
        return llm
    harness = copy.copy(llm)
    harness.reasoning_effort = None
    harness.tool_protocol = {"tool_choice": "auto"}
    if llm.provider == "openai":
        harness.tool_protocol["parallel_tool_calls"] = True
    return harness


async def run_agent(
    llm: ChatClient,
    server: DroneServer,
    mcp,
    system_prompt: str,
    user_prompt: str,
    limits: Limits | None = None,
    *,
    think_floor_s: float = 1.0,
    on_event=None,
) -> AgentRun:
    """Run one mission trial. Returns everything that happened."""
    llm = harness_protocol(llm)
    limits = limits or Limits()
    tools = tool_specs(await mcp.list_tools())
    known = {t["function"]["name"] for t in tools}
    run = AgentRun()
    clock = time.perf_counter()
    deadline = clock + limits.wall_clock_s
    flight_began = server.vehicle.stack.now
    say = on_event or (lambda *a, **k: None)
    messages: list[dict] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    run.messages = messages
    turn_index = 0
    try:
        while True:
            if turn_index >= limits.max_turns:
                run.stop_reason = f"turn limit ({limits.max_turns}) reached"
                break
            if time.perf_counter() > deadline:
                run.stop_reason = f"wall-clock limit ({limits.wall_clock_s:.0f}s) reached"
                break
            if server.vehicle.stack.now - flight_began > limits.flight_clock_s:
                run.stop_reason = f"flight-clock limit ({limits.flight_clock_s:.0f}s) reached"
                break
            if run.input_tokens + run.output_tokens > limits.max_total_tokens:
                run.stop_reason = f"token budget ({limits.max_total_tokens}) exhausted"
                break

            turn_index += 1
            reply = llm.chat(messages, tools=tools)
            # The aircraft does not wait for the model.
            server.vehicle.sleep(max(reply.seconds, think_floor_s))
            record = TurnRecord(
                index=turn_index,
                decision_latency_ms=round(reply.seconds * 1000, 1),
                input_tokens=reply.usage.get("input", 0),
                cached_input_tokens=reply.usage.get("cached", 0),
                output_tokens=reply.usage.get("output", 0),
                reasoning_tokens=reply.usage.get("reasoning", 0),
                finish_reason=reply.finish_reason,
                text=reply.text,
                tool_calls=[c.name for c in reply.tool_calls],
                resolved_model=reply.model,
            )
            run.turns.append(record)
            say("turn", record)
            messages.append(reply.as_message())

            if not reply.tool_calls:
                run.final_text = reply.text
                run.stop_reason = "model declared the mission finished"
                break

            for seq, call in enumerate(reply.tool_calls, start=1):
                if len(run.calls) >= limits.max_tool_calls:
                    run.stop_reason = f"tool-call limit ({limits.max_tool_calls}) reached"
                    # Every tool call a provider was shown must be answered before the next request.
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"status": "error", "error": "the trial's tool-call limit was reached"})})
                    continue
                if call.error is not None:
                    result = {
                        "status": "error",
                        "error": (
                            f"Your arguments for {call.name} were not valid JSON ({call.error}). "
                            "Nothing was sent to the aircraft. Send the call again with well-formed arguments."
                        ),
                    }
                    call_record = _client_side(turn_index, seq, call.name, {}, "argument_parse_error", result)
                elif call.name not in known:
                    result = {
                        "status": "error",
                        "error": (
                            f"There is no tool called '{call.name}' on this server. Nothing was sent to the "
                            "aircraft. Choose a tool from the list you were given."
                        ),
                    }
                    call_record = _client_side(turn_index, seq, call.name, call.arguments, "unknown_tool", result)
                else:
                    result = result_of(await mcp.call_tool(call.name, call.arguments))
                    call_record = CallRecord(
                        turn=turn_index,
                        seq=seq,
                        tool=call.name,
                        arguments=call.arguments,
                        status=str(result.get("status", "")),
                        rule=result.get("rule"),
                        error=result.get("error"),
                        confirmation_required=result.get("status") == "confirmation_required",
                        flight_time_s=round(server.vehicle.stack.now, 1),
                        result=result,
                    )
                run.calls.append(call_record)
                say("call", call_record)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
            if run.stop_reason:
                break
    except LLMError as e:
        run.error = f"{type(e).__name__}: {e}"
        run.stop_reason = f"the model could not be reached: {e}"
    except Exception as e:  # noqa: BLE001 -- a crashed trial is a failed trial, and it says why
        run.error = f"{type(e).__name__}: {e}"
        run.stop_reason = f"harness error: {run.error}"
    run.duration_s = time.perf_counter() - clock
    if not run.stop_reason:
        run.stop_reason = "loop ended"
    return run


def _client_side(turn: int, seq: int, tool: str, arguments: dict, why: str, result: dict) -> CallRecord:
    """A call the harness answered itself because it could not be sent at all."""
    return CallRecord(
        turn=turn, seq=seq, tool=tool, arguments=arguments, status="client_rejected", error=result.get("error"),
        client_side_rejection=why, result=result,
    )
