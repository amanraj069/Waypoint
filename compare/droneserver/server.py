"""The DroneServer MCP server, on the simulated vehicle (upstream: app.py, server.py, tools/).

A real MCP server, built with the official MCP Python SDK's FastMCP as
upstream's is. Every one of the 98 released tools is registered through the
safety guard: "There is no registration path that bypasses the guard - that is
the point: a tool added without touching the safety code is still validated,
tiered, and audited" (upstream app.py).

    server = DroneServer(vehicle)                    # control-scope key configured
    async with server.client() as mcp:               # an MCP client session
        tools = await mcp.list_tools()               # the 98 tool schemas
        result = await mcp.call_tool("takeoff", {"takeoff_altitude": 20})

What differs from upstream's deployment: the transport is the SDK's in-memory
MCP session, not HTTP over a private network, and the API key is presented by
the session, not in a header. The paper's network measures (no public ports, a
tailnet) are about that deployment and are not rebuilt.
"""

from __future__ import annotations

import inspect
import json
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from compare.droneserver import clock
from compare.droneserver.safety.config import SafetySettings
from compare.droneserver.safety.middleware import SafetyLayer, Session, guard
from compare.droneserver.tools.core import CATALOG, implemented, unsupported
from compare.droneserver.vehicle import Vehicle

CONTROL_KEY = "harness-control-key"
TELEMETRY_KEY = "recorder-telemetry-key"
#: Names the released defaults refer to: float("nan"), and the offboard module's stale-setpoint window.
_DEFAULT_NAMES = {"float": float, "DEFAULT_STALE_TIMEOUT_S": 15.0}
_ANNOTATION_NAMES = {"float": float, "int": int, "str": str, "bool": bool, "list": list, "dict": dict, "None": None, "Any": Any}


def control_settings(**overrides: Any) -> SafetySettings:
    """Upstream's default limits, with one control-scope key and one telemetry-scope key configured.

    With no key configured upstream grants read-only scope and nothing can be
    commanded, so every flying deployment configures keys; these are the keys
    the benchmark harness and its recorder use.
    """
    overrides.setdefault("api_keys", f"harness:{CONTROL_KEY}:control,recorder:{TELEMETRY_KEY}:telemetry")
    return SafetySettings(**overrides)


def _signature(name: str) -> inspect.Signature:
    """A tool's released parameters, from the catalogue."""
    parameters = []
    for spec in CATALOG[name]["params"]:
        annotation = eval(spec["annotation"], {"__builtins__": {}}, _ANNOTATION_NAMES)  # noqa: S307 -- our own data file
        default = (
            inspect.Parameter.empty
            if spec["required"]
            else eval(spec["default"], {"__builtins__": {}}, _DEFAULT_NAMES)  # noqa: S307
        )
        parameters.append(
            inspect.Parameter(spec["name"], inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        )
    return inspect.Signature(parameters, return_annotation=dict)


class DroneServer:
    """One MCP server commanding one aircraft, with its safety layer."""

    def __init__(
        self,
        vehicle: Vehicle,
        settings: SafetySettings | None = None,
        *,
        api_key: str | None = CONTROL_KEY,
        client_name: str | None = None,
        audit_path: str | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.settings = settings or control_settings()
        self.layer = SafetyLayer(self.settings, audit_path)
        self.session = Session(
            vehicle=vehicle, layer=self.layer, api_key=api_key, model=client_name, session_launch=vehicle.session_launch
        )
        # The safety layer's windows (token lifetime, rate limit, take-off settling) run on flight time.
        # Offset from zero, as a monotonic clock is: the state cache marks itself stale by stamping 0.
        clock.use(lambda: 1.0e6 + vehicle.stack.now)
        self.mcp = FastMCP("MAVLink MCP")
        self.implemented = implemented(vehicle, fence=lambda: self.layer.fence(self.settings))
        #: The server-side mission runner (the release's RUNNER), shared by the managed-mission tools.
        self.runner = self.implemented.pop("_runner")
        for name in CATALOG:
            fn = self.implemented.get(name) or unsupported(name)
            released = _signature(name)
            if name in self.implemented:
                own = inspect.signature(fn)
                if list(own.parameters) != list(released.parameters):
                    raise RuntimeError(f"{name} does not have its released parameters: {own} vs {released}")
            fn.__signature__ = released  # type: ignore[attr-defined]
            fn.__doc__ = CATALOG[name]["doc"]
            self.mcp.tool()(guard(fn, self.session))

    @asynccontextmanager
    async def client(self):
        """An MCP client session connected to this server."""
        async with create_connected_server_and_client_session(self.mcp._mcp_server) as session:
            yield session

    async def call(self, session, name: str, arguments: dict | None = None) -> dict:
        """One tools/call, with the result as the dictionary the tool returned."""
        reply = await session.call_tool(name, arguments or {})
        return result_of(reply)

    @property
    def audit(self) -> list[dict]:
        return self.layer.records


def result_of(reply) -> dict:
    """An MCP tool result as the dictionary the tool returned; a protocol error as a status "error"."""
    text = "\n".join(getattr(block, "text", "") for block in reply.content if getattr(block, "text", None))
    if reply.isError:
        return {"status": "error", "error": text or "the tool call failed"}
    structured = getattr(reply, "structuredContent", None)
    if isinstance(structured, dict) and "result" not in structured:
        return structured
    try:
        value = json.loads(text)
    except ValueError:
        return {"status": "error", "error": f"the tool returned something that is not JSON: {text[:200]}"}
    return value if isinstance(value, dict) else {"status": "success", "result": value}
