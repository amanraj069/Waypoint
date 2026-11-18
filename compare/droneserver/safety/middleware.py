"""The guard pipeline: every tool call passes through here before the vehicle (upstream: safety/middleware.py).

Check order (fixed, and the order the audit ``rule`` fields will show):

1. **authenticate** - resolve the presented API key to a client + scope
2. **state**        - refresh the vehicle snapshot (only when a rule needs it)
3. **tier**         - base tier + conditional escalation for these arguments
4. **authorize**    - client scope vs. tier
5. **rate limit**   - per client, separate budget for critical calls
6. **confirmation** - critical tools need a valid single-use token
7. **bounds**       - parameter bounds (altitude, speed, coordinates, size)
8. **geofence**     - server-side fence: single targets and whole missions
9. **preconditions**- vehicle-state rules incl. the takeoff settling window and,
   when state cannot be read, the energy-direction failsafe
10. **execute**     - the tool itself
11. **record**      - command history + append-only audit line

Failing any check returns a normal tool result with ``status="rejected"`` (or
``"confirmation_required"``), never an exception: the LLM sees a readable
explanation with a ``remedy`` and can correct itself. An exception inside the
guard refuses the call (fail closed).

The order, the rules it calls (all vendored unchanged), the failure results,
the audit record and the `confirm_token` schema addition are upstream's. Two
things are adapted to this rebuild:

* Upstream keeps one process-wide layer (`LAYER`) and finds the API key and
  the drone in the MCP request context (HTTP headers, lifespan). Here a
  `SafetyLayer` belongs to one `Session` (one server, one aircraft, one
  presented key), so several servers can run in one process and the in-memory
  MCP transport, which has no headers, still presents a key.
* The launch-point re-anchor for back-to-back trials on one long-lived
  simulator is left out: every trial here starts on a fresh aircraft.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from compare.droneserver.safety import auth as auth_mod
from compare.droneserver.safety.audit import AuditLog, AuditRecord, new_call_id
from compare.droneserver.safety.config import SafetySettings
from compare.droneserver.safety.geofence import Geofence, parse_polygon
from compare.droneserver.safety.state import StateTracker
from compare.droneserver.safety.tiers import ESCALATIONS, TOOL_TIERS, Tier, effective_tier
from compare.droneserver.safety.tokens import (
    CONFIRM_ARG,
    ConfirmationStore,
    confirmation_failed_result,
    confirmation_required_result,
)
from compare.droneserver.safety.validation import (
    ENERGY_ADDING_TOOLS,
    MISSION_START_TOOLS,
    MISSION_UPLOAD_TOOLS,
    NAVIGATION_TOOLS,
    RateLimiter,
    check_geofence,
    check_parameter_bounds,
    check_preconditions,
)

logger = logging.getLogger("compare.droneserver")

#: Tools whose rules consult vehicle state (others skip the telemetry refresh).
_STATE_DEPENDENT = (
    NAVIGATION_TOOLS
    | MISSION_START_TOOLS
    | MISSION_UPLOAD_TOOLS
    | ENERGY_ADDING_TOOLS
    | {
        "takeoff",
        "disarm_drone",
        "arm_drone",
        "land",
        "reposition",
        "do_orbit",
        "flight_altitudes",
        "return_to_launch",
        "set_flight_mode",
        "clear_geofence",
        "upload_geofence",
        "raw_geofence_transfer",
        "import_qgc_mission",
        "calibrate",
        "cancel_calibration",
    }
)


class SafetyLayer:
    """Holds the layer's mutable state (limiter, tokens, vehicle state, log)."""

    def __init__(self, settings: SafetySettings, audit_path: str | None = None) -> None:
        self.settings = settings
        self.rate_limiter = RateLimiter()
        self.confirmations = ConfirmationStore()
        self.state_tracker = StateTracker()
        self._audit = AuditLog(audit_path) if audit_path else None
        #: Every audit record, in memory too, whether or not a file is written.
        self.records: list[dict] = []
        #: Measured duration of the previous durable audit write (ms).
        self.last_audit_write_ms: float = 0.0

    def fence(self, s: SafetySettings) -> Geofence:
        try:
            polygon = parse_polygon(s.geofence_polygon)
        except ValueError as e:
            logger.error("Invalid SAFETY_GEOFENCE_POLYGON, fence polygon disabled: %s", e)
            polygon = ()
        return Geofence(
            polygon=polygon,
            max_altitude_m=s.geofence_max_altitude_m,
            max_radius_m=s.geofence_max_radius_m,
            home=self.state_tracker.state.home,
        )

    def reset(self) -> None:
        self.rate_limiter.reset()
        self.confirmations.clear()
        self.state_tracker.reset()


@dataclass
class Session:
    """One server's link to one aircraft, and the client talking to it."""

    vehicle: Any
    layer: SafetyLayer
    #: The API key the client presents (upstream: the X-API-Key or Authorization header).
    api_key: str | None = None
    #: What the client reported itself as at MCP initialize.
    model: str | None = None
    #: Where this session started: the one ground elevation that does not move with the aircraft.
    session_launch: dict | None = None
    extras: dict = field(default_factory=dict)


def _guards_in_force(s: SafetySettings) -> dict:
    return {
        "enabled": s.enabled,
        "validation": s.validation_enabled,
        "geofence": s.geofence_enabled,
        "tiers": s.tiers_enabled,
        "auth": s.auth_enabled,
        "rate_limit": s.rate_limit_enabled,
    }


def _evaluate(tool: str, args: dict, session: Session, s: SafetySettings) -> tuple[dict | None, Tier, object, dict]:
    """Run checks 1-9. Returns (rejection_result_or_None, tier, client, state)."""
    layer = session.layer
    client = auth_mod.authenticate(session.api_key, s)

    state: dict = {"unknown": True}
    if tool in _STATE_DEPENDENT:
        state = layer.state_tracker.refresh(session.vehicle, s.state_cache_ttl_s, session_launch=session.session_launch)
    else:
        layer.state_tracker.note_session_launch(session.session_launch)
        state = layer.state_tracker.state.snapshot()

    tier, consequence = effective_tier(tool, args, state)

    # 4. authorize
    if s.auth_enabled and not client.can(tier):
        return auth_mod.authorization_failed_result(client, tool, tier), tier, client, state

    # 5. rate limit (emergency stop is deliberately exempt)
    if s.rate_limit_enabled and tier is not Tier.EMERGENCY:
        rejection = layer.rate_limiter.check(client.client_id, tier is Tier.CRITICAL, s)
        if rejection is not None:
            return rejection.as_result(), tier, client, state

    # 6. confirmation token for critical tools
    if s.tiers_enabled and tier is Tier.CRITICAL:
        token = args.get(CONFIRM_ARG)
        if not token:
            pending = layer.confirmations.issue(
                client.client_id, tool, args, consequence, s.confirmation_ttl_s, s.confirmation_max_outstanding
            )
            return confirmation_required_result(pending, tool), tier, client, state
        ok, reason = layer.confirmations.redeem(str(token), client.client_id, tool, args)
        if not ok:
            return confirmation_failed_result(tool, reason), tier, client, state

    # 7. parameter bounds, then 8. geofence, then 9. state preconditions.
    # Argument-intrinsic violations are reported BEFORE state-dependent ones:
    # a target outside the fence stays illegal no matter how long you wait.
    if s.validation_enabled:
        rejection = check_parameter_bounds(tool, args, s, state)
        if rejection is not None:
            return rejection.as_result(), tier, client, state

    if s.geofence_enabled:
        rejection = check_geofence(tool, args, layer.fence(s), s, state)
        if rejection is not None:
            return rejection.as_result(), tier, client, state

    if s.validation_enabled:
        rejection = check_preconditions(tool, args, state, s)
        if rejection is not None:
            return rejection.as_result(), tier, client, state

    return None, tier, client, state


def _pre_execute_effects(layer: SafetyLayer, tool: str) -> None:
    """The takeoff settling window is measured from when takeoff was COMMANDED, not when it returned."""
    if tool == "takeoff":
        layer.state_tracker.note_takeoff()


def _record_effects(layer: SafetyLayer, tool: str, args: dict, result) -> None:
    """Update the command history from a successful call (check 11)."""
    status = result.get("status") if isinstance(result, dict) else None
    if status not in ("success", None):
        return
    if tool in ("land", "return_to_launch", "kill_motors", "disarm_drone", "emergency_stop"):
        layer.state_tracker.note_landed()
    elif tool in MISSION_UPLOAD_TOOLS or tool in ("import_qgc_mission", "raw_mission_control"):
        if tool != "raw_mission_control" or str(args.get("action", "")).lower() != "clear":
            layer.state_tracker.note_mission_uploaded(True)
        else:
            layer.state_tracker.note_mission_uploaded(False)
    elif tool == "clear_mission":
        layer.state_tracker.note_mission_uploaded(False)
    elif tool in ("arm_drone", "set_flight_mode"):
        layer.state_tracker.invalidate()


def can_be_critical(tool: str) -> bool:
    """True if this tool is CRITICAL always, or can escalate to CRITICAL.

    Such tools get a ``confirm_token`` parameter added to their public schema
    so the model can discover the round-trip from the tool definition itself.
    """
    return TOOL_TIERS.get(tool) is Tier.CRITICAL or tool in ESCALATIONS


def _with_confirm_token_param(fn: Callable) -> inspect.Signature:
    """The tool's signature plus a keyword-only ``confirm_token`` parameter."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if any(p.name == CONFIRM_ARG for p in params):
        return sig
    params.append(inspect.Parameter(CONFIRM_ARG, inspect.Parameter.KEYWORD_ONLY, default="", annotation=str))
    return sig.replace(parameters=params)


CRITICAL_DOC = (
    "\n\n    SAFETY: this is a CRITICAL action. Calling it without confirm_token returns a\n"
    "    single-use token plus a consequence statement; repeat the call with that exact\n"
    "    token and unchanged arguments to execute. Never invent or reuse a token.\n"
)


def guard(fn: Callable, session: Session) -> Callable:
    """Wrap a tool coroutine with the safety pipeline. Applied at registration, so no tool escapes it."""
    tool_name = fn.__name__

    @functools.wraps(fn)
    async def wrapper(*fn_args, **kwargs):
        started = time.perf_counter()
        layer = session.layer
        s = layer.settings

        if not s.enabled:
            # A guardrails-off run must still be self-documenting: audit the call, flagged.
            return await _run_unguarded(fn, fn_args, kwargs, tool_name, session, s, started)

        call_args = dict(kwargs)
        safety_started = time.perf_counter()
        guard_error: str | None = None
        try:
            rejection, tier, client, _state = _evaluate(tool_name, call_args, session, s)
        except Exception as e:  # noqa: BLE001
            # FAIL CLOSED. An exception inside the guard means we do not know
            # whether this call is safe, so we refuse it.
            logger.exception("safety layer error on %s: %s", tool_name, e)
            guard_error = f"{type(e).__name__}: {e}"
            tier, client = Tier.CRITICAL, auth_mod.ANONYMOUS
            rejection = {
                "status": "rejected",
                "error": (
                    "The safety layer failed while evaluating this call, so the call was "
                    "refused. This is a server fault, not a problem with your request."
                ),
                "rule": "guard.internal_error",
                "remedy": (
                    "Retry once. If it fails again, stop commanding the drone and tell the "
                    "operator - the safety layer needs attention. Read-only telemetry tools "
                    "are unaffected by the failing check only if they also succeed."
                ),
                "safety_layer": "droneserver.safety",
            }
        safety_ms = (time.perf_counter() - safety_started) * 1000

        record = AuditRecord(
            call_id=new_call_id(),
            client_id=client.client_id,
            authenticated=client.authenticated,
            key_fp=client.key_fingerprint,
            model=session.model,
            tool=tool_name,
            tier=tier.value,
            args=call_args,
            verdict="allowed",
            guards=_guards_in_force(s),
            safety_ms=safety_ms,
            guard_error=guard_error,
        )

        if rejection is not None:
            record.verdict = rejection.get("status", "rejected")
            record.rule = rejection.get("rule")
            record.outcome_error = rejection.get("error")
            record.latency_ms = (time.perf_counter() - started) * 1000
            _write_audit(layer, record, s)
            return rejection

        # 10. execute - the token argument is plumbing, not a tool parameter
        kwargs.pop(CONFIRM_ARG, None)
        _pre_execute_effects(layer, tool_name)
        try:
            result = await fn(*fn_args, **kwargs)
        except Exception as e:
            record.verdict = "error"
            record.outcome_error = str(e)
            record.latency_ms = (time.perf_counter() - started) * 1000
            _write_audit(layer, record, s)
            raise

        try:
            _record_effects(layer, tool_name, call_args, result)
        except Exception:  # noqa: BLE001
            logger.exception("safety layer failed to record command effects")

        if isinstance(result, dict):
            record.outcome_status = result.get("status")
            record.outcome_error = result.get("error")
        record.latency_ms = (time.perf_counter() - started) * 1000
        _write_audit(layer, record, s)
        return result

    if can_be_critical(tool_name):
        # Publish confirm_token in the tool's schema so the model can see the round-trip exists.
        wrapper.__signature__ = _with_confirm_token_param(fn)  # type: ignore[attr-defined]
        wrapper.__doc__ = (fn.__doc__ or "") + CRITICAL_DOC
    return wrapper


async def _run_unguarded(fn, fn_args, kwargs, tool_name: str, session: Session, s: SafetySettings, started: float):
    """Execute a tool with the safety layer disabled - but still audit it."""
    record = AuditRecord(
        call_id=new_call_id(),
        client_id="safety_disabled",
        authenticated=False,
        key_fp="",
        model=session.model,
        tool=tool_name,
        tier="unclassified",
        args=dict(kwargs),
        verdict="allowed_safety_disabled",
        guards=_guards_in_force(s),
        safety_ms=0.0,
    )
    kwargs.pop(CONFIRM_ARG, None)
    try:
        result = await fn(*fn_args, **kwargs)
    except Exception as e:
        record.verdict = "error"
        record.outcome_error = str(e)
        record.latency_ms = (time.perf_counter() - started) * 1000
        _write_audit(session.layer, record, s)
        raise
    if isinstance(result, dict):
        record.outcome_status = result.get("status")
        record.outcome_error = result.get("error")
    record.latency_ms = (time.perf_counter() - started) * 1000
    _write_audit(session.layer, record, s)
    return result


def _write_audit(layer: SafetyLayer, record: AuditRecord, s: SafetySettings) -> None:
    """Write the record and measure the durable-write cost (reported on the NEXT record, as upstream)."""
    if not s.audit_enabled:
        return
    import json  # noqa: PLC0415

    record.audit_write_ms = layer.last_audit_write_ms
    write_started = time.perf_counter()
    try:
        layer.records.append(json.loads(record.to_json()))
        if layer._audit is not None:
            layer._audit.write(record)
    except Exception:  # noqa: BLE001
        logger.exception("failed to write audit record")
    finally:
        layer.last_audit_write_ms = round((time.perf_counter() - write_started) * 1000, 3)
