"""Which of the loop's capabilities a mission runs with.

Every capability the closed loop has beyond plan -> verify -> fly is a switch
here, so the experiment harness can turn each off and measure what it is
worth, and so a published configuration can be named and reproduced:

    intent   read the mission into checkable requirements (`waypoint.intent`),
             refuse a mission that cannot be flown as asked, and check every
             plan against what is left of it
    repair   fix a rejected plan geometrically before asking the LLM again
             (`waypoint.verifier.repair`)
    replan   "tiered": in flight, try repair (or, for a battery shortfall,
             trimming the mission) before the LLM; "llm": always ask the LLM
    hold     "hover": the aircraft hovers, burning battery, for as long as an
             in-flight LLM replan takes; "freeze": the simulation waits
    shield   the runtime safety filter under the controller (`waypoint.sim.shield`)
    gate     "preflight": every plan is verified before it flies (the design);
             "runtime": no preflight check, only a per-command guard in flight
             (altitude clamped, a command leaving the geofence aborted), the way
             runtime-only LLM agents enforce safety; "none": nothing checks
             anything (the unconstrained baseline)

`CLASSIC` is the loop as first published and the library default, so code
that calls `run_mission` without saying keeps its behaviour. `FULL` is the
loop as designed now, and what every entry point flies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: The on/off switches: the first three always appear in a label, the other two only when off.
_SWITCHES = ("intent", "repair", "shield", "monitor_energy", "subtask_feedback")
_FLAGS = ("monitor_energy", "subtask_feedback")
REPLAN_POLICIES = ("tiered", "llm")
HOLDS = ("hover", "freeze")
GATES = ("preflight", "runtime", "none")


@dataclass(frozen=True)
class SystemOptions:
    intent: bool = True
    repair: bool = True
    replan: str = "tiered"
    hold: str = "hover"
    shield: bool = True
    gate: str = "preflight"
    #: In the air, verify new plans with the monitor's energy model (observed
    #: drain rate, landing reserve), so a plan that passes does not trip the
    #: monitor the moment it flies.
    monitor_energy: bool = True
    #: Tell the planner which subtask each violation is in, and which subtasks passed and should stay as they are.
    subtask_feedback: bool = True
    name: str = ""

    def __post_init__(self):
        if self.replan not in REPLAN_POLICIES:
            raise ValueError(f"replan must be one of {', '.join(REPLAN_POLICIES)}")
        if self.hold not in HOLDS:
            raise ValueError(f"hold must be one of {', '.join(HOLDS)}")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {', '.join(GATES)}")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        for known in PRESETS.values():
            if self == replace(known, name=self.name) and known.name:
                return known.name
        parts = [
            f"gate={self.gate}",
            "intent" if self.intent else "no-intent",
            "repair" if self.repair else "no-repair",
            f"replan={self.replan}",
            f"hold={self.hold}",
            "shield" if self.shield else "no-shield",
        ]
        # The two newer switches are named only when off, so older labels read as they did.
        parts += [f"no-{flag.replace('_', '-')}" for flag in _FLAGS if not getattr(self, flag)]
        return ",".join(parts)

    def record(self) -> dict:
        return {
            "label": self.label, "intent": self.intent, "repair": self.repair, "replan": self.replan,
            "hold": self.hold, "shield": self.shield, "gate": self.gate, "monitor_energy": self.monitor_energy,
            "subtask_feedback": self.subtask_feedback,
        }


FULL = SystemOptions(name="full")
#: The loop as first published: preflight verification and LLM corrections, nothing else.
CLASSIC = SystemOptions(
    intent=False, repair=False, replan="llm", hold="freeze", shield=False, monitor_energy=False,
    subtask_feedback=False, name="classic",
)
#: Runtime-only enforcement, as LLM agents on autopilots do it: clamp altitude, abort a fence exit, replan.
RUNTIME_GUARD = SystemOptions(
    intent=False, repair=False, replan="llm", hold="hover", shield=False, gate="runtime", monitor_energy=False,
    subtask_feedback=False,
    name="runtime-guard",
)
#: No gate at all: the LLM's plans fly as written; the monitor still replans.
NO_GATE = SystemOptions(
    intent=False, repair=False, replan="llm", hold="hover", shield=False, gate="none", monitor_energy=False,
    subtask_feedback=False,
    name="no-gate",
)

PRESETS = {preset.name: preset for preset in (FULL, CLASSIC, RUNTIME_GUARD, NO_GATE)}


def system_from(spec: str) -> SystemOptions:
    """A preset by name, optionally with switches: "full", "full,no-shield", "full,replan=llm"."""
    name, *switches = [part.strip() for part in spec.split(",") if part.strip()]
    if name not in PRESETS:
        raise ValueError(f"unknown system {name!r}; one of {', '.join(PRESETS)}")
    options = PRESETS[name]
    changes: dict = {}
    for switch in switches:
        if "=" in switch:
            key, value = switch.split("=", 1)
            if key not in ("replan", "hold", "gate"):
                raise ValueError(f"unknown setting {key!r}")
            changes[key] = value
        elif switch.startswith("no-") and switch[3:].replace("-", "_") in _SWITCHES:
            changes[switch[3:].replace("-", "_")] = False
        elif switch.replace("-", "_") in _SWITCHES:
            changes[switch.replace("-", "_")] = True
        else:
            raise ValueError(f"unknown switch {switch!r}")
    if changes:
        options = replace(options, name="", **changes)
    return options
