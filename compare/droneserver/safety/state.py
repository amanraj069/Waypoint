"""Vehicle-state snapshot used by the precondition rules (upstream: safety/state.py).

The snapshot's fields and the tracker's methods are upstream's. What differs is
where the readings come from: upstream awaits MavSDK telemetry streams (with
re-requests and fallbacks for topics an autopilot stops publishing), and this
reads the same facts from the simulated vehicle, which always answers. A
vehicle whose telemetry is switched off (`vehicle.telemetry_lost`) reads as
unknown, which is what the energy-direction failsafe is tested with.

Two facts the autopilot cannot tell us are still tracked from the server's own
command history, as upstream: when take-off was commanded (the settling
window) and whether a mission was uploaded in this session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compare.droneserver import clock


@dataclass
class FlightState:
    """Server-side view of the vehicle, refreshed on demand."""

    armed: bool = False
    in_air: bool = False
    unknown: bool = True
    home: tuple[float, float] | None = None
    home_altitude_m: float | None = None
    session_launch_amsl_m: float | None = None
    position: dict | None = None
    fetched_at: float = 0.0
    takeoff_commanded_at: float | None = field(default=None)
    mission_uploaded: bool | None = field(default=None)

    def snapshot(self) -> dict:
        since = None
        if self.takeoff_commanded_at is not None:
            since = clock.monotonic() - self.takeoff_commanded_at
        return {
            "armed": self.armed,
            "in_air": self.in_air,
            "unknown": self.unknown,
            "home": self.home,
            "home_altitude_m": self.home_altitude_m,
            "session_launch_amsl_m": self.session_launch_amsl_m,
            "position": self.position,
            "seconds_since_takeoff": since,
            "mission_uploaded": self.mission_uploaded,
        }


class StateTracker:
    """Caches telemetry-derived state and records our own command history."""

    def __init__(self) -> None:
        self.state = FlightState()

    def note_session_launch(self, session_launch: dict | None) -> None:
        """Adopt the session's launch elevation. First writer wins: the datum must not follow the aircraft."""
        if self.state.session_launch_amsl_m is not None or not session_launch:
            return
        amsl = session_launch.get("absolute_altitude_m")
        if amsl is not None:
            self.state.session_launch_amsl_m = float(amsl)

    def refresh(self, vehicle, ttl_s: float, session_launch: dict | None = None) -> dict:
        """Return a state snapshot, refreshing from telemetry if stale."""
        self.note_session_launch(session_launch)
        now = clock.monotonic()
        if vehicle is None or getattr(vehicle, "telemetry_lost", False):
            self.state.unknown = True
            return self.state.snapshot()
        if (now - self.state.fetched_at) < ttl_s and not self.state.unknown:
            return self.state.snapshot()
        self.state.armed = bool(vehicle.armed)
        self.state.position = vehicle.position()
        self.state.in_air = bool(vehicle.in_air)
        self.state.unknown = False
        if self.state.home is None:
            # The fence is centred on the home first read, and stays there for the life of the server.
            home = vehicle.home()
            if home is not None:
                self.state.home = (home["latitude_deg"], home["longitude_deg"])
                self.state.home_altitude_m = home["absolute_altitude_m"]
        self.state.fetched_at = clock.monotonic()
        return self.state.snapshot()

    def note_takeoff(self) -> None:
        self.state.takeoff_commanded_at = clock.monotonic()
        self.state.fetched_at = 0.0  # force a telemetry refresh next call

    def note_landed(self) -> None:
        self.state.takeoff_commanded_at = None
        self.state.fetched_at = 0.0

    def note_mission_uploaded(self, uploaded: bool = True) -> None:
        self.state.mission_uploaded = uploaded

    def invalidate(self) -> None:
        self.state.fetched_at = 0.0

    def reset(self) -> None:
        self.state = FlightState()
