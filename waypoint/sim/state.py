"""UAV state carried through simulation, feedback and the run log."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Vector = tuple[float, float, float]


@dataclass
class UAVState:
    """One tick of truth about the aircraft.

    `x, y, z` is the *true* position. `gps_offset` is what a drifting GPS adds
    to it, so `reported_position` is what the aircraft believes -- and what the
    controller steers on. With no GPS fault the two are identical.

    `vx, vy, vz` is the ground velocity actually flown, wind included.
    """

    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    heading: float = 0.0
    battery_pct: float = 100.0
    timestamp: float = 0.0
    gps_offset: Vector = (0.0, 0.0, 0.0)
    wind: Vector = (0.0, 0.0, 0.0)
    ground: float = 0.0
    target_index: int | None = None
    subtask: str = ""
    active_faults: tuple[str, ...] = field(default_factory=tuple)

    @property
    def position(self) -> Vector:
        return (self.x, self.y, self.z)

    @property
    def velocity_vector(self) -> Vector:
        return (self.vx, self.vy, self.vz)

    @property
    def velocity(self) -> float:
        """Speed in m/s -- the magnitude of the velocity vector."""
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def ground_speed(self) -> float:
        """Horizontal speed, ignoring climb or descent."""
        return math.hypot(self.vx, self.vy)

    @property
    def reported_position(self) -> Vector:
        """Where the aircraft thinks it is: true position plus any GPS drift."""
        return (
            self.x + self.gps_offset[0],
            self.y + self.gps_offset[1],
            self.z + self.gps_offset[2],
        )

    @property
    def agl(self) -> float:
        """Height above the ground directly below, which is not the same as z."""
        return self.z - self.ground

    @property
    def gps_error(self) -> float:
        return math.sqrt(sum(component**2 for component in self.gps_offset))

    def as_record(self) -> dict:
        """The flat dict written to a run file and replayed in Gazebo."""
        record = {
            "timestamp": round(self.timestamp, 4),
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "velocity": self.velocity,
            "heading": self.heading,
            "battery_pct": self.battery_pct,
            "target_index": self.target_index,
            "subtask": self.subtask,
        }
        if self.ground:
            record["ground"] = round(self.ground, 2)
            record["agl"] = round(self.agl, 2)
        # Only carried when something actually disturbed the flight, so a clean
        # run stays small and a fault stands out in the file.
        if any(self.gps_offset):
            record["gps_offset"] = list(self.gps_offset)
            record["reported"] = list(self.reported_position)
        if any(self.wind):
            record["wind"] = list(self.wind)
        if self.active_faults:
            record["faults"] = list(self.active_faults)
        return record


# The pipeline called this MissionState before the simulator grew a velocity
# vector; the old name still works.
MissionState = UAVState
