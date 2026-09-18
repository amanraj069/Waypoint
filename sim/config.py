"""Airframe model and wind model for the simulator.

Both are plain pydantic models, so a scenario can be written in JSON and
replayed exactly. Nothing here is random unless you ask for turbulence, and
even that is seeded.
"""

from __future__ import annotations

import math
import random

from pydantic import BaseModel, Field

Vector = tuple[float, float, float]


class SimConfig(BaseModel):
    """How the simulated aircraft behaves.

    The flight model is a point mass: the controller picks a target velocity
    each tick and the airframe accelerates toward it within `max_acceleration`.
    Nothing teleports, and nothing exceeds `max_velocity` (the mission's limit
    overrides this when one is given).
    """

    dt: float = Field(default=0.1, gt=0, description="Tick length in seconds.")
    max_acceleration: float = Field(default=3.0, gt=0, description="m/s^2, any axis.")
    max_velocity: float = Field(default=15.0, gt=0, description="Airframe speed ceiling, m/s.")
    max_climb_rate: float = Field(default=4.0, gt=0, description="Vertical speed cap, m/s.")

    arrival_radius: float = Field(
        default=1.5, gt=0, description="A waypoint counts as reached inside this radius."
    )
    min_approach_speed: float = Field(
        default=1.0,
        gt=0,
        description="Floor on the commanded closing speed, so wind cannot stall an approach.",
    )
    wind_compensation: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the wind the autopilot senses and cancels. What it misses "
            "blows the aircraft off the straight line; 0 models a UAV flying blind."
        ),
    )
    hover_seconds: float = Field(default=2.0, ge=0, description="Dwell at a HOVER waypoint.")
    capture_seconds: float = Field(default=0.0, ge=0, description="Dwell at a CAPTURE waypoint.")

    meters_per_battery_pct: float = Field(
        default=40.0, gt=0, description="Metres flown per 1% of battery."
    )
    seconds_per_battery_pct: float = Field(
        default=60.0, gt=0, description="Baseline hover/avionics draw: seconds per 1%."
    )
    start_battery_pct: float = Field(default=100.0, gt=0, le=100)

    collisions: bool = Field(
        default=True,
        description="Whether hitting terrain or a building ends the flight.",
    )
    max_leg_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Safety valve: give up on a leg after this long and move on.",
    )
    home: Vector = (0.0, 0.0, 0.0)

    @property
    def ticks_per_second(self) -> float:
        return 1.0 / self.dt


class Wind(BaseModel):
    """A wind vector that may vary slowly, plus optional seeded turbulence.

    `at(t)` is a pure function of time: two runs of the same scenario see
    exactly the same wind.
    """

    vector: Vector = (0.0, 0.0, 0.0)
    variation: Vector = Field(
        default=(0.0, 0.0, 0.0),
        description="Amplitude added as a sine over `period`, for a slowly varying wind.",
    )
    period: float = Field(default=30.0, gt=0, description="Seconds per variation cycle.")
    turbulence: float = Field(
        default=0.0, ge=0, description="Std-dev of per-tick random gustiness, m/s."
    )
    seed: int = Field(default=0, description="Seed for turbulence, so runs replay identically.")

    def at(self, t: float) -> Vector:
        """The wind vector at time `t`, in m/s."""
        phase = math.sin(2.0 * math.pi * t / self.period)
        wind = [
            base + amplitude * phase
            for base, amplitude in zip(self.vector, self.variation)
        ]
        if self.turbulence:
            # Keyed on the time so the noise is reproducible rather than
            # dependent on how many times we have been called.
            rng = random.Random(f"{self.seed}:{round(t * 1000)}")
            wind = [component + rng.gauss(0.0, self.turbulence) for component in wind]
        return (wind[0], wind[1], wind[2])

    @property
    def is_still(self) -> bool:
        return not any(self.vector) and not any(self.variation) and not self.turbulence


NO_WIND = Wind()
DEFAULT_CONFIG = SimConfig()
