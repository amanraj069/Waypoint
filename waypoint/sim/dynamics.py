"""Point-mass flight dynamics.

The whole flight model is here, and it is deliberately small:

    desired = where the controller wants the velocity to be, given how far the
              waypoint is and how fast it was told to fly
    step    = accelerate toward that velocity within the airframe's limits,
              add the wind, and integrate one tick of motion

No teleporting: velocity changes by at most `max_acceleration * dt` per tick,
so the aircraft accelerates out of a waypoint and decelerates into the next.
The approach speed is capped at sqrt(2 * a * distance) -- the fastest the
aircraft could still stop in the distance remaining -- which is what makes it
settle on a waypoint instead of orbiting it.
"""

from __future__ import annotations

import math

from waypoint.sim.config import SimConfig, Vector

EPS = 1e-12


def subtract(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v: Vector, factor: float) -> Vector:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def norm(v: Vector) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def clamp_norm(v: Vector, limit: float) -> Vector:
    """Shorten `v` to `limit` if it is longer, keeping its direction."""
    magnitude = norm(v)
    if magnitude <= limit or magnitude <= EPS:
        return v
    return scale(v, limit / magnitude)


def desired_velocity(
    position: Vector,
    target: Vector,
    cruise_speed: float,
    config: SimConfig,
) -> Vector:
    """The velocity the controller asks for this tick.

    Points straight at the target, at the commanded speed, slowing down near
    the target so the aircraft can stop on it. The `min_approach_speed` floor
    keeps a headwind from stalling the approach short of the waypoint.
    """
    delta = subtract(target, position)
    distance = norm(delta)
    if distance <= EPS:
        return (0.0, 0.0, 0.0)

    stopping_speed = math.sqrt(2.0 * config.max_acceleration * distance)
    speed = min(cruise_speed, max(stopping_speed, min(config.min_approach_speed, cruise_speed)))
    velocity = scale(delta, speed / distance)

    # Climb and descent are slower than horizontal flight on a multirotor.
    if abs(velocity[2]) > config.max_climb_rate:
        velocity = (velocity[0], velocity[1], math.copysign(config.max_climb_rate, velocity[2]))
    return velocity


def step(
    position: Vector,
    velocity: Vector,
    desired: Vector,
    wind: Vector,
    config: SimConfig,
    speed_limit: float | None = None,
) -> tuple[Vector, Vector]:
    """Integrate one tick, returning the new (position, velocity).

    `velocity` and the returned velocity are ground velocity -- what a GPS would
    measure -- so the wind is part of it. `desired` is a *ground* velocity: the
    autopilot cancels the `wind_compensation` fraction of the wind it senses and
    commands the rest through the air, so the wind it misses pushes the aircraft
    off the straight line and a gust it has not caught up with pushes hardest.
    """
    air_velocity = subtract(velocity, wind)
    commanded = subtract(desired, scale(wind, config.wind_compensation))
    change = clamp_norm(subtract(commanded, air_velocity), config.max_acceleration * config.dt)
    air_velocity = clamp_norm(add(air_velocity, change), speed_limit or config.max_velocity)

    ground_velocity = add(air_velocity, wind)
    new_position = add(position, scale(ground_velocity, config.dt))
    return new_position, ground_velocity


def heading_of(velocity: Vector, previous: float = 0.0) -> float:
    """Bearing in degrees [0, 360), holding the old heading when hovering."""
    if math.hypot(velocity[0], velocity[1]) <= 1e-6:
        return previous
    return math.degrees(math.atan2(velocity[1], velocity[0])) % 360.0


def distance(a: Vector, b: Vector) -> float:
    return norm(subtract(a, b))
