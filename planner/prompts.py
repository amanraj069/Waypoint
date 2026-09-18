"""Prompt and response-schema construction for the hierarchical planner.

The plan never comes back as prose. The request pins the response to
`application/json` with a strict schema (`additionalProperties: false`,
everything required), so the decoded body maps onto the `Plan` model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from planner.models import Action, MissionConstraints, Plan

if TYPE_CHECKING:  # avoids a cycle: world imports planner.models
    from world.models import World

PLAN_SCHEMA = {
    "type": "object",
    "description": (
    "The finished hierarchical flight plan. The subtasks are flown in the "
    "order given, and the waypoints within each subtask are flown in the "
    "order given."
    ),
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "description": (
                "Ordered mission phases. Each is one coherent piece of the "
                "mission, e.g. takeoff, transit, survey, return, land."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short snake_case phase name, e.g. 'takeoff', "
                            "'survey_north_field', 'return_to_base'."
                        ),
                    },
                    "waypoints": {
                        "type": "array",
                        "description": "Ordered waypoints flown during this phase.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {
                                    "type": "number",
                                    "description": "East position in metres, local ENU frame.",
                                },
                                "y": {
                                    "type": "number",
                                    "description": "North position in metres, local ENU frame.",
                                },
                                "z": {
                                    "type": "number",
                                    "description": (
                                        "Altitude above ground in metres. Must be within "
                                        "the min/max altitude limits, except 0 for a "
                                        "waypoint on the ground."
                                    ),
                                },
                                "speed": {
                                    "type": "number",
                                    "description": (
                                        "Ground speed in m/s. Strictly greater than 0 and "
                                        "no greater than the max velocity limit."
                                    ),
                                },
                                "action": {
                                    "type": "string",
                                    "enum": [a.value for a in Action],
                                    "description": (
                                        "MOVE to fly to the point, HOVER to hold there, "
                                        "CAPTURE to collect sensor data there, LAND to "
                                        "touch down there."
                                    ),
                                },
                            },
                            "required": ["x", "y", "z", "speed", "action"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "waypoints"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a flight planner for a small autonomous multirotor UAV. You turn a \
natural-language mission into a concrete, safe, hierarchical flight plan.

Work in two stages, in this order:

Stage 1 - decompose. Read the mission and break it into a short ordered list of \
named subtasks: the phases the aircraft flies through. A typical mission is 3 to \
6 subtasks. Every plan begins with a takeoff phase and ends with a landing phase, \
because the aircraft starts and finishes on the ground at the launch point you \
are given, which is not always the origin.

Stage 2 - expand. Give each subtask its concrete waypoints: position, speed and \
action. Waypoints are absolute positions in a local ENU frame in metres, where \
+x is east, +y is north and z is altitude above ground.

Safety is your responsibility, not someone else's. A downstream verifier will \
reject the plan, but you should be planning inside the envelope on your own:

- Never place a waypoint outside the geofence polygon. Leave margin at the edges.
- Keep every airborne waypoint between the minimum and maximum altitude. Only a \
takeoff or landing waypoint may sit at z = 0.
- Never command a speed above the maximum velocity, and never command 0 or a \
negative speed. Use a slower speed while capturing data than while transiting.
- Keep the whole route short enough to fly within the battery budget, and always \
leave enough margin to get home and land. Roughly, the aircraft uses about 1% of \
battery per 40 m flown plus about 1% per 30 s of hovering.
- Never put a waypoint inside a no-fly zone, and never let the straight line \
between two consecutive waypoints clip one. Route around a zone with extra \
waypoints rather than flying through it.
- If an environment is described, fly in it: z is height above the launch pad, \
not above the ground below, so over rising ground or a building you must raise z \
to keep clearance. Either climb above every roof before crossing a built-up area, \
or route between the buildings -- never straight through one.
- Keep the stated separation from every listed obstacle, both at the waypoints \
and along the legs between them. Standing off sideways or flying over the top \
both work, as long as the clearance is there.
- Do not chain waypoints into hairpin turns at transit speed. The faster the \
aircraft is flying, the wider the turn has to be: slow down through a sharp \
corner, or insert an intermediate waypoint to open it up.

Rules for the plan itself:

- Use CAPTURE for waypoints where the mission asks for imagery, inspection, \
survey or sensor data.
- Use HOVER for holding station, e.g. while observing something.
- The very last waypoint of the plan must be a LAND at the takeoff point, at z = 0.
- Do not invent constraints that were not given, and do not refuse a mission \
because it is underspecified. Choose sensible values and plan it.

Respond with a single JSON object matching the required schema. Do not write \
a prose plan, and do not wrap the JSON in markdown fences or commentary.

Worked example. For the mission "photograph the comms tower 80 m east of base, \
then come home", with a geofence covering (-100,-100) to (200,200), altitude \
between 0 and 120 m and max speed 12 m/s, a good plan is:

  subtask "takeoff": one waypoint (0, 0, 25) at 3 m/s, MOVE
  subtask "transit_to_tower": one waypoint (70, 0, 25) at 8 m/s, MOVE
  subtask "inspect_tower": waypoints (80, 0, 25) at 2 m/s CAPTURE, then \
(80, 10, 25) at 2 m/s CAPTURE
  subtask "return_to_base": one waypoint (0, 0, 25) at 8 m/s, MOVE
  subtask "land": one waypoint (0, 0, 0) at 1.5 m/s, LAND

Note how it stands off from the structure, slows down to capture, climbs before \
transiting, and returns to the origin before descending.\
"""


def describe_home(home: tuple[float, float, float] | None) -> str:
    """Where the aircraft is standing, in the words the plan has to use."""
    x, y, z = home or (0.0, 0.0, 0.0)
    return f"({x:g}, {y:g}, {z:g})"


def describe_pads(
    home: tuple[float, float, float] | None,
    landing: tuple[float, float, float] | None,
) -> str:
    """Where it starts and where it must end, which are not always the same."""
    pad = describe_home(home)
    if landing is None or landing == home:
        return (
            f"The aircraft starts on the ground at {pad} and must finish landed "
            f"there -- the first waypoint climbs from {pad} and the last is a LAND "
            f"back at it."
        )
    return (
        f"The aircraft starts on the ground at {pad} and must finish landed "
        f"somewhere else: at {describe_home(landing)}. The first waypoint climbs "
        f"from {pad}, and the last waypoint is a LAND at exactly "
        f"{describe_home(landing)} -- that spot has been checked and is flat "
        f"enough to set down on, so land there and nowhere else."
    )


def build_plan_prompt(
    mission_text: str,
    constraints: MissionConstraints,
    world: "World | None" = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
) -> str:
    """The user turn for a fresh plan."""
    environment = f"Environment you are flying in:\n{world.describe()}\n\n" if world else ""
    return (
        f"Mission: {mission_text}\n\n"
        f"Operating constraints:\n{constraints.describe()}\n\n"
        f"{environment}"
        f"{describe_pads(home, landing)} Every coordinate is in the same frame as "
        "those points. Decompose this mission into named subtasks, then give each "
        "subtask its waypoints, and return the result as JSON."
    )


def build_replan_prompt(
    mission_text: str,
    previous_plan: Plan,
    violation_report: list[str],
    constraints: MissionConstraints,
    world: "World | None" = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
) -> str:
    """The user turn for a corrective replan after the verifier rejected a plan."""
    violations = "\n".join(f"  {i}. {v}" for i, v in enumerate(violation_report, 1))
    rendered = _render_plan(previous_plan)
    environment = f"Environment you are flying in:\n{world.describe()}\n\n" if world else ""
    return (
        f"Mission: {mission_text}\n\n"
        f"Operating constraints:\n{constraints.describe()}\n\n"
        f"{environment}"
        f"{describe_pads(home, landing)}\n\n"
        "You previously submitted this plan:\n\n"
        f"{rendered}\n\n"
        "The safety verifier REJECTED it for these reasons:\n"
        f"{violations}\n\n"
        "Produce a corrected plan that still accomplishes the mission. Fix every "
        "listed violation. Change only what you need to change -- keep the parts "
        "of the plan that were already safe and already served the mission. Do "
        "not simply shorten the mission to make the violations go away unless "
        "there is no other way to fly it safely. Return the corrected plan as JSON."
    )


def _render_plan(plan: Plan) -> str:
    lines = []
    for subtask in plan.subtasks:
        lines.append(f'  subtask "{subtask.name}":')
        for wp in subtask.waypoints:
            lines.append(
                f"    ({wp.x:g}, {wp.y:g}, {wp.z:g}) at {wp.speed:g} m/s, {wp.action.value}"
            )
    return "\n".join(lines) if lines else "  (empty plan)"
