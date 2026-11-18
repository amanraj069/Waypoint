"""Whether a planned trajectory does what its instruction asked. These checkers are this rebuild's.

"We automatically check resulting trajectories via hard-coded checkers" (paper,
section IV), but the checkers are not in the release. The ten HouseWorld
instruction families are (ten paraphrases each; the first of each is quoted):

 1  "Proceed towards the green bedroom, and then make your way to the yellow bathroom."
 2  "...mustard-colored lavatory, followed by ... the rosy-hued sleeping quarters. Make sure to
    explore both the culinary space and the eating area. Conclude ... in the crimson washroom."
 3  family 1, "but it's important not to disturb the livingroom at all."
 4  "...the room painted in cyan, proceed into the bedroom and linger there for 5 seconds, and then
    ... the restroom. Be sure to avoid contact with the two areas colored in true blue."
 5  "Go to two spaces that possess a hue closely resembling red, but not exactly the intense, pure
    shade of red."
 6  "Explore all the chambers that are neither light pink nor cyan."
 7  "Before stepping into any of the bathrooms, make sure you visit the kitchen first. Venture into
    the red and yellow rooms, but refrain from going into any room that's blue."
 8  "Firstly, transition to the kitchen, afterwards advance to the bedroom ensuring you avoid all
    rooms painted blue. Ensure you remain in the bedroom for a minimum of 10 seconds, and then
    navigate back to the kitchen through the living room."
 9  "Make sure you check out all the rooms, but don't step foot in the pink room until you've been
    to the living room."
10  "Be sure to enter every room at least once. However, before proceeding to any other room, make
    sure to visit the living room. Ultimately, your journey should conclude in the smaller blue room."

Each checker below is a reading of its family, written before any model was
run. A trajectory is judged from the rooms it is inside over time: it is
sampled densely along its straight segments, and a room counts as entered when
a sample lies strictly inside the room's box (the test the release uses for its
state sequence). Every case must end at a given end position across the house,
which no path reaches from the red restroom without crossing another room, so
family 2's "conclude in" is read as an order (the red restroom after the yellow
restroom and the bedroom); family 10's is the last room entered, which the
house does allow.

HouseWorld2 adds the paper's hard time constraint: the trajectory must also
finish within the case's time limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compare.autotamp.envs import houseworld

ROOMS: dict[str, tuple[float, float, float, float]] = {
    box[-1]: (-box[0], box[1], -box[2], box[3])
    for colour, boxes in houseworld().items()
    if colour != "black"
    for box in boxes
}
ALL_ROOMS = tuple(ROOMS)
BLUE = ("LivingRoom", "ExerciseRoom")
SAMPLE_STEP = 0.01


@dataclass
class Visit:
    room: str
    enter: float
    leave: float

    @property
    def stay(self) -> float:
        return self.leave - self.enter


def room_at(x: float, y: float) -> str | None:
    for name, (x0, x1, y0, y1) in ROOMS.items():
        if x0 < x < x1 and y0 < y < y1:
            return name
    return None


def visits(trajectory: list) -> list[Visit]:
    """The rooms the trajectory is inside, in order, with when it entered and left each."""
    out: list[Visit] = []
    current: Visit | None = None
    for (xa, ya, ta), (xb, yb, tb) in zip(trajectory, trajectory[1:]):
        steps = max(1, math.ceil(math.hypot(xb - xa, yb - ya) / SAMPLE_STEP))
        for k in range(steps + 1):
            share = k / steps
            x, y, t = xa + (xb - xa) * share, ya + (yb - ya) * share, ta + (tb - ta) * share
            room = room_at(x, y)
            if current is not None and room == current.room:
                current.leave = t
                continue
            if current is not None:
                out.append(current)
            current = Visit(room, t, t) if room is not None else None
    if current is not None:
        out.append(current)
    return out


def _first(vs: list[Visit], room: str, after: float = -math.inf, stay: float = 0.0) -> Visit | None:
    return next((v for v in vs if v.room == room and v.enter >= after and v.stay >= stay - 1e-6), None)


def _in_order(vs: list[Visit], steps: list[tuple[tuple[str, ...], float]]) -> bool:
    """Each step (any of these rooms, staying at least this long) is found after the one before it."""
    t = -math.inf
    for names, stay in steps:
        hits = [v for name in names if (v := _first(vs, name, t, stay)) is not None]
        if not hits:
            return False
        t = min(v.enter for v in hits)
    return True


def _never(vs: list[Visit], rooms: tuple[str, ...], until: float = math.inf) -> bool:
    return not any(v.room in rooms and v.enter < until for v in vs)


def _all(vs: list[Visit], rooms: tuple[str, ...]) -> bool:
    seen = {v.room for v in vs}
    return all(room in seen for room in rooms)


def check_houseworld(family: int, trajectory: list | None) -> bool:
    """Whether `trajectory` ([[x, y, t], ...]) satisfies instruction family 1 to 10."""
    if not trajectory or len(trajectory) < 2:
        return False
    return check_visits(family, visits(trajectory))


def check_visits(family: int, vs: list[Visit]) -> bool:
    """Whether a sequence of room visits satisfies instruction family 1 to 10."""
    if not vs:
        return False
    if family == 1:
        return _in_order(vs, [(("MasterBedroom",), 0), (("RestRoom",), 0)])
    if family == 2:
        return _in_order(vs, [(("RestRoom",), 0), (("Bedroom",), 0), (("RestRoom2",), 0)]) and _all(
            vs, ("Kitchen", "DiningRoom")
        )
    if family == 3:
        return _in_order(vs, [(("MasterBedroom",), 0), (("RestRoom",), 0)]) and _never(vs, ("LivingRoom",))
    if family == 4:
        return _in_order(
            vs, [(("Kitchen",), 0), (("Bedroom", "MasterBedroom"), 5.0), (("RestRoom", "RestRoom2"), 0)]
        ) and _never(vs, BLUE)
    if family == 5:
        return _all(vs, ("Bedroom", "DiningRoom"))
    if family == 6:
        return _all(vs, tuple(r for r in ALL_ROOMS if r not in ("Bedroom", "Kitchen")))
    if family == 7:
        kitchen = _first(vs, "Kitchen")
        return (
            kitchen is not None
            and _never(vs, ("RestRoom", "RestRoom2"), until=kitchen.enter)
            and _all(vs, ("RestRoom", "RestRoom2"))
            and _never(vs, BLUE)
        )
    if family == 8:
        kitchen = _first(vs, "Kitchen")
        if kitchen is None:
            return False
        bedroom = min(
            (v for name in ("Bedroom", "MasterBedroom") if (v := _first(vs, name, kitchen.enter, 10.0)) is not None),
            key=lambda v: v.enter,
            default=None,
        )
        if bedroom is None or not _never(vs, BLUE, until=bedroom.enter):
            return False
        return _in_order([v for v in vs if v.enter >= bedroom.leave], [(("LivingRoom",), 0), (("Kitchen",), 0)])
    if family == 9:
        living = _first(vs, "LivingRoom")
        return _all(vs, ALL_ROOMS) and living is not None and _never(vs, ("Bedroom",), until=living.enter)
    if family == 10:
        return _all(vs, ALL_ROOMS) and vs[0].room == "LivingRoom" and vs[-1].room == "ExerciseRoom"
    raise ValueError(f"no checker for family {family}")


def within_time(trajectory: list | None, limit: float | None) -> bool:
    return bool(trajectory) and (limit is None or trajectory[-1][2] <= limit + 1e-6)
