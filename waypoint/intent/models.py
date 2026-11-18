"""What a mission asks for, as data the verifier can check a plan against.

The safety verifier answers "can this plan be flown without hitting
anything?". It cannot answer "does this plan do what was asked?": a plan that
safely photographs the wrong building passes it. A `MissionIntent` is the
mission written down as checkable requirements, so the second question gets a
deterministic answer too:

    targets   what must be served: photographed (CAPTURE), held over (HOVER),
              or flown over (OVERFLY), each grounded to a place in the world,
              with how many distinct views, from which sides, from above, at
              what height
    ordered   whether the targets must be served in the order listed
    avoid     places the route must not pass over at all
    landing   where the flight must end

It comes from the mission text (`waypoint.intent.extract`), is grounded
against the world's landmarks deterministically, and is checked by
`waypoint.intent.check`. Every number in it is in the plan's frame: local ENU
metres, altitudes above the datum unless a field says AGL.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class TargetAction(str, Enum):
    """How a target is served."""

    CAPTURE = "CAPTURE"
    HOVER = "HOVER"
    OVERFLY = "OVERFLY"


class Side(str, Enum):
    """A compass side to view a target from, as a bearing from the target (ENU, degrees)."""

    north = "north"
    east = "east"
    south = "south"
    west = "west"

    @property
    def bearing(self) -> float:
        return {"east": 0.0, "north": 90.0, "west": 180.0, "south": 270.0}[self.value]


class Target(BaseModel):
    """One thing the mission must serve, at a known place."""

    name: str
    x: float
    y: float
    radius: float = Field(default=5.0, gt=0, description="Rough extent in metres.")
    height: float = Field(default=0.0, ge=0, description="How far it stands up from its base.")
    base: float = Field(default=0.0, description="Ground elevation at its centre.")
    action: TargetAction = TargetAction.CAPTURE
    views: int = Field(default=1, ge=1, le=8, description="Distinct viewing directions required.")
    sides: list[Side] = Field(default_factory=list, description="Sides it must be seen from.")
    overhead: bool = Field(default=False, description="Seen from directly above.")
    altitude_agl: float | None = Field(
        default=None, gt=0, description="Height above its base to serve it from, when the mission says."
    )
    max_altitude_agl: float | None = Field(
        default=None, gt=0, description="For an OVERFLY: no higher than this over it (a low pass)."
    )
    landmark: bool = Field(default=False, description="Grounded to a named landmark of the world.")
    terrain: bool = Field(
        default=False, description="The target is the ground itself (a hill): its slopes are the subject, not in the way.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> "Target":
        if len(self.sides) > self.views:
            self.views = len(set(self.sides))
        if self.overhead:
            self.views = 1
        return self

    @property
    def top(self) -> float:
        """Absolute altitude of its top."""
        return self.base + self.height

    def describe(self) -> str:
        if self.action is TargetAction.OVERFLY:
            how = "fly over"
            if self.max_altitude_agl:
                how += f" no higher than {self.max_altitude_agl:g} m above its base"
        else:
            how = "photograph" if self.action is TargetAction.CAPTURE else "hover at"
            if self.overhead:
                how += " from directly above"
            elif self.sides:
                how += " from the " + ", ".join(s.value for s in self.sides) + " side" + ("s" if len(self.sides) > 1 else "")
            elif self.views > 1:
                how += f" from {self.views} distinct sides"
            if self.altitude_agl:
                how += f", {self.altitude_agl:g} m above its base"
        size = f"about {self.radius * 2:.0f} m across" + (f", {self.height:g} m tall" if self.height else "")
        return (
            f"{how}: '{self.name}' centred at ({self.x:.0f}, {self.y:.0f}), {size}, "
            f"ground there at {self.base:.1f} m"
        )


class Avoid(BaseModel):
    """A place the route must not pass over, at any altitude."""

    name: str
    x: float
    y: float
    radius: float = Field(gt=0)

    def describe(self) -> str:
        return f"never pass over '{self.name}': keep every leg {self.radius:.0f} m from ({self.x:.0f}, {self.y:.0f})"


class MissionIntent(BaseModel):
    """The mission as checkable requirements."""

    targets: list[Target] = Field(default_factory=list)
    ordered: bool = Field(default=False, description="Targets must be served in the order listed.")
    avoid: list[Avoid] = Field(default_factory=list)
    landing: tuple[float, float, float] | None = Field(
        default=None, description="Where the flight must end, when it is known."
    )
    notes: list[str] = Field(
        default_factory=list, description="What grounding changed or could not represent, for the record."
    )

    @property
    def empty(self) -> bool:
        return not self.targets and not self.avoid and self.landing is None

    def describe(self) -> str:
        """Prompt-ready checklist: the requirements the plan will be checked against."""
        lines = []
        for number, target in enumerate(self.targets, 1):
            lines.append(f"  {number}. {target.describe()}")
        if self.ordered and len(self.targets) > 1:
            lines.append("  Serve them in exactly this order.")
        for avoid in self.avoid:
            lines.append(f"  - {avoid.describe()}")
        if self.landing is not None:
            x, y, z = self.landing
            lines.append(f"  - end with a LAND at ({x:g}, {y:g}, {z:g})")
        return "\n".join(lines) if lines else "  (nothing beyond the safety envelope)"

    def key(self) -> tuple:
        """What two extractions must agree on to count as the same reading of a mission."""
        return (
            tuple(sorted(
                (
                    t.name.lower() if t.landmark else f"{round(t.x / 10) * 10},{round(t.y / 10) * 10}",
                    t.action.value,
                    t.views,
                    tuple(sorted(s.value for s in t.sides)),
                    t.overhead,
                    round(t.altitude_agl or 0),
                    round(t.max_altitude_agl or 0),
                )
                for t in self.targets
            )),
            self.ordered and len(self.targets) > 1,
            tuple(sorted(a.name.lower() for a in self.avoid)),
        )
