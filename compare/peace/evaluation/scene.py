"""The scene the evaluation prompts are flown in.

The dataset's object-referential prompts use five classes: person, car,
bench, traffic light, fire hydrant ("due to the availability of assets in the
simulation", section IV-A). The paper's Gazebo world is not part of the
release, so where those five objects stand is not known; the layout below is
this rebuild's own. One of each, 8 to 14 m from the take-off point, none in
view straight ahead at take-off, so a search prompt needs a search.
"""

from __future__ import annotations

from compare.common.sensors import SceneObject

OBJECT_VOCABULARY = ("person", "car", "bench", "traffic light", "fire hydrant")

EVALUATION_SCENE: tuple[SceneObject, ...] = (
    SceneObject("person", 4.0, 9.0, width=0.5, height=1.8),
    SceneObject("car", -9.0, 7.0, width=1.8, height=1.5),
    SceneObject("bench", -8.0, -6.0, width=1.5, height=0.9),
    SceneObject("traffic light", 3.0, -11.0, width=0.4, height=3.2),
    SceneObject("fire hydrant", 11.0, -8.0, width=0.4, height=0.9),
)
