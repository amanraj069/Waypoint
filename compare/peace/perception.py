"""PEACE's perception path: a detector, the depth projector, and the vision node's loop.

The paper, section III-C: "Perception is abstracted behind a detector interface
with two interchangeable backends: a fast 2D object detector and a
vision-language model generating structured detections." `ObjectDetector` is
that interface, as released. The two released backends (YOLO, and a VLM asked
for boxes) need rendered camera images, which Waypoint's simulator does not
produce; `SimulatedDetector` is a third backend behind the same interface,
returning the boxes a detector would for the objects a simulated camera has in
view. Everything after the detector is PEACE's own: the depth projector lifts
each box to a world position and sets the depth-validity flag, and the scene
description goes to the world model in the released JSON shape.

`VisionNode` is the released vision node's loop without ROS: every
1 / `vision_inference_rate` seconds of flight time it takes a frame, detects,
projects, and hands the result to the agent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from compare.common.flight import FlightStack
from compare.common.sensors import Camera, SceneObject
from compare.peace.config import PeaceSettings
from compare.peace.schemas import Object, SceneDescription
from compare.peace.services.depth import DepthProjectionService


@runtime_checkable
class ObjectDetector(Protocol):
    """Interface for all object detection backends (peace/backends/detector.py)."""

    def detect(self, image) -> list[Object]:
        """Run detection on a frame and return detected objects."""
        ...

    @property
    def is_ready(self) -> bool: ...

    @property
    def error_message(self) -> str | None: ...


class SimulatedFrame:
    """One frame from the simulated camera: what is in view, and the depth image."""

    def __init__(self, sightings, depth: np.ndarray):
        self.sightings = sightings
        self.depth = depth


class SimulatedDetector:
    """The detector backend for a simulator without rendered images.

    Reports each object in the frame with the confidence the scene gives it;
    a detection below `confidence_threshold` is dropped, as YOLO's `conf`
    argument drops it in the released backend.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self._conf = confidence_threshold

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def error_message(self) -> str | None:
        return None

    def detect(self, image: SimulatedFrame) -> list[Object]:
        objects: list[Object] = []
        for sighting in image.sightings:
            if sighting.item.confidence / 100.0 < self._conf:
                continue
            objects.append(
                Object(
                    id=len(objects),
                    label=sighting.item.label,
                    bbox=list(sighting.bbox),
                    confidence=round(float(sighting.item.confidence), 1),
                )
            )
        return objects


class VisionNode:
    """Detect, project to 3D, publish: the released vision node's inference loop.

    `publish` receives the scene description as the JSON string the released
    node publishes on /agent/vision/detections.
    """

    def __init__(
        self,
        stack: FlightStack,
        objects: list[SceneObject] | tuple[SceneObject, ...],
        publish: Callable[[str], None],
        settings: PeaceSettings,
        *,
        camera: Camera | None = None,
        detector: ObjectDetector | None = None,
    ) -> None:
        self.stack = stack
        self.objects = list(objects)
        self.settings = settings
        self.camera = camera or Camera()
        self._publish = publish
        self._detector = detector or SimulatedDetector(settings.yolo_confidence_threshold)
        self._depth_svc = DepthProjectionService(
            fx=self.camera.fx,
            fy=self.camera.fy,
            cx=self.camera.cx,
            cy=self.camera.cy,
            width=self.camera.width,
            height=self.camera.height,
            min_depth=settings.min_depth,
            max_depth=settings.max_depth,
            confidence_threshold=settings.vision_confidence_threshold,
        )
        self._period = 1.0 / settings.vision_inference_rate
        self._next = 0.0
        self.inferences = 0
        stack.on_odom(self._on_tick)

    def _on_tick(self, *_odom) -> None:
        if self.stack.now + 1e-9 >= self._next:
            self._next = self.stack.now + self._period
            self.run_inference()

    def run_inference(self) -> SceneDescription:
        """One inference: detect in the current frame, lift to 3D, publish."""
        stack = self.stack
        sightings, depth_img = self.camera.observe(self.objects, stack.state.position, stack.yaw, stack.world)
        pos_x, pos_y, pos_z = stack.position
        # The projector works in the agent's local frame; the camera looked from the same pose.
        objects = self._depth_svc.project_detections(
            self._detector.detect(SimulatedFrame(sightings, depth_img)), depth_img, pos_x, pos_y, pos_z, stack.yaw
        )
        valid = [o for o in objects if o.depth_valid]
        description = SceneDescription(
            summary=f"{len(valid)} object(s) with valid depth",
            objects=objects,
            camera_pos=[pos_x, pos_y, pos_z],
            img_res=[self.camera.width, self.camera.height],
        )
        self.inferences += 1
        self._publish(description.model_dump_json())
        return description


def detections_from(message: str) -> list[dict]:
    """The agent's /agent/vision/detections callback: the published JSON as world-model detections."""
    data = json.loads(message)
    detections = []
    for obj in data.get("objects", []):
        world_pos = obj.get("world_pos", [0.0, 0.0, 0.0])
        detections.append(
            {
                "id": obj.get("id", -1),
                "label": obj.get("label", "unknown"),
                "confidence": obj.get("confidence", 0) / 100.0,
                "world_x": world_pos[0] if len(world_pos) > 0 else 0.0,
                "world_y": world_pos[1] if len(world_pos) > 1 else 0.0,
                "world_z": world_pos[2] if len(world_pos) > 2 else 0.0,
                "depth_meters": obj.get("depth_meters", 0.0),
                "depth_valid": obj.get("depth_valid", False),
            }
        )
    return detections
