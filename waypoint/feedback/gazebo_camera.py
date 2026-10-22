"""`GazeboCameraPerceptionSource`: perception from the PX4 airframe's camera. A stub, by design.

This is where perception stops being symbolic. The `px4` backend can fly
`x500_depth` (`WAYPOINT_PX4_MODEL=x500_depth`), PX4's x500 carrying an OAK-D
Lite, and Gazebo then publishes, headless or with the window, on macOS too:

    /world/waypoint/model/x500_depth_0/link/camera_link/sensor/IMX214/image   RGB, 1920 x 1080
    /depth_camera                                                              depth, 640 x 480
    /depth_camera/points                                                       point cloud

`Sitl.camera_topics()` lists them for the running flight.

The design, for when it is built:

1. **Read the frames.** Gazebo's Python bindings are built for Homebrew's
   Python, not the project's, so frames come through a bridge process like
   `waypoint/gazebo/bridge.py`: it subscribes to the RGB and depth topics under
   `.venv-gz` and writes the newest pair to a pipe or shared memory. The source
   reads the newest pair on each `get_events` call and never blocks the loop
   (the loop calls it every few ticks; a missed frame is simply no news).
2. **Detect.** A detector over the RGB frame (any model that returns 2D boxes
   with a confidence: a YOLO-class network, or a colour or shape rule for
   Gazebo's plain obstacle cylinders to start with).
3. **Place it in the world.** For each box, the median depth inside it gives
   the range; the camera intrinsics (from `camera_info`), the camera's mount on
   the airframe (`.12 .03 .242` in `x500_depth/model.sdf`) and the aircraft's
   pose from `current_state` turn the box centre into a point in world ENU.
   The box's width at that range gives a radius; the top of the box, a height.
4. **Report.** One `PerceptionEvent` per detection with an `Obstacle`, the
   same severity rule as `SymbolicPerceptionSource` (critical within 10 m,
   warning within 40 m), and `source=self.name`. Detections of the same thing
   across frames are merged before reporting, so the loop is told once, as it is
   by the symbolic source.

Nothing imports this by default: the loop's perception stays
`SymbolicPerceptionSource` unless `run_mission(perception=...)` is given this.
"""

from __future__ import annotations

from waypoint.feedback.perception import PerceptionEvent, PerceptionSource
from waypoint.sim.state import UAVState

#: The OAK-D Lite on PX4's x500_depth, as Gazebo publishes it.
RGB_TOPIC = "/world/waypoint/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEPTH_TOPIC = "/depth_camera"


class GazeboCameraPerceptionSource(PerceptionSource):
    """Obstacles detected in the x500_depth's RGB and depth frames. Not implemented yet."""

    name = "gazebo_camera"

    def __init__(
        self,
        *,
        rgb_topic: str = RGB_TOPIC,
        depth_topic: str = DEPTH_TOPIC,
        min_confidence: float = 0.5,
        max_range: float = 40.0,
    ):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.min_confidence = min_confidence
        self.max_range = max_range

    def get_events(self, current_state: UAVState) -> list[PerceptionEvent]:
        raise NotImplementedError(
            "camera perception is designed (see waypoint/feedback/gazebo_camera.py) but not built yet; "
            "use SymbolicPerceptionSource, the default"
        )
