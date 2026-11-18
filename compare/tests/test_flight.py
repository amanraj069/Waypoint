"""The flight stack: PX4's offboard semantics over Waypoint's simulator."""

from __future__ import annotations

import math

from compare.common.flight import FlightStack, make_stack
from compare.common.sensors import Camera, SceneObject, lidar_scan
from waypoint.planner.models import Action, Obstacle


def airborne(z: float = 5.0) -> FlightStack:
    stack = make_stack()
    stack.set_target(0, 0, z, yaw=0.0)
    assert stack.set_mode("OFFBOARD")
    assert stack.arm(True)
    assert stack.advance_until(lambda: abs(stack.position[2] - z) <= 0.3, 30)
    return stack


def test_starts_on_the_pad_disarmed_in_loiter():
    stack = make_stack()
    assert stack.position == (0.0, 0.0, 0.0)
    assert (stack.mode, stack.armed, stack.on_ground) == ("AUTO.LOITER", False, True)


def test_offboard_is_refused_without_a_setpoint_stream():
    stack = make_stack()
    assert not stack.set_mode("OFFBOARD")
    stack.set_target(0, 0, 3)
    assert stack.set_mode("OFFBOARD")


def test_nothing_moves_until_armed():
    stack = make_stack()
    stack.set_target(0, 0, 5)
    stack.set_mode("OFFBOARD")
    stack.advance(5.0)
    assert stack.on_ground


def test_flies_to_a_setpoint_and_holds_it():
    stack = airborne()
    stack.set_target(20, -10, 5)
    assert stack.advance_until(lambda: math.dist(stack.position, (20, -10, 5)) <= 0.3, 30)
    stack.advance(10.0)
    assert math.dist(stack.position, (20, -10, 5)) <= 0.3
    assert not stack.backend.crashed


def test_yaw_slews_at_the_rate_limit():
    stack = airborne()
    stack.set_target(0, 0, 5, yaw=math.radians(90))
    stack.advance(1.0)
    assert 40 <= stack.yaw_deg <= 50  # 45 deg/s
    stack.advance(2.0)
    assert abs(stack.yaw_deg - 90) < 0.5


def test_offboard_falls_back_to_loiter_when_the_stream_stops():
    stack = airborne()
    stack.stop_stream()
    stack.advance(0.5)
    assert stack.mode == "OFFBOARD"
    stack.advance(1.0)
    assert stack.mode == "AUTO.LOITER"
    assert not stack.on_ground


def test_land_descends_and_disarms_on_touchdown():
    stack = airborne()
    assert not stack.arm(False), "PX4 refuses to disarm in the air"
    stack.stop_stream()
    assert stack.set_mode("AUTO.LAND")
    assert stack.advance_until(lambda: not stack.armed, 60)
    assert stack.on_ground and not stack.backend.crashed


def test_return_to_launch_comes_home():
    stack = airborne()
    stack.set_target(40, 30, 5)
    stack.advance(15.0)
    assert stack.set_mode("AUTO.RTL")
    assert stack.advance_until(lambda: not stack.armed, 120)
    assert math.hypot(*stack.position[:2]) < 1.0


def test_an_armed_aircraft_left_on_the_ground_disarms_itself():
    stack = make_stack()
    stack.arm(True)
    stack.advance(9.5)
    assert stack.armed
    stack.advance(1.0)
    assert not stack.armed


def test_the_local_frame_starts_at_the_pad():
    stack = make_stack(home=(50.0, -20.0, 0.0))
    assert stack.position == (0.0, 0.0, 0.0)
    assert stack.world_of((1.0, 2.0, 3.0)) == (51.0, -18.0, 3.0)
    stack.set_target(10, 0, 5)
    stack.set_mode("OFFBOARD")
    stack.arm(True)
    stack.advance(20.0)
    assert math.dist((stack.state.x, stack.state.y), (60.0, -20.0)) < 0.5


def test_marks_are_recorded_like_the_simulators_waypoints():
    stack = airborne()
    stack.mark_waypoint(Action.CAPTURE, "photo")
    event = [e for e in stack.events() if e["type"] == "WAYPOINT"][-1]
    assert event["payload"]["action"] == "CAPTURE"
    assert event["payload"]["waypoint_index"] == 0
    assert abs(event["payload"]["position"][2] - 5.0) < 0.5


def test_lidar_reads_an_obstacle_in_the_right_direction():
    obstacle = Obstacle(x=10.0, y=0.0, radius=2.0, height=30.0, name="mast")
    ranges, angle_min, increment = lidar_scan(None, [obstacle], (0.0, 0.0, 5.0), 0.0)
    ahead = ranges[round((0.0 - angle_min) / increment)]
    behind = ranges[0]
    assert abs(ahead - 8.0) < 0.05
    assert behind == math.inf
    # Facing north, the same mast is on the right: 90 degrees clockwise.
    ranges, angle_min, increment = lidar_scan(None, [obstacle], (0.0, 0.0, 5.0), math.radians(90))
    right = ranges[round((-math.pi / 2 - angle_min) / increment)]
    assert abs(right - 8.0) < 0.05
    # Above its top, a planar scan passes over it.
    ranges, _, _ = lidar_scan(None, [obstacle], (0.0, 0.0, 40.0), 0.0)
    assert all(r == math.inf for r in ranges)


def test_camera_sees_what_is_ahead_and_reads_its_depth():
    camera = Camera()
    person = SceneObject("person", 10.0, 0.0)
    behind = SceneObject("car", -10.0, 0.0)
    far = SceneObject("bench", 30.0, 0.0)
    sightings, depth = camera.observe([person, behind, far], (0.0, 0.0, 1.0), 0.0)
    labels = {s.item.label for s in sightings}
    assert labels == {"person", "bench"}
    seen = next(s for s in sightings if s.item.label == "person")
    x1, y1, x2, y2 = seen.bbox
    assert x1 < camera.cx < x2
    assert abs(float(depth[(y1 + y2) // 2, (x1 + x2) // 2]) - 10.0) < 1e-3
    # Beyond the depth sensor's range the object is in the image but has no depth.
    far_box = next(s for s in sightings if s.item.label == "bench").bbox
    assert float(depth[(far_box[1] + far_box[3]) // 2, far_box[0]]) in (0.0, 10.0)
