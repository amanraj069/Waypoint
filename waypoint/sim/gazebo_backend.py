"""`Px4GazeboBackend`: the PX4 SITL + Gazebo backend, by the name the Stage 8 spec gives it.

The implementation lives in `waypoint.px4` (`backend.py` flies plans, `link.py`
speaks MAVLink, `sitl.py` starts PX4 and Gazebo); this module only makes it
importable from where the spec looks for it. It is not imported by
`waypoint.sim`, so the custom simulator never needs PX4 or pymavlink.

    from waypoint.sim.gazebo_backend import Px4GazeboBackend
    backend = Px4GazeboBackend(world=world)       # same as make_backend("px4", world=world)
"""

from waypoint.px4.backend import Px4Backend as Px4GazeboBackend

__all__ = ["Px4GazeboBackend"]
