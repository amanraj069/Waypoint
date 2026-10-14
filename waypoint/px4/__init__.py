"""PX4 SITL in Gazebo as a flight backend (`--backend px4`).

`sitl` starts PX4 and Gazebo on a world written from a waypoint `World`,
`link` speaks MAVLink to PX4, and `backend` flies plans through them behind
the same `ExecutorBackend` interface as the built-in simulator.
"""
