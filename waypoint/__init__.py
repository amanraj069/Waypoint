"""waypoint: say a UAV mission in English, and have it planned, verified and flown.

The pipeline, one subpackage per stage:

    world        where it flies: terrain, buildings, landmarks
    planner      English -> hierarchical Plan, via Gemini
    verifier     deterministic safety check of a Plan against MissionConstraints
    sim          point-mass physics, wind, fault injection: flies the Plan
    feedback     in-flight observation (still a stub)
    orchestrator plan -> verify -> fly, the CLI, and run files

and two ways to look at or drive it from outside:

    gazebo       a run as a Gazebo world, the flight replayed in it
    api          the same pipeline over HTTP

`waypoint.paths` says where data, worlds and runs live on disk, and
`waypoint.scenario` turns short specs (`lodhi`, `sample`, `4,0,0`) into inputs.
"""
