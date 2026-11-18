"""The same flights: Waypoint, PEACE and DroneServer on Waypoint's own missions.

    python -m compare.versus run                  # fly every mission with every system
    python -m compare.versus report <dir>         # write compare/Comp_res.md from a result directory

Every system gets the same mission text, the same world, the same launch and
landing pads, the same wind and the same fault schedule, flies the same
simulated airframe, plans with the same LLM, and is scored by the same
function against the same hand-written ground truth
(`waypoint.experiments.truth`). `scenario.py` builds what is shared,
`fly.py` holds one runner per system, `score.py` grades a flight from its run
file, and `report.py` writes the comparison.
"""
