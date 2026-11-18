"""Rebuilds of the published systems closest to Waypoint, for comparison.

Three papers from RELATED_WORK.md are rebuilt here from the paper and the
authors' released code, each in its own package:

    compare.peace        PEACE (Uysal, Kehrer, Panichella; arXiv:2606.00104)
    compare.autotamp     AutoTAMP (Chen et al.; ICRA 2024, arXiv:2306.06531)
    compare.droneserver  DroneServer (Ramos Silva, Burke; arXiv:2601.15486)

`compare.common` holds what they share: the LLM client and the flight stack
that stands in for MAVROS / MAVSDK over Waypoint's `ExecutorBackend`.
`compare/README.md` says what was rebuilt, how closely, and how to run it.
"""
